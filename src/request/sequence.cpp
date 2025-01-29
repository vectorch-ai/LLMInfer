#include "sequence.h"

#include <absl/strings/match.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "common/metrics.h"
#include "common/slice.h"
#include "tokenizer/tokenizer.h"

DEFINE_COUNTER_FAMILY(detokenization_latency_seconds,
                      "Latency of detokenization in seconds");
DEFINE_COUNTER_INSTANCE(stream_decode_latency_seconds,
                        detokenization_latency_seconds,
                        {{"mode", "stream"}});
DEFINE_COUNTER_INSTANCE(non_stream_decode_latency_seconds,
                        detokenization_latency_seconds,
                        {{"mode", "non-stream"}});

namespace llm {

Sequence::Sequence(size_t index,
                   const std::string_view& prompt,
                   const std::vector<int32_t>& prompt_token_ids,
                   const absl::Time& created_time,
                   size_t capacity,
                   const Options& option)
    : index_(index),
      last_token_time_(created_time),
      options_(option),
      incremental_decoder_(prompt,
                           prompt_token_ids.size(),
                           option.echo,
                           option.skip_special_tokens),
      num_kv_cache_tokens_(static_cast<size_t>(EngineType::COUNT), 0) {
  CHECK(!prompt_token_ids.empty()) << "empty prompt token ids";
  CHECK_GT(capacity, prompt_token_ids.size()) << "capacity too small";

  num_prompt_tokens_ = prompt_token_ids.size();
  // allocate space for token ids, logprobs, top tokens and top logprobs
  token_ids_.resize(capacity);
  logprobs_.resize(capacity);
  top_tokens_.resize(capacity);
  top_logprobs_.resize(capacity);

  // add the prompt tokens
  for (const auto token_id : prompt_token_ids) {
    token_ids_[num_tokens_++] = token_id;
    token_to_count_map_[token_id]++;
  }
}

Sequence::Sequence(const std::string_view& prompt,
                   const std::vector<int32_t>& prompt_token_ids,
                   size_t capacity,
                   const Options& option)
    : Sequence(0, prompt, prompt_token_ids, absl::Now(), capacity, option) {}

Sequence::Sequence(const std::vector<int32_t>& prompt_token_ids,
                   size_t capacity,
                   const Options& option)
    : Sequence("", prompt_token_ids, capacity, option) {}

void Sequence::append_token(const Token& token) {
  CHECK(num_tokens_ < token_ids_.size())
      << "exceed the token capacity of the sequence";
  CHECK(!is_finished_) << "cannot append token to a finished sequence";
  CHECK(!is_prefill_stage()) << "cannot append token to a prefill sequence";

  // check if the token is the first token after the prompt
  is_first_token_ = num_tokens_ == num_prompt_tokens_;

  // append the token id and update the token count
  const auto cur_idx = num_tokens_++;
  const int32_t token_id = static_cast<int32_t>(token.id);
  token_ids_[cur_idx] = token_id;
  token_to_count_map_[token_id]++;
  // update logprobs if needed
  if (options_.sampling_param.logprobs) {
    update_logprobs(cur_idx, token);
  }

  // invalidate the finish status once a new token is appended
  finish_status_invalidated_ = true;
}

size_t Sequence::validate_tokens(const std::vector<Token>& tokens) {
  const size_t len = tokens.size();
  CHECK_GT(len, 0) << "empty accepted token ids";
  CHECK_GT(num_tokens_, len) << "accepted tokens exceed the sequence length";
  const auto bonus_token_id = tokens.back().id;
  CHECK(bonus_token_id == -1 || bonus_token_id == token_ids().back())
      << "bonus token mismatch with the last token";

  // validate the accepted tokens with draft tokens, stop at the first mismatch
  const size_t start_idx = num_tokens_ - len;

  // check if the token is the first token after the prompt
  is_first_token_ = start_idx == num_prompt_tokens_;

  bool mismatch = false;
  size_t num_accpeted = 0;
  for (size_t i = 0; i < len; ++i) {
    const auto& token = tokens[i];
    const size_t cur_idx = start_idx + i;
    const int32_t draft_token_id = token_ids_[cur_idx];
    const int32_t target_token_id = static_cast<int32_t>(token.id);

    // stop at first mismatch or rejected token
    if (mismatch || target_token_id == -1) {
      num_tokens_ = cur_idx;
      break;
    }
    ++num_accpeted;
    mismatch = target_token_id != draft_token_id;
    if (mismatch) {
      // overwrite the token id with the accepted token id
      token_ids_[cur_idx] = target_token_id;
      // update the token count
      --token_to_count_map_[draft_token_id];
      ++token_to_count_map_[target_token_id];
    }
    // update logprobs if needed
    if (options_.sampling_param.logprobs) {
      update_logprobs(cur_idx, token);
    }

    // check if sequence is finished
    const Slice<int32_t> token_ids(token_ids_, cur_idx + 1);
    auto finish_reason = options_.stopping_criteria.check_finished(
        token_ids, num_prompt_tokens_);
    if (finish_reason != FinishReason::NONE) {
      finish_reason_ = finish_reason;
      is_finished_ = true;
      // update num tokens, including current token
      num_tokens_ = cur_idx + 1;
      break;
    }
  }

  // adjust the token count for remaining discarded tokens
  for (size_t i = num_accpeted; i < len; ++i) {
    --token_to_count_map_[token_ids_[start_idx + i]];
  }

  // adjust kv cache position
  // num_tokens must be at least one more than num_kv_cache_tokens
  for (auto& num_kv_cache_tokens : num_kv_cache_tokens_) {
    num_kv_cache_tokens = std::min(num_kv_cache_tokens, num_tokens_ - 1);
  }

  CHECK_GT(num_accpeted, 0) << "no token accepted";

  // the finish status is valid after the validation
  finish_status_invalidated_ = false;
  return num_accpeted;
}

size_t Sequence::validate_tokens(const std::vector<int64_t>& token_ids) {
  std::vector<Token> tokens;
  tokens.reserve(token_ids.size());
  for (int64_t token_id : token_ids) {
    tokens.emplace_back(token_id);
  }
  return validate_tokens(tokens);
}

std::optional<SequenceOutput> Sequence::build_delta_output_until(
    size_t size,
    const Tokenizer& tokenizer) {
  CHECK_LE(size, num_tokens_);
  AUTO_COUNTER(stream_decode_latency_seconds);

  const auto ids = Slice<int32_t>(token_ids_, size);

  // record the start index of token ids
  const size_t start = incremental_decoder_.output_offset();
  auto delta = incremental_decoder_.decode(ids, tokenizer);
  if (delta.empty() && finish_reason_ == FinishReason::NONE) {
    // no delta text and not finished
    return std::nullopt;
  }

  SequenceOutput output;
  output.index = index_;
  output.text = std::move(delta);
  if (finish_reason_ != FinishReason::NONE) {
    output.finish_reason = to_string(finish_reason_);
  }

  const size_t end = incremental_decoder_.output_offset();
  output.token_ids = ids.slice(start, end);

  // prepare logprobs and top tokens if available
  if (options_.logprobs && start < end) {
    // output logprobs for tokens [start_idx, end_idx)
    auto logprob_contents = build_logprobs(start, end, tokenizer);
    if (!logprob_contents.empty()) {
      output.logprobs = std::move(logprob_contents);
    }
  }
  return output;
}

SequenceOutput Sequence::build_output(const Tokenizer& tokenizer) {
  AUTO_COUNTER(non_stream_decode_latency_seconds);

  // return embeddings if available
  if (this->embeddings_.has_value()) {
    SequenceOutput output;
    output.index = index_;
    output.embeddings = std::move(this->embeddings_);
    return output;
  }

  const auto ids = token_ids();
  const size_t size = ids.size();

  // record the start index of token ids
  const size_t start = incremental_decoder_.output_offset();

  // decide which position to start incremental decoding
  // leave 6 tokens for potential unfinished byte sequence
  size_t incremental_start = size <= 6 ? 0 : size - 6;
  // at least start from the first generated token
  if (incremental_start < num_prompt_tokens_) {
    incremental_start = num_prompt_tokens_;
  }
  // incrementally decode tokens between [incremental_start, size)
  std::stringstream ss;
  for (size_t end = incremental_start; end <= size; ++end) {
    ss << incremental_decoder_.decode(ids.slice(0, end), tokenizer);
  }

  SequenceOutput output;
  output.index = index_;
  output.text = ss.str();

  if (finish_reason_ != FinishReason::NONE) {
    output.finish_reason = to_string(finish_reason_);
  }

  const size_t end = incremental_decoder_.output_offset();
  output.token_ids = ids.slice(start, end);

  // build logprobs for generated tokens
  if (options_.logprobs) {
    auto logprob_contents = build_logprobs(start, end, tokenizer);
    if (!logprob_contents.empty()) {
      output.logprobs = std::move(logprob_contents);
    }
  }

  return output;
}

void Sequence::append_blocks(const std::vector<Block>& new_blocks) {
  blocks_.insert(blocks_.end(), new_blocks.begin(), new_blocks.end());
}

// append shared cache blocks from prefix cache
void Sequence::set_shared_blocks(std::vector<Block>&& shared_blocks) {
  CHECK(blocks_.empty()) << "shared blocks should be appended before any "
                            "other blocks";
  if (shared_blocks.empty()) {
    return;
  }

  blocks_ = std::move(shared_blocks);

  // update the kv cache position
  size_t num_shared_tokens = blocks_.size() * blocks_[0].size();

  // It is possible that num_shared_tokens == num_tokens_, indicating
  // that the exact same prompt has been received again. In this case, it
  // becomes necessary to adjust the kv cache position to the previous token,
  // allowing the model proceed. While the shared blocks should be immutable
  // ideally, but it remains safe to regenerate the kv cache in this context,
  // given the utiliztion of the exact same token.
  if (num_shared_tokens == num_tokens_) {
    num_shared_tokens -= 1;
  }
  CHECK_LT(num_shared_tokens, num_tokens_);
  // update the kv cache position
  std::fill(num_kv_cache_tokens_.begin(),
            num_kv_cache_tokens_.end(),
            num_shared_tokens);
}

// release all cache blocks
void Sequence::release_blocks() {
  // reset the kv cache position to 0
  std::fill(num_kv_cache_tokens_.begin(), num_kv_cache_tokens_.end(), 0);
  blocks_.clear();
}

size_t Sequence::kv_cache_capacity() const {
  if (blocks_.empty()) {
    return 0;
  }
  // all blocks have the same size
  const size_t block_size = blocks_[0].size();
  return blocks_.size() * block_size;
}

std::vector<int32_t> Sequence::kv_cache_slots(int32_t pos_start,
                                              int32_t pos_end) const {
  CHECK(!blocks_.empty()) << "no cache blocks available";

  std::vector<int32_t> slots;
  slots.reserve(pos_end - pos_start);

  const size_t block_size = blocks_[0].size();
  for (int32_t i = pos_start; i < pos_end; ++i) {
    const int32_t block_id = blocks_[i / block_size].id();
    const int32_t block_offset = i % block_size;
    slots.push_back(block_id * block_size + block_offset);
  }
  return slots;
}

bool Sequence::is_finished() const {
  // return the cached finish status
  if (!finish_status_invalidated_) {
    return is_finished_;
  }

  // reset the finish status invalidation flag
  finish_status_invalidated_ = false;

  auto finish_reason = options_.stopping_criteria.check_finished(
      token_ids(), num_prompt_tokens_);
  if (finish_reason != FinishReason::NONE) {
    finish_reason_ = finish_reason;
    is_finished_ = true;
    return true;
  }
  return false;
}

double Sequence::inter_token_latency(const absl::Time& now) {
  const double latency = absl::ToDoubleSeconds(now - last_token_time_);
  last_token_time_ = now;
  return latency;
}

float Sequence::logprob() const {
  if (num_tokens_ <= num_prompt_tokens_) {
    // return a small value for empty sequence
    return -9999.0;
  }

  double sum = 0.0;
  for (size_t i = num_prompt_tokens_; i < num_tokens_; ++i) {
    if (logprobs_[i].has_value()) {
      sum += logprobs_[i].value();
    }
  }
  return static_cast<float>(sum / (num_tokens_ - num_prompt_tokens_));
}

std::vector<LogProb> Sequence::build_logprobs(size_t start_idx,
                                              size_t end_idx,
                                              const Tokenizer& tokenizer) {
  // TODO: support logprobs for the entire sequence?
  if (start_idx < num_prompt_tokens_) {
    start_idx = num_prompt_tokens_;
  }

  std::vector<LogProb> logprob_contents;
  for (size_t i = start_idx; i < end_idx; ++i) {
    if (logprobs_[i].has_value()) {
      const int32_t token_id = token_ids_[i];
      auto token = tokenizer.decode(std::vector<int32_t>{token_id},
                                    options_.skip_special_tokens);
      // skip empty token
      if (token.empty()) {
        continue;
      }

      LogProb logprob_content;
      if (absl::EndsWith(token, "�")) {
        token = tokenizer.id_to_token(token_id);
        logprob_content.finished_token = false;
      }

      // add token and logprob
      logprob_content.token = std::move(token);
      logprob_content.token_id = token_id;
      logprob_content.logprob = logprobs_[i].value();

      // add top logprobs if available
      if (!top_tokens_[i].empty()) {
        const auto& top_tokens = top_tokens_[i];
        const auto& top_logprobs = top_logprobs_[i];
        DCHECK_EQ(top_tokens.size(), top_logprobs.size());
        std::vector<LogProbData> logprobs;
        for (size_t j = 0; j < top_tokens.size(); ++j) {
          LogProbData logprob;
          const int32_t top_token_id = top_tokens[j];
          const float top_logprob = top_logprobs[j];

          auto top_token = tokenizer.decode(std::vector<int32_t>{top_token_id},
                                            options_.skip_special_tokens);
          if (absl::EndsWith(top_token, "�")) {
            top_token = tokenizer.id_to_token(top_token_id);
            logprob.finished_token = false;
          }

          logprob.token = top_token;
          logprob.token_id = top_token_id;
          logprob.logprob = top_logprob;
          logprobs.push_back(std::move(logprob));
        }
        logprob_content.top_logprobs = std::move(logprobs);
      }
      logprob_contents.push_back(std::move(logprob_content));
    }
  }
  return logprob_contents;
}

void Sequence::update_logprobs(size_t index, const Token& token) {
  logprobs_[index] = token.logprob;

  // update top tokens and top logprobs if needed
  const auto num_top_tokens = options_.sampling_param.top_logprobs;
  if (num_top_tokens > 0) {
    DCHECK_EQ(token.top_tokens.size(), token.top_logprobs.size());
    if (token.top_tokens.size() > num_top_tokens) {
      top_tokens_[index] = token.top_tokens.slice(0, num_top_tokens);
      top_logprobs_[index] = token.top_logprobs.slice(0, num_top_tokens);
    } else {
      DCHECK_EQ(token.top_tokens.size(), num_top_tokens);
      top_tokens_[index] = token.top_tokens;
      top_logprobs_[index] = token.top_logprobs;
    }
  }
}

}  // namespace llm
