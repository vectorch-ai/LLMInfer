#include "sequence.h"

#include <absl/strings/match.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "common/slice.h"
#include "tokenizer/tokenizer.h"

namespace llm {

// NOLINTNEXTLINE
std::atomic<int64_t> Sequence::next_id_{1};

Sequence::Sequence(size_t index,
                   const std::string_view& prompt,
                   const std::vector<int32_t>& prompt_token_ids,
                   const absl::Time& created_time,
                   size_t capacity,
                   const Options& option)
    : id_(next_id_.fetch_add(1)),
      index_(index),
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

void Sequence::append_token(const TokenInfo& token_info) {
  CHECK(num_tokens_ < token_ids_.size())
      << "exceed the token capacity of the sequence";
  CHECK(!is_finished_) << "cannot append token to a finished sequence";
  CHECK(!is_prefill_stage()) << "cannot append token to a prefill sequence";

  // check if the token is the first token after the prompt
  is_first_token_ = num_tokens_ == num_prompt_tokens_;

  // append the token id and update the token count
  const auto cur_idx = num_tokens_++;
  const int32_t token_id = token_info.token_id;
  token_ids_[cur_idx] = token_id;

  // update logprobs if needed
  if (options_.sampling_param.logprobs) {
    logprobs_[cur_idx] = token_info.logprob;
  }

  // update top tokens and top logprobs if needed
  const auto num_top_tokens = options_.sampling_param.top_logprobs;
  DCHECK_EQ(token_info.top_tokens.size(), token_info.top_logprobs.size());
  if (num_top_tokens > 0) {
    if (token_info.top_tokens.size() > num_top_tokens) {
      top_tokens_[cur_idx] = token_info.top_tokens.slice(0, num_top_tokens);
      top_logprobs_[cur_idx] = token_info.top_logprobs.slice(0, num_top_tokens);
    } else {
      DCHECK_EQ(token_info.top_tokens.size(), num_top_tokens);
      top_tokens_[cur_idx] = token_info.top_tokens;
      top_logprobs_[cur_idx] = token_info.top_logprobs;
    }
  }

  token_to_count_map_[token_id]++;

  // invalidate the finish status once a new token is appended
  finish_status_invalidated_ = true;
}

size_t Sequence::validate_tokens(const Slice<int64_t>& accpeted_token_ids) {
  const size_t len = accpeted_token_ids.size();
  CHECK_GT(len, 0) << "empty accepted token ids";
  CHECK_GT(num_tokens_, len) << "accepted tokens exceed the sequence length";
  const auto bonus_token_id = accpeted_token_ids.back();
  CHECK(bonus_token_id == -1 || bonus_token_id == token_ids().back())
      << "bonus token mismatch with the last token";

  // validate the accepted tokens with draft tokens, stop at the first mismatch
  const size_t start_idx = num_tokens_ - len;

  // check if the token is the first token after the prompt
  is_first_token_ = start_idx == num_prompt_tokens_;

  bool mismatch = false;
  size_t num_accpeted = 0;
  for (size_t i = 0; i < len; ++i) {
    const size_t cur_idx = start_idx + i;
    const int32_t draft_token_id = token_ids_[cur_idx];
    const int32_t target_token_id = static_cast<int32_t>(accpeted_token_ids[i]);

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

std::optional<SequenceOutput> Sequence::build_delta_output_until(
    size_t end_idx,
    const Tokenizer& tokenizer) {
  CHECK_LE(end_idx, num_tokens_);
  const auto ids = Slice<int32_t>(token_ids_, end_idx);

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

  // prepare logprobs and top tokens if available
  const size_t end = incremental_decoder_.output_offset();
  // output logprobs for tokens [start_idx, end_idx)
  if (start < end) {
    auto logprob_contents = build_logprobs(start, end, tokenizer);
    if (!logprob_contents.empty()) {
      output.logprobs = std::move(logprob_contents);
    }
  }
  return output;
}

std::optional<SequenceOutput> Sequence::build_output(
    const Tokenizer& tokenizer) {
  const auto ids = token_ids();
  const size_t size = ids.size();
  // leave 6 tokens for potential unfinished byte sequence from byte fallback
  // tokenization
  size_t start_idx = size <= 7 ? 0 : size - 7;
  // at least start from the first generated token
  if (start_idx < num_prompt_tokens_) {
    start_idx = num_prompt_tokens_;
  }
  std::stringstream ss;
  // first output leading tokens
  ss << incremental_decoder_.decode(ids.slice(0, start_idx), tokenizer);
  // then decode one by one to avoid potential unfinished bytes
  // incrementally decode tokens [start_idx, size)
  for (size_t i = start_idx; i < size; ++i) {
    ss << incremental_decoder_.decode(ids.slice(0, i + 1), tokenizer);
  }

  SequenceOutput output;
  output.index = index_;
  output.text = ss.str();
  if (finish_reason_ != FinishReason::NONE) {
    output.finish_reason = to_string(finish_reason_);
  }

  // build logprobs for generated tokens
  auto logprob_contents = build_logprobs(0, size, tokenizer);
  if (!logprob_contents.empty()) {
    output.logprobs = std::move(logprob_contents);
  }

  return output;
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

          logprob.token = tokenizer.decode(std::vector<int32_t>{top_token_id},
                                           options_.skip_special_tokens);
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

  // It is possible that num_shared_tokens == num_prompt_tokens_, indicating
  // that the exact same prompt has been received again. In this case, it
  // becomes necessary to adjust the kv cache position to the previous token,
  // allowing the model proceed. While the shared blocks should be immutable
  // ideally, but it remains safe to regenerate the kv cache in this context,
  // given the utiliztion of the exact same token.
  if (num_shared_tokens == num_prompt_tokens_) {
    num_shared_tokens -= 1;
  }
  CHECK(num_shared_tokens < num_prompt_tokens_);
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

}  // namespace llm
