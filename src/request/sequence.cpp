#include "sequence.h"

#include <absl/strings/match.h>

#include <cstdint>
#include <string>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace llm {
namespace {
// Returns whether a given `sequence` ends with `suffix`.
inline bool sequence_end_withs(const Slice<int32_t>& sequence,
                               const Slice<int32_t>& suffix) noexcept {
  return suffix.empty() ||
         (sequence.size() >= suffix.size() &&
          memcmp(sequence.data() + (sequence.size() - suffix.size()),
                 suffix.data(),
                 suffix.size() * sizeof(int32_t)) == 0);
}
}  // namespace

// NOLINTNEXTLINE
std::atomic<int64_t> Sequence::next_id_{1};

Sequence::Sequence(const std::vector<int32_t>& token_ids,
                   const SamplingParameter& sampling_param,
                   const StoppingCriteria& stopping_criteria,
                   bool echo,
                   OnDelta on_delta)
    : Sequence("",
               token_ids,
               sampling_param,
               stopping_criteria,
               echo,
               on_delta) {}

Sequence::Sequence(const std::string_view& prompt,
                   const std::vector<int32_t>& prompt_token_ids,
                   const SamplingParameter& sampling_param,
                   const StoppingCriteria& stopping_criteria,
                   bool echo,
                   OnDelta on_delta)
    : prompt_(prompt),
      id_(next_id_.fetch_add(1)),
      sampling_param_(sampling_param),
      stopping_criteria_(stopping_criteria),
      on_delta_(on_delta) {
  CHECK(!prompt_token_ids.empty()) << "empty prompt token ids";

  // allocate space for the token ids and add the prompt tokens
  const size_t max_tokens = stopping_criteria.max_tokens;
  token_ids_.resize(max_tokens + prompt_token_ids.size());
  for (const auto token_id : prompt_token_ids) {
    token_ids_[num_tokens_++] = token_id;
    token_to_count_map_[token_id]++;
  }
  num_prompt_tokens_ = num_tokens_;
  // if echo is true, set prefix_offset_ and output_offset_ to 0 to print the
  // whole sequence, otherwise set them to the length of the prompt to skip the
  // prompt.
  prefix_offset_ = echo ? 0 : num_prompt_tokens_;
  output_offset_ = echo ? 0 : num_prompt_tokens_;

  kv_cache_pos_[0] = 0;
  kv_cache_pos_[1] = 0;
}

void Sequence::append_new_token_id(int32_t next_token_id) {
  CHECK(!is_finished_) << "cannot append token to a finished sequence";
  CHECK(!is_prefill_stage()) << "cannot append token to a prefill sequence";

  // append the token id and update the token count
  token_ids_[num_tokens_++] = next_token_id;
  ++token_to_count_map_[next_token_id];

  // reset the finish status once a new token is appended
  finish_status_invalidated_ = true;
}

void Sequence::validate_token_ids(const Slice<int32_t>& accpeted_token_ids) {
  const size_t len = accpeted_token_ids.size();
  CHECK_GE(num_tokens_, len) << "accepted tokens exceed the sequence length";

  // validate the accepted tokens with draft tokens, stop at the first mismatch
  const size_t base_idx = num_tokens_ - len;
  size_t i = 0;
  for (; i < len; ++i) {
    const int32_t token_id = accpeted_token_ids[i];
    // stop at the first mismatch
    if (token_id != token_ids_[base_idx + i]) {
      // overwrite the token id with the accepted token id
      token_ids_[base_idx + i] = token_id;
      ++token_to_count_map_[token_id];

      // update num tokens
      num_tokens_ = base_idx + i + 1;
      break;
    }

    // check if sequence is finished
    if (check_finished(base_idx + i)) {
      // update num tokens, including the last token
      num_tokens_ = base_idx + i + 1;
      break;
    }
  }

  // adjust the token count for mismatched tokens
  for (; i < len; ++i) {
    const auto token_id = token_ids_[base_idx + i];
    --token_to_count_map_[token_id];
  }

  // adjust kv cache position
  kv_cache_pos_[0] = std::min(kv_cache_pos_[0], num_tokens_ - 1);
  kv_cache_pos_[1] = std::min(kv_cache_pos_[1], num_tokens_ - 1);
}

// get token ids in kv cache
Slice<int32_t> Sequence::tokens_in_kv_cache() const {
  return {token_ids_, kv_cache_pos()};
}

// get the number of tokens in the kvcache
size_t Sequence::num_tokens_in_kv_cache() const { return kv_cache_pos(); }

// decode the sequence to get delta text using the tokenizer
std::string Sequence::decode_delta_text(size_t end,
                                        const Tokenizer& tokenizer) {
  // return prompt directly if prompt string is not empty
  if (output_offset_ < num_prompt_tokens_ && !prompt_.empty()) {
    // leave 6 tokens for the prefix to defeat cleanup algorithms in decode
    // which decide to add a space or not depending on the surrouding ids.
    prefix_offset_ = num_prompt_tokens_ <= 6 ? 0 : num_prompt_tokens_ - 6;
    output_offset_ = num_prompt_tokens_;
    return std::string(prompt_);
  }

  const auto tokens = token_ids();
  const auto prefix_text =
      tokenizer.decode(tokens.slice(prefix_offset_, output_offset_));
  const auto new_text = tokenizer.decode(tokens.slice(prefix_offset_, end));
  // utf-8 char � at the end means it is a potential unfinished byte sequence
  // from byte fallback tokenization.
  if (new_text.size() > prefix_text.size() && !absl::EndsWith(new_text, "�")) {
    prefix_offset_ = output_offset_;
    output_offset_ = end;
    // only print the delta text
    return new_text.substr(prefix_text.size());
  }
  return "";
}

size_t Sequence::num_generated_tokens() const {
  const size_t n_tokens = num_tokens();
  const size_t n_prompt_tokens = num_prompt_tokens();
  return (n_tokens <= n_prompt_tokens) ? 0 : n_tokens - n_prompt_tokens;
}

void Sequence::append_blocks(const std::vector<Block>& new_blocks) {
  blocks_.insert(blocks_.end(), new_blocks.begin(), new_blocks.end());
}

// append shared cache blocks from prefix cache
void Sequence::append_shared_blocks(const std::vector<Block>& shared_blocks) {
  CHECK(blocks_.empty()) << "shared blocks should be appended before any "
                            "other blocks";
  if (shared_blocks.empty()) {
    return;
  }
  // update the kv cache position
  const size_t block_size = shared_blocks[0].size();
  size_t kv_cache_pos = shared_blocks.size() * block_size;
  blocks_.insert(blocks_.end(), shared_blocks.begin(), shared_blocks.end());

  // It is possible that kv_cache_pos == num_prompt_tokens_, indicating that
  // the exact same prompt has been received again. In this case, it becomes
  // necessary to adjust the kv cache position to the previous token, allowing
  // the model proceed. While the shared blocks should be immutable ideally, but
  // it remains safe to regenerate the kv cache in this context, given the
  // utiliztion of the exact same token.
  if (kv_cache_pos == num_prompt_tokens_) {
    kv_cache_pos -= 1;
  }
  CHECK(kv_cache_pos < num_prompt_tokens_);
  // update the kv cache position for both engines
  kv_cache_pos_[0] = kv_cache_pos;
  kv_cache_pos_[1] = kv_cache_pos;
}

// release all cache blocks
void Sequence::release_blocks() {
  // reset the kv cache position to 0
  kv_cache_pos_[0] = 0;
  kv_cache_pos_[1] = 0;
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

void Sequence::commit_kv_cache(size_t size) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
  size_t& kv_cache_pos = kv_cache_pos_[engine_type_];
  CHECK(kv_cache_pos + size <= kv_cache_capacity());
  kv_cache_pos += size;
}

void Sequence::rewind_kv_cache(size_t size) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
  size_t& kv_cache_pos = kv_cache_pos_[engine_type_];
  CHECK(kv_cache_pos >= size);
  kv_cache_pos -= size;
}

void Sequence::stream_delta(const std::string& delta, FinishReason reason) {
  if (on_delta_) {
    if (!on_delta_(delta, reason)) {
      LOG(ERROR) << "failed to stream the delta";
      // TODO: handle the failure
    }
  }
}

bool Sequence::is_finished() const {
  // return the cached finish status
  if (!finish_status_invalidated_) {
    return is_finished_;
  }
  CHECK_GT(num_tokens_, 0) << "empty sequence";
  return check_finished(num_tokens_ - 1);
}

bool Sequence::check_finished(size_t last_token_idx) const {
  // reset the finish status invalidation flag
  finish_status_invalidated_ = false;

  const auto last_token_id = token_ids_[last_token_idx];
  if (!stopping_criteria_.ignore_eos_token &&
      last_token_id == stopping_criteria_.eos_token_id) {
    finish_reason_ = FinishReason::STOP;
    return is_finished_ = true;
  }
  // check against stop tokens ids
  if (stopping_criteria_.stop_token_ids.count(last_token_id) > 0) {
    finish_reason_ = FinishReason::STOP;
    return is_finished_ = true;
  }

  // check against stop sequences after adding the token
  for (const auto& stop_sequence : stopping_criteria_.stop_sequences) {
    if (stop_sequence.back() == last_token_id &&
        sequence_end_withs(token_ids(), stop_sequence)) {
      finish_reason_ = FinishReason::STOP;
      return is_finished_ = true;
    }
  }

  // check against max tokens
  const size_t max_new_tokens = stopping_criteria_.max_tokens;
  if (max_new_tokens > 0 && num_generated_tokens() >= max_new_tokens) {
    finish_reason_ = FinishReason::LENGTH;
    return is_finished_ = true;
  }
  return is_finished_ = false;
}

}  // namespace llm
