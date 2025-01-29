#pragma once

#include <absl/time/time.h>

#include <cstdint>
#include <vector>

#include "common/slice.h"
#include "incremental_decoder.h"
#include "memory/block.h"
#include "output.h"
#include "sampling/parameters.h"
#include "stopping_criteria.h"
#include "tokenizer/tokenizer.h"

namespace llm {

// The sequence is shared between LLM and SSM for speculative decoding, and
// it's possible that the numbers of tokens in kv cache are out of sync.
// Specifying the engine type to ensure accurate updating of the the number
// tokens in kv cache separately for LLM and SSM.
enum class EngineType : int8_t {
  // LLM engine
  LLM = 0,
  // SSM engine
  SSM = 1,
  // total number of engines
  COUNT = 2,
};

struct Token {
  explicit Token(int64_t id) : id(id) {}

  int64_t id = 0;
  std::optional<float> logprob;
  Slice<int64_t> top_tokens;
  Slice<float> top_logprobs;
};

// The sequence encapsulates all the necessary
// information for a sequence, including the prompt, the token ids, and the
// current position in generating tokens, etc.
class Sequence final {
 public:
  struct Options {
    // the sampling parameters for the sequence
    SamplingParameter sampling_param;

    // the stopping criteria for the sequence
    StoppingCriteria stopping_criteria;

    // whether to skip special tokens when decoding the output
    bool skip_special_tokens = true;

    // whether to echo the prompt tokens back
    bool echo = false;

    // whether to output log probabilities for output tokens
    bool logprobs = false;
  };

  Sequence(size_t index,
           const std::string_view& prompt,
           const std::vector<int32_t>& prompt_token_ids,
           const absl::Time& created_time,
           size_t capacity,
           const Options& option);

  // simple constructor for testing
  Sequence(const std::string_view& prompt,
           const std::vector<int32_t>& prompt_token_ids,
           size_t capacity,
           const Options& option);

  Sequence(const std::vector<int32_t>& prompt_token_ids,
           size_t capacity,
           const Options& option);

  // get the index of the sequence in the request
  size_t index() const { return index_; }

  // get token ids
  Slice<int32_t> token_ids() const { return {token_ids_, num_tokens_}; }

  // get token ids to count map
  const std::unordered_map<int32_t, int32_t>& token_to_count_map() const {
    return token_to_count_map_;
  }

  // get the total number of tokens
  size_t num_tokens() const { return num_tokens_; }

  // get the number of prompt tokens
  size_t num_prompt_tokens() const { return num_prompt_tokens_; }

  // get the number of generated tokens
  // returns 0 if still in prefill stage
  size_t num_generated_tokens() const {
    return num_tokens_ - num_prompt_tokens_;
  }

  // get token ids in kv cache
  Slice<int32_t> tokens_in_kv_cache() const {
    // it is a little bit tricky to get the tokens in kv cache for speculative
    // decoding where the number of tokens in kv cache may be out of sync by at
    // most 1 between LLM and SSM.
    const size_t ssm_kv_cache_size = num_kv_cache_tokens(EngineType::SSM);
    const size_t llm_kv_cache_size = num_kv_cache_tokens(EngineType::LLM);
    CHECK_GE(llm_kv_cache_size, ssm_kv_cache_size);
    const size_t diff = llm_kv_cache_size - ssm_kv_cache_size;
    // at most one token difference between LLM and SSM for speculative decoding
    const size_t kv_cache_size =
        diff <= 1 ? ssm_kv_cache_size : llm_kv_cache_size;
    return {token_ids_, kv_cache_size};
  }

  // get the number of tokens in the kvcache
  size_t num_kv_cache_tokens() const {
    return num_kv_cache_tokens_[engine_type_];
  }

  size_t num_kv_cache_tokens(EngineType engine_type) const {
    CHECK(engine_type < EngineType::COUNT) << "Invalid engine type";
    return num_kv_cache_tokens_[static_cast<size_t>(engine_type)];
  }

  // get the capacity of the kv cache allocated
  size_t kv_cache_capacity() const;

  // generate the kv cache slots for the position range [pos_start, pos_end)
  std::vector<int32_t> kv_cache_slots(int32_t pos_start, int32_t pos_end) const;

  // get the number of tokens to process
  size_t num_tokens_to_process() const {
    return num_tokens() - num_kv_cache_tokens();
  }

  // check if the sequence is in prefill stage
  bool is_prefill_stage() const {
    return num_kv_cache_tokens() < num_prompt_tokens();
  }

  // add a new token id to the sequence and update the count
  // the token would be discarded if the sequence is still in prefill stage
  void append_token(const Token& token);
  void append_token(int64_t token_id) { append_token(Token(token_id)); }

  // set embeddings for the sequence
  void set_embeddings(std::vector<float>&& embeddings) {
    embeddings_ = std::move(embeddings);
  }
  
  // validate draft tokens with accepted tokens for speculative decoding
  // N.B. take int64_t as input to be compatible with torch::Tensor
  // returns the number of accepted tokens, including the resampled token
  size_t validate_tokens(const std::vector<Token>& tokens);
  size_t validate_tokens(const std::vector<int64_t>& token_ids);

  // whether the new added token is the first token
  bool is_first_token() const { return is_first_token_; }

  // add new cache blocks
  void append_block(const Block& new_block) {
    return append_blocks({new_block});
  }
  void append_blocks(const std::vector<Block>& new_blocks);

  // set shared cache blocks from prefix cache
  void set_shared_blocks(std::vector<Block>&& shared_blocks);

  // release all cache blocks
  void release_blocks();

  // returns allocated cache blocks
  Slice<Block> blocks() const { return blocks_; }

  // get the number of blocks
  size_t num_blocks() const { return blocks_.size(); }

  // get the reason why the sequence is finished
  FinishReason finish_reason() const { return finish_reason_; }

  // whether has pending tokens to output
  bool has_pending_tokens() const {
    return num_tokens_ > incremental_decoder_.output_offset();
  }

  // check finish status, use cached value if not invalidated
  bool is_finished() const;

  // get the output of the sequence until the specified number of tokens,
  // returns nullopt if no delta text and not finished
  std::optional<SequenceOutput> build_delta_output_until(
      size_t size,
      const Tokenizer& tokenizer);

  // get the full output of the sequence
  SequenceOutput build_output(const Tokenizer& tokenizer);

  // set engine type this sequence is used for
  void set_engine_type(EngineType engine_type) {
    CHECK(engine_type < EngineType::COUNT) << "Invalid engine type.";
    engine_type_ = static_cast<size_t>(engine_type);
  }

  // commit the kv cache by n tokens
  void commit_kv_cache(size_t size) {
    size_t& num_kv_cache_tokens = num_kv_cache_tokens_[engine_type_];
    CHECK(num_kv_cache_tokens + size <= kv_cache_capacity());
    num_kv_cache_tokens += size;
  }

  // get the sampling parameters
  const SamplingParameter* sampling_param() const {
    return &options_.sampling_param;
  }

  // get the stopping criteria
  const StoppingCriteria* stopping_criteria() const {
    return &options_.stopping_criteria;
  }

  // close the sequence once all outputs have been sent
  void close() { closed_ = true; }

  bool is_closed() const { return closed_; }

  // get the inter-token latency
  double inter_token_latency(const absl::Time& now);

  // get the average log probability of the sequence (generated tokens only)
  float logprob() const;

 private:
  // build log probabilities for the tokens in the range [start_idx, end_idx)
  std::vector<LogProb> build_logprobs(size_t start_idx,
                                      size_t end_idx,
                                      const Tokenizer& tokenizer);

  void update_logprobs(size_t index, const Token& token);

  // the index of the sequence in the request
  size_t index_ = 0;

  // last token generation time
  absl::Time last_token_time_;

  // whether the added token is the first generated token
  bool is_first_token_ = false;

  // options for the sequence
  Options options_;

  // incremental decoder to decode the tokens
  IncrementalDecoder incremental_decoder_;

  // token ids generated for the sequence
  std::vector<int32_t> token_ids_;

  // log probabilities of the sequence
  std::vector<std::optional<float>> logprobs_;

  // top k log probabilities of the sequence
  std::vector<std::vector<int64_t>> top_tokens_;
  std::vector<std::vector<float>> top_logprobs_;

  // sequence embeddings
  std::optional<std::vector<float>> embeddings_;

  // number of tokens in the sequence
  size_t num_tokens_ = 0;

  // the count of each token id
  std::unordered_map<int32_t, int32_t> token_to_count_map_;

  // the length of the prompt tokens
  size_t num_prompt_tokens_ = 0;

  // number of tokens in kv cache
  std::vector<size_t> num_kv_cache_tokens_;
  // current using engine type
  size_t engine_type_ = 0;

  // physical blocks that hold the kv cache.
  std::vector<Block> blocks_;

  // is the sequence finished
  mutable bool is_finished_ = false;

  // is the finish status invalidated
  mutable bool finish_status_invalidated_ = true;

  // the reason why the sequence is finished
  mutable FinishReason finish_reason_ = FinishReason::NONE;

  // is the sequence closed.
  bool closed_ = false;
};

}  // namespace llm
