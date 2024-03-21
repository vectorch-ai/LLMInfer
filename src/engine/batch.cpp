#include "batch.h"

#include <torch/torch.h>

#include <vector>

#include "models/parameters.h"
#include "request/sequence.h"

namespace llm {

namespace {

template <typename T>
torch::Tensor create_2d_tensor(std::vector<std::vector<T>>& vec,
                               size_t max_col_size,
                               torch::ScalarType dtype,
                               T pad_value = 0) {
  const int64_t n_rows = vec.size();
  auto tensor =
      torch::empty({n_rows, static_cast<int64_t>(max_col_size)}, dtype);
  for (int64_t i = 0; i < n_rows; ++i) {
    vec[i].resize(max_col_size, pad_value);
    tensor[i] = torch::tensor(vec[i], dtype);
  }
  return tensor;
}

}  // namespace

Batch::Batch(Sequence* sequence) { add(sequence); }
Batch::Batch(const std::vector<Sequence*>& sequences) { add(sequences); }

void Batch::reset(const std::vector<Sequence*>& sequences) {
  clear();
  add(sequences);
}

void Batch::add(Sequence* sequence, uint32_t max_tokens_to_process) {
  DCHECK(sequence != nullptr);
  sequences_.push_back(sequence);
  max_tokens_to_process_.push_back(max_tokens_to_process);
}

void Batch::add(const std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    add(sequence);
  }
}

void Batch::clear() {
  sequences_.clear();
  max_tokens_to_process_.clear();
}

// prepare inputs for the batch
ModelInput Batch::prepare_model_inputs() {
  ModelInput model_inputs;

  // flatten the token ids and positions
  std::vector<int32_t> flatten_tokens_vec;
  std::vector<int32_t> flatten_positions_vec;
  // track the last token index in the flattened tokens
  std::vector<int32_t> last_token_idxes;

  // track the token ids and counts in the batch
  std::vector<std::vector<int64_t>> token_ids_vec;
  std::vector<int32_t> token_ids_lens_vec;
  std::vector<std::vector<int32_t>> token_counts_vec;
  size_t max_unique_tokens = 0;

  // process prefill requests
  bool all_prefill_sequences = true;
  uint32_t max_seq_len = 0;
  uint32_t q_max_seq_len = 0;
  std::vector<int32_t> cu_seq_lens = {0};
  std::vector<int32_t> q_cu_seq_lens = {0};
  // slot ids for new token
  std::vector<int32_t> new_token_slot_ids;
  std::vector<std::vector<int32_t>> block_tables_vec;
  int32_t max_block_table_len = 0;
  const int32_t num_sequences = static_cast<int32_t>(sequences_.size());
  for (int32_t i = 0; i < num_sequences; ++i) {
    auto* sequence = sequences_[i];
    CHECK(!sequence->is_finished());

    all_prefill_sequences &= sequence->is_prefill();

    const auto seq_token_ids = sequence->token_ids();
    // TODO: update the logic to handle chunked prefill sequences
    const uint32_t seq_len = seq_token_ids.size();

    // check if the sequence has enough cache slots
    CHECK_GE(sequence->kv_cache_capacity(), seq_len);

    const uint32_t n_tokens_in_kv_cache = sequence->num_tokens_in_kv_cache();
    const uint32_t n_tokens_to_process =
        std::min(seq_len - n_tokens_in_kv_cache, max_tokens_to_process_[i]);

    // at least one token to process otherwise the sequence should be finished.
    CHECK(n_tokens_to_process != 0)
        << "at least one token should be processed. "
        << "seq_len: " << seq_len
        << ", n_tokens_in_kv_cache: " << n_tokens_in_kv_cache
        << ", max_tokens_to_process: " << max_tokens_to_process_[i];

    // pack the token ids and positions into one-dimensional tensors
    for (uint32_t i = n_tokens_in_kv_cache; i < seq_len; ++i) {
      flatten_tokens_vec.push_back(seq_token_ids[i]);
      flatten_positions_vec.push_back(static_cast<int32_t>(i));
    }
    last_token_idxes.push_back(
        static_cast<int32_t>(flatten_tokens_vec.size() - 1));

    // add token id and count for each sequence
    const auto& seq_token_counts = sequence->token_to_count_map();
    const auto unique_tokens = seq_token_counts.size();

    auto& ids = token_ids_vec.emplace_back();
    auto& counts = token_counts_vec.emplace_back();
    ids.reserve(unique_tokens);
    counts.reserve(unique_tokens);
    for (const auto& [token_id, count] : seq_token_counts) {
      ids.push_back(token_id);
      counts.push_back(count);
    }
    token_ids_lens_vec.push_back(static_cast<int32_t>(unique_tokens));
    max_unique_tokens = std::max(max_unique_tokens, unique_tokens);

    max_seq_len = std::max(max_seq_len, seq_len);
    q_max_seq_len = std::max(q_max_seq_len, n_tokens_to_process);
    cu_seq_lens.push_back(cu_seq_lens.back() + seq_len);
    q_cu_seq_lens.push_back(q_cu_seq_lens.back() + n_tokens_to_process);

    // advance kv cache pos
    sequence->advance_kv_cache_pos_by(n_tokens_to_process);

    // add sampling parameters
    model_inputs.sampling_params.add(sequence->sampling_param());

    // assign slot ids for new tokens [n_tokens_in_kvcache, total_tokens)
    const auto blocks = sequence->blocks();
    const auto slot_ids =
        sequence->kv_cache_slots(n_tokens_in_kv_cache, seq_len);
    new_token_slot_ids.insert(
        new_token_slot_ids.end(), slot_ids.begin(), slot_ids.end());

    // construct block ids for each sequence
    std::vector<int32_t> block_ids;
    block_ids.reserve(blocks.size());
    for (const auto& block : blocks) {
      block_ids.push_back(block.id());
    }
    block_tables_vec.push_back(block_ids);

    max_block_table_len =
        std::max(max_block_table_len, static_cast<int32_t>(blocks.size()));
  }

  // construct two-dimensional tensors for token ids and counts
  auto token_ids = create_2d_tensor(token_ids_vec,
                                    max_unique_tokens,
                                    torch::kInt64,
                                    /*pad_value=*/int64_t(0));
  auto token_counts = create_2d_tensor(
      token_counts_vec, max_unique_tokens, torch::kInt, /*pad_value=*/0);

  auto block_tables = create_2d_tensor(
      block_tables_vec, max_block_table_len, torch::kInt, /*pad_value=*/0);

  model_inputs.token_ids = torch::tensor(flatten_tokens_vec, torch::kInt);
  model_inputs.positions = torch::tensor(flatten_positions_vec, torch::kInt);

  auto& input_params = model_inputs.input_params;
  input_params.all_prefill_sequences = all_prefill_sequences;
  input_params.num_sequences = num_sequences;
  input_params.kv_max_seq_len = max_seq_len;
  input_params.q_max_seq_len = q_max_seq_len;
  input_params.kv_cu_seq_lens = torch::tensor(cu_seq_lens, torch::kInt);
  input_params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens, torch::kInt);
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, torch::kInt);
  input_params.block_tables = block_tables;

  auto& sampling_params = model_inputs.sampling_params;
  sampling_params.last_token_idxes =
      torch::tensor(last_token_idxes, torch::kInt);
  sampling_params.token_ids = token_ids;
  sampling_params.token_counts = token_counts;
  sampling_params.token_ids_lens =
      torch::tensor(token_ids_lens_vec, torch::kInt);

  return model_inputs;
}

ModelInput Batch::prepare_model_validate_inputs() {
  // TODO: implement this with different logic
  return prepare_model_inputs();
}

}  // namespace llm
