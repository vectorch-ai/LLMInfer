#include <ATen/cuda/Exceptions.h>
#include <absl/random/random.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "attention_launch_sm80.cuh"
#include "attention_params.h"
#include "cute/layout.hpp"
#include "static_dispatch.h"

namespace llm {
namespace {
// Multi-head attention implementation using pytorch
torch::Tensor attention_ref(
    torch::Tensor query,  // [n_heads, q_len, head_dim]
    torch::Tensor key,    // [n_kv_heads, kv_len, head_dim]
    torch::Tensor value,  // [n_kv_heads, kv_len, head_dim]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window) {
  const auto q_len = query.size(1);
  const auto kv_len = key.size(1);
  const auto n_heads = query.size(0);
  const auto n_kv_heads = key.size(0);
  const auto head_dim = query.size(2);
  assert(kv_len >= q_len);

  if (n_heads != n_kv_heads) {
    assert(n_heads % n_kv_heads == 0);
    const auto group_size = n_heads / n_kv_heads;
    key = key.repeat_interleave(/*repeats=*/group_size, /*dim=*/-3);
    value = value.repeat_interleave(/*repeats=*/group_size, /*dim=*/-3);
  }

  const float sm_scale = 1.0 / sqrt(head_dim);
  // query * key => [n_heads, q_seq_len, seq_len]
  auto scores = torch::einsum("hqd,hkd->hqk",
                              {query.to(torch::kFloat), key.to(torch::kFloat)});
  // apply scale
  scores *= sm_scale;

  // apply softcap if needed
  if (logits_soft_cap != 0.0) {
    scores = torch::tanh(scores / logits_soft_cap) * logits_soft_cap;
  }

  // apply alibi bias
  if (alibi_slopes) {
    const auto& slopes = alibi_slopes.value();
    // calculate alibi attention bias
    // since it's causal mask, we can just use [0, 1, ...,, kv_len)
    auto distance = torch::arange(0, kv_len, query.options());
    // [n_heads, 1, kv_len]
    auto bias = distance.view({1, 1, kv_len}) * slopes.view({n_heads, 1, 1});
    scores += bias;
  }

  auto mask = torch::ones({q_len, kv_len}, torch::kBool);
  if (sliding_window >= 0) {
    // sliding window mask
    // returns the upper triangular part of a matrix
    mask = torch::triu(mask, /*diagonal=*/kv_len - q_len - sliding_window);
  }

  // apply causal mask
  // causal mask: returns the lower triangular part of a matrix
  mask = torch::tril(mask, /*diagonal=*/kv_len - q_len).to(query);
  scores = scores.masked_fill(mask == 0, -INFINITY);

  // safe softmax
  scores = torch::softmax(scores, /*dim=*/-1);

  // score * value => [batch_size, n_heads, q_seq_len, head_dim]
  return torch::einsum("hqk,hkd->hqd", {scores, value.to(torch::kFloat)})
      .type_as(query);
}

torch::Tensor attention_varlen_ref(
    torch::Tensor query,           // [n_heads, q_seq_len, head_dim]
    torch::Tensor key,             // [n_kv_heads, kv_seq_len, head_dim]
    torch::Tensor value,           // [n_kv_heads, kv_seq_len, head_dim]
    torch::Tensor q_cu_lens,       // [batch_size+1]
    torch::Tensor kv_cu_seq_lens,  // [batch_size+1]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window) {
  torch::Tensor q_cu_lens_cpu = q_cu_lens.cpu();
  torch::Tensor kv_cu_seq_lens_cpu = kv_cu_seq_lens.cpu();
  const size_t n_seqs = q_cu_lens_cpu.numel() - 1;
  const int32_t* q_cu_lens_ptr = q_cu_lens_cpu.data_ptr<int32_t>();
  const int32_t* kv_cu_lens_ptr = kv_cu_seq_lens_cpu.data_ptr<int32_t>();

  std::vector<torch::Tensor> out_list;
  // process sequence one by one
  for (int64_t i = 0; i < n_seqs; ++i) {
    // calaculate attention for each sequence
    const int32_t q_start = q_cu_lens_ptr[i];
    const int32_t q_end = q_cu_lens_ptr[i + 1];
    const int32_t kv_start = kv_cu_lens_ptr[i];
    const int32_t kv_end = kv_cu_lens_ptr[i + 1];

    torch::Tensor q = query.slice(/*dim=*/1, /*start=*/q_start, /*end=*/q_end);
    torch::Tensor k = key.slice(/*dim=*/1, /*start=*/kv_start, /*end=*/kv_end);
    torch::Tensor v =
        value.slice(/*dim=*/1, /*start=*/kv_start, /*end=*/kv_end);

    auto output =
        attention_ref(q, k, v, alibi_slopes, logits_soft_cap, sliding_window);
    out_list.push_back(output);
  }
  return torch::cat(out_list, /*dim=*/1);
}

torch::Tensor attention_pagedkv_sm80(
    torch::Tensor query,          // [n_heads, q_seq_len, head_dim]
    torch::Tensor key_cache,      // [n_kv_heads, n_slots, head_dim]
    torch::Tensor value_cache,    // [n_kv_heads, n_slots, head_dim]
    torch::Tensor q_cu_lens,      // [batch_size+1]
    torch::Tensor kv_cu_lens,     // [batch_size+1]
    torch::Tensor block_table,    // [n_blocks]
    torch::Tensor block_cu_lens,  // [batch_size+1]
    int block_size,
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window,
    int32_t max_q_len) {
  const auto n_heads = query.size(0);
  const auto n_kv_heads = key_cache.size(0);
  const auto head_dim = query.size(2);
  const auto batch_size = q_cu_lens.size(0) - 1;

  auto out = torch::empty_like(query);

  const float sm_scale = 1.0 / sqrt(head_dim);

  // construct attention params
  PagedKVAttentionParams params;
  params.q_ptr = query.const_data_ptr();
  params.q_stride = make_stride(query.stride(0), query.stride(1));
  params.k_ptr = key_cache.const_data_ptr();
  params.k_stride = make_stride(key_cache.stride(0), key_cache.stride(1));
  params.v_ptr = value_cache.const_data_ptr();
  params.v_stride = make_stride(value_cache.stride(0), value_cache.stride(1));
  params.o_ptr = out.mutable_data_ptr();
  params.o_stride = make_stride(out.stride(0), out.stride(1));
  params.alibi_slopes_ptr = alibi_slopes.has_value()
                                ? alibi_slopes.value().const_data_ptr<float>()
                                : nullptr;
  params.batch_size = batch_size;
  params.max_q_len = max_q_len;
  params.n_heads = n_heads;
  params.n_kv_heads = n_kv_heads;
  params.head_dim = head_dim;
  params.sm_scale = sm_scale;
  params.logits_soft_cap = logits_soft_cap;
  params.sliding_window = sliding_window;

  params.q_cu_lens = q_cu_lens.const_data_ptr<int32_t>();
  params.kv_cu_lens = kv_cu_lens.const_data_ptr<int32_t>();

  params.block_table = block_table.const_data_ptr<int32_t>();
  params.block_cu_lens = block_cu_lens.const_data_ptr<int32_t>();
  params.block_size = block_size;

  DISPATCH_TORCH_DTYPE(query.dtype(), QTYPE, [&] {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, [&] {
      run_attention_kernel_sm80<QTYPE, HEAD_DIM>(params);
    });
  });
  return out;
}

}  // namespace

class AttentionKernelPagedKVTest
    : public ::testing::TestWithParam<std::tuple<int64_t /*batch_size*/,
                                                 int64_t /*block_size*/,
                                                 int64_t /*q_len*/,
                                                 int64_t /*kv_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*head_dim*/,
                                                 float /*logits_soft_cap*/,
                                                 bool /*alibi*/,
                                                 int32_t /*sliding_window*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(AttentionKernelPagedKVTest, PageKV) {
  const auto [batch_size,
              block_size,
              max_q_len,
              max_kv_len,
              n_heads,
              n_kv_heads,
              head_dim,
              logits_soft_cap,
              alibi,
              sliding_window] = GetParam();

  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);

  std::vector<int32_t> block_table_vec;
  std::vector<int32_t> block_cu_lens_vec = {0};
  std::vector<int> slot_ids;

  const int32_t total_blocks = (max_kv_len * batch_size) / block_size + 2;
  // random generate seq lens with size in [1, max_seq_len]
  std::vector<int32_t> q_cu_lens_vec = {0};
  std::vector<int32_t> kv_cu_lens_vec = {0};
  int32_t n_kv_tokens = 0;
  int32_t n_q_tokens = 0;
  absl::BitGen gen;
  for (int i = 0; i < batch_size; ++i) {
    // q_len: [1, q_max_seq_len]
    const int32_t q_len =
        absl::Uniform<int>(absl::IntervalClosedClosed, gen, 1, max_q_len);
    n_q_tokens += q_len;
    q_cu_lens_vec.push_back(n_q_tokens);

    // kv_len >= q_len
    int32_t kv_len = q_len;
    if (q_len < max_kv_len) {
      // sample kv_len from [q_len, kv_max_seq_len]
      kv_len = absl::Uniform<int>(
          absl::IntervalClosedClosed, gen, q_len, max_kv_len);
    }
    n_kv_tokens += kv_len;
    kv_cu_lens_vec.push_back(n_kv_tokens);
    assert(kv_len >= q_len);

    // assign blocks for each sequence
    const int32_t n_blocks = (kv_len + block_size - 1) / block_size;
    std::vector<int32_t> block_ids;
    block_ids.reserve(n_blocks);
    for (int j = 0; j < n_blocks; ++j) {
      // random assign block size
      block_ids.push_back(absl::Uniform<int>(
          absl::IntervalClosedClosed, gen, 1, total_blocks - 1));
    }
    block_table_vec.insert(
        block_table_vec.end(), block_ids.begin(), block_ids.end());
    block_cu_lens_vec.push_back(block_table_vec.size());
    for (int j = 0; j < kv_len; ++j) {
      const int32_t block_id = block_ids[j / block_size];
      const int32_t block_offset = j % block_size;
      slot_ids.push_back(block_id * block_size + block_offset);
    }
  }

  // construct non-contiguous query, key and value
  // generate query, key and value
  torch::Tensor query = torch::rand({n_heads, n_q_tokens, head_dim}, options);
  const auto n_slots = total_blocks * block_size;
  torch::Tensor key_cache =
      torch::rand({n_kv_heads, n_slots, head_dim}, options);
  torch::Tensor value_cache =
      torch::rand({n_kv_heads, n_slots, head_dim}, options);

  torch::Tensor q_cu_lens = torch::tensor(
      q_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));
  torch::Tensor kv_cu_lens = torch::tensor(
      kv_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));

  torch::Tensor block_table = torch::tensor(
      block_table_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));
  torch::Tensor block_cu_lens = torch::tensor(
      block_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));

  torch::optional<torch::Tensor> alibi_slopes;
  if (alibi) {
    alibi_slopes = torch::rand(
        {n_heads}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // get combined key and value
  std::vector<torch::Tensor> keys;
  keys.reserve(slot_ids.size());
  std::vector<torch::Tensor> values;
  values.reserve(slot_ids.size());
  for (int slot_id : slot_ids) {
    using ISlice = torch::indexing::Slice;
    // kv = kv_cache[:, slot_idx, :]
    const auto key = key_cache.index({ISlice(), slot_id, ISlice()});
    const auto value = value_cache.index({ISlice(), slot_id, ISlice()});
    keys.push_back(key.reshape({n_kv_heads, head_dim}));
    values.push_back(value.reshape({n_kv_heads, head_dim}));
  }
  const auto key = torch::stack(keys).transpose(0, 1);
  const auto value = torch::stack(values).transpose(0, 1);

  auto ref_out = attention_varlen_ref(query,
                                      key,
                                      value,
                                      q_cu_lens,
                                      kv_cu_lens,
                                      alibi_slopes,
                                      logits_soft_cap,
                                      sliding_window);

  auto out = attention_pagedkv_sm80(query,
                                    key_cache,
                                    value_cache,
                                    q_cu_lens,
                                    kv_cu_lens,
                                    block_table,
                                    block_cu_lens,
                                    block_size,
                                    alibi_slopes,
                                    logits_soft_cap,
                                    sliding_window,
                                    max_q_len);

  EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
}

INSTANTIATE_TEST_SUITE_P(
    PagedKV,
    AttentionKernelPagedKVTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 4),                          // batch_size
        ::testing::Values(1, 8),                             // block_size
        ::testing::Values(1, 125),                           // max_q_len
        ::testing::Values(127, 1000),                        // max_kv_len
        ::testing::Values(6),                                // n_heads
        ::testing::Values(6 /*mha*/, 3 /*gqa*/, 1 /*mqa*/),  // n_kv_heads
        ::testing::Values(64),                               // head_dim
        ::testing::Values(0.0, 50.0),                        // logits_soft_cap
        ::testing::Values(false, true),                      // alibi slope
        ::testing::Values(-1, 0, 10)                         // sliding window
        ));

}  // namespace llm