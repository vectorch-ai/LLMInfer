#include "logits_processor.h"

#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <gtest/gtest.h>
#include <kernels/sampling/sampling_kernels.h>
#include <torch/torch.h>
#include <torch/types.h>

namespace llm {
torch::Tensor unique_randint(int64_t low,
                             int64_t high,
                             at::IntArrayRef size,
                             at::TensorOptions options) {
  auto tensor = torch::empty(size, options);
  for (int64_t i = 0; i < size[0]; ++i) {
    auto range_tensor = torch::arange(low, high, options);
    auto unique_tensor = range_tensor.index_select(
        /*dim=*/0,
        torch::randperm(range_tensor.size(0),
                        torch::dtype(torch::kInt64).device(options.device()))
            .slice(/*dim=*/0, 0, size[1]));
    tensor[i] = unique_tensor;
  }
  return tensor;
}

TEST(LogitsProcessorTest, Temperature) {
  // Test TemperatureLogitsProcessor
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  const std::vector<float> temperatures = {0.5, 1.5};
  TemperatureLogitsProcessor processor(temperatures, dtype, device);

  int64_t batch_size = 2;
  int64_t vocab_size = 32000;
  const auto logits = torch::randn({batch_size, vocab_size},
                                   torch::dtype(dtype).device(device));
  // calculate desired logits one by one
  auto desired_logits = logits.clone();
  for (int i = 0; i < batch_size; ++i) {
    desired_logits[i] /= temperatures[i];
  }

  torch::Tensor token_ids;
  torch::Tensor token_counts;
  torch::Tensor tokens_ids_lens;
  auto output =
      processor(token_ids, token_counts, tokens_ids_lens, logits.clone());
  EXPECT_TRUE(torch::allclose(output, desired_logits));
}

TEST(LogitsProcessorTest, TemperatureKernel) {
  // Test TemperatureLogitsProcessor
  torch::ScalarType dtype(torch::kHalf);
  torch::Device device(torch::kCUDA);
  const auto temperatures = torch::tensor(std::vector<float>{0.5, 1.5},
                                          torch::dtype(dtype).device(device))
                                .unsqueeze(1);

  int64_t batch_size = 2;
  int64_t vocab_size = 32000;
  const auto logits = torch::randn({batch_size, vocab_size},
                                   torch::dtype(dtype).device(device));

  torch::Tensor token_ids;
  torch::Tensor token_counts;
  torch::Tensor tokens_ids_lens;
  auto output = logits.clone();
  detail::apply_temperature_penalty(output, temperatures);
  auto kernel_output = logits.clone();
  kernel::apply_temperature_penalty(kernel_output, temperatures);
  EXPECT_TRUE(torch::allclose(output,
                              kernel_output,
                              /*rtol=*/1e-03,
                              /*atol=*/1e-05));
}

TEST(LogitsProcessorTest, FrequencyPresencePenalty) {
  // Test FrequencyPresencePenaltyLogitsProcessor
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  std::vector<float> frequency_penalties = {0.01, 0.02};
  std::vector<float> presence_penalties = {0.1, 0.2};
  FrequencyPresencePenaltyLogitsProcessor processor(
      frequency_penalties, presence_penalties, dtype, device);

  int64_t batch_size = 2;
  int64_t max_seq_len = 1023;
  int64_t vocab_size = 32000;
  const auto logits = torch::randn({batch_size, vocab_size},
                                   torch::dtype(dtype).device(device));
  const torch::Tensor token_ids = unique_randint(
      /*low=*/1,
      /*high=*/vocab_size,
      /*size=*/{batch_size, max_seq_len},
      torch::dtype(torch::kInt64).device(device));
  const torch::Tensor token_counts = torch::randint(
      /*low=*/1,
      /*high=*/3,
      /*size=*/{batch_size, max_seq_len},
      torch::dtype(torch::kInt32).device(device));

  // calculate desired logits one by one
  auto desired_logits = logits.clone();
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < max_seq_len; ++j) {
      auto token_id = token_ids[i][j].item<int64_t>();
      auto token_count = token_counts[i][j].item<int32_t>();
      desired_logits[i][token_id] -= (frequency_penalties[i] * token_count);
      if (token_count > 0) {
        desired_logits[i][token_id] -= presence_penalties[i];
      }
    }
  }

  torch::Tensor tokens_ids_lens;
  auto output =
      processor(token_ids, token_counts, tokens_ids_lens, logits.clone());
  EXPECT_TRUE(torch::allclose(output, desired_logits));
}

TEST(LogitsProcessorTest, FrequencyPresencePenaltyKernel) {
  // Test FrequencyPresencePenaltyLogitsProcessor
  torch::ScalarType dtype(torch::kHalf);
  torch::Device device(torch::kCUDA);
  const auto frequency_penalties =
      torch::tensor(std::vector<float>{0.01, 0.02},
                    torch::dtype(dtype).device(device))
          .unsqueeze(1);
  const auto presence_penalties =
      torch::tensor(std::vector<float>{0.1, 0.2},
                    torch::dtype(dtype).device(device))
          .unsqueeze(1);

  int32_t batch_size = 2;
  int32_t max_seq_len = 1023;
  int32_t vocab_size = 32000;
  const auto logits = torch::randn({batch_size, vocab_size},
                                   torch::dtype(dtype).device(device));
  const torch::Tensor token_ids = unique_randint(
      /*low=*/1,
      /*high=*/vocab_size,
      /*size=*/{batch_size, max_seq_len},
      torch::dtype(torch::kInt64).device(device));
  const torch::Tensor token_counts = torch::randint(
      /*low=*/1,
      /*high=*/3,
      /*size=*/{batch_size, max_seq_len},
      torch::dtype(torch::kInt32).device(device));

  torch::Tensor tokens_ids_lens =
      torch::tensor(std::vector<int32_t>{max_seq_len, max_seq_len},
                    torch::dtype(torch::kInt).device(device));

  auto output = logits.clone();
  detail::apply_frequency_presence_penalty(output,
                                           token_ids,
                                           token_counts,
                                           tokens_ids_lens,
                                           frequency_penalties,
                                           presence_penalties);
  auto kernel_output = logits.clone();
  kernel::apply_frequency_presence_penalty(kernel_output,
                                           token_ids,
                                           token_counts,
                                           tokens_ids_lens,
                                           frequency_penalties,
                                           presence_penalties);
  EXPECT_TRUE(torch::allclose(output,
                              kernel_output,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));
}

TEST(LogitsProcessorTest, RepetitionPenalty) {
  // Test RepetitionPenaltyLogitsProcessor
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  const std::vector<float> repetition_penalties = {1.0, 2.0};
  RepetitionPenaltyLogitsProcessor processor(
      repetition_penalties, dtype, device);

  int64_t batch_size = 2;
  int64_t max_seq_len = 1023;
  int64_t vocab_size = 32000;
  const auto logits = torch::randn({batch_size, vocab_size},
                                   torch::dtype(dtype).device(device));
  const torch::Tensor token_ids = unique_randint(
      /*low=*/1,
      /*high=*/vocab_size,
      /*size=*/{batch_size, max_seq_len},
      torch::dtype(torch::kInt64).device(device));

  // calculate the desired logits
  auto desired_logits = logits.clone();
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < max_seq_len; ++j) {
      auto token_id = token_ids[i][j].item<int64_t>();
      auto score = desired_logits[i][token_id].item<float>();
      if (score < 0) {
        desired_logits[i][token_id] = score * repetition_penalties[i];
      } else {
        desired_logits[i][token_id] = score / repetition_penalties[i];
      }
    }
  }

  torch::Tensor token_counts;
  torch::Tensor tokens_ids_lens;
  auto output =
      processor(token_ids, token_counts, tokens_ids_lens, logits.clone());
  EXPECT_TRUE(torch::allclose(output, desired_logits));
}

TEST(LogitsProcessorTest, RepetitionPenaltyKernel) {
  // Test RepetitionPenaltyLogitsProcessor
  torch::ScalarType dtype(torch::kHalf);
  torch::Device device(torch::kCUDA);
  const auto repetition_penalties =
      torch::tensor(std::vector<float>{1.0, 2.0},
                    torch::dtype(dtype).device(device))
          .unsqueeze(1);

  int32_t batch_size = 2;
  int32_t max_seq_len = 1023;
  int32_t vocab_size = 32000;
  const auto logits = torch::randn({batch_size, vocab_size},
                                   torch::dtype(dtype).device(device));
  const torch::Tensor token_ids = unique_randint(
      /*low=*/1,
      /*high=*/vocab_size,
      /*size=*/{batch_size, max_seq_len},
      torch::dtype(torch::kInt64).device(device));

  torch::Tensor token_counts;
  torch::Tensor tokens_ids_lens =
      torch::tensor(std::vector<int32_t>{max_seq_len, max_seq_len},
                    torch::dtype(torch::kInt).device(device));

  auto output = logits.clone();
  detail::apply_repetition_penalty(
      output, token_ids, tokens_ids_lens, repetition_penalties);
  auto kernel_output = logits.clone();
  kernel::apply_repetition_penalty(
      kernel_output, token_ids, tokens_ids_lens, repetition_penalties);
  EXPECT_TRUE(torch::allclose(output,
                              kernel_output,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));
}

TEST(LogitsProcessorTest, DISABLED_TopK) {
  // Test TopKLogitsProcessor
  // Set the random seed
  torch::manual_seed(100);
  torch::ScalarType dtype(torch::kHalf);
  torch::Device device(torch::kCUDA);
  int64_t batch_size = 5;
  int64_t vocab_size = 100;
  const std::vector<int64_t> top_k = {50, 60, 70, 80, 100};
  const float filter_value = -std::numeric_limits<float>::infinity();
  TopKLogitsProcessor processor(top_k, dtype, device, filter_value);

  auto logits = torch::randn({batch_size, vocab_size},
                             torch::dtype(dtype).device(device));
  torch::Tensor token_ids;
  torch::Tensor token_counts;
  torch::Tensor tokens_ids_lens;
  auto logits_output =
      processor(token_ids, token_counts, tokens_ids_lens, logits.clone());

  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t k = top_k[i];

    // calculate top k values one by one
    auto [top_k_values, top_k_indices] =
        logits[i].topk(k, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
    auto [output_sorted, output_indices] =
        logits_output[i].sort(/*dim=*/-1, /*descending=*/true);
    // top k values should be the same
    ASSERT_TRUE(
        torch::equal(output_sorted.slice(/*dim=*/0, 0, k), top_k_values));
    // all remaining values should be filter_value
    auto masked_output = output_sorted.slice(/*dim=*/0, k);
    ASSERT_TRUE(torch::equal(masked_output,
                             torch::full_like(masked_output, filter_value)));
  }
}

TEST(LogitsProcessorTest, DISABLED_TopP) {
  // Test TopPLogitsProcessor
  // Set the random seed
  torch::manual_seed(100);
  torch::ScalarType dtype(torch::kHalf);
  torch::Device device(torch::kCUDA);
  int64_t batch_size = 4;
  int64_t vocab_size = 100;
  int64_t min_tokens_to_keep = 1;
  std::vector<float> top_p = {0.5, 0.7, 0.9, 1.0};
  const float filter_value = -std::numeric_limits<float>::infinity();
  TopPLogitsProcessor processor(
      top_p, dtype, device, filter_value, min_tokens_to_keep);

  auto logits = torch::randn({batch_size, vocab_size},
                             torch::dtype(dtype).device(device));
  torch::Tensor token_ids;
  torch::Tensor token_counts;
  torch::Tensor tokens_ids_lens;
  auto logits_output =
      processor(token_ids, token_counts, tokens_ids_lens, logits.clone());

  // verify result one by one
  for (int64_t i = 0; i < batch_size; ++i) {
    const float p = top_p[i];

    // calculate number of values to keep (k)
    auto probs = torch::softmax(logits[i], /*dim=*/-1);
    // calculate top p values one by one
    auto [sorted_probs, sorted_indices] =
        probs.sort(/*dim=*/-1, /*descending=*/true);
    auto probs_sum = sorted_probs.cumsum(/*dim=*/-1);
    torch::Tensor mask = (probs_sum - sorted_probs) <= p;
    const int64_t k = std::max(mask.sum().item<int64_t>(), min_tokens_to_keep);

    // gather top logits value
    auto [sorted_logits, sorted_logits_indices] =
        logits[i].sort(/*dim=*/-1, /*descending=*/true);

    auto [output_sorted, output_indices] =
        logits_output[i].sort(/*dim=*/-1, /*descending=*/true);
    ASSERT_TRUE(torch::equal(output_sorted.slice(/*dim=*/0, 0, k),
                             sorted_logits.slice(/*dim=*/0, 0, k)));
    // all remaining values should be filter_value
    auto masked_output = output_sorted.slice(/*dim=*/0, k);
    ASSERT_TRUE(torch::equal(masked_output,
                             torch::full_like(masked_output, filter_value)));
  }
}

}  // namespace llm
