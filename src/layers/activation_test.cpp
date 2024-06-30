#include "activation.h"

#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <map>
#include <tuple>

#include "kernels/activation_kernels.h"

namespace llm {

class ActivationTest
    : public ::testing::TestWithParam<std::tuple<torch::Device,
                                                 torch::ScalarType,
                                                 std::string /*activation*/,
                                                 int64_t /*in_features*/,
                                                 int64_t /*out_features*/>> {};

const std::map<std::string, ActFunc> activations = {
    {"gelu", detail::gelu},
    {"gelu_fast", detail::gelu_fast},
    {"gelu_new", detail::gelu_new},
    {"gelu_pytorch_tanh", detail::gelu_pytorch_tanh},
    {"relu", detail::relu},
    {"silu", detail::silu},
};
const std::map<std::string, ActFunc> activation_kernels = {
    {"gelu_fast", kernel::gelu_fast},
    {"gelu_new", kernel::gelu_new},
    {"silu", kernel::silu},
};

TEST_P(ActivationTest, Basic) {
  const auto& [device, dtype, activation, in_features, out_features] =
      GetParam();

  if (device.is_cuda() && !torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  auto input = torch::rand({in_features, out_features},
                           torch::dtype(dtype).device(device));

  // use float result as baseline
  auto input_float = input.to(torch::kFloat);
  auto desired_output = activations.at(activation)(input).to(dtype);
  // same dtype and device
  EXPECT_EQ(input.dtype(), desired_output.dtype());
  EXPECT_EQ(input.device(), desired_output.device());

  auto output = Activation::get_act_func(activation, device)(input);
  // same dtype and device
  EXPECT_EQ(input.dtype(), output.dtype());
  EXPECT_EQ(input.device(), output.device());

  EXPECT_TRUE(torch::allclose(desired_output,
                              output,
                              /*rtol=*/1e-01,
                              /*atol=*/1e-02));
}

INSTANTIATE_TEST_SUITE_P(
    ActivationCUDATest,
    ActivationTest,
    ::testing::Combine(
        ::testing::Values(torch::kCUDA),
        ::testing::Values(torch::kFloat, torch::kHalf, torch::kBFloat16),
        ::testing::Values("gelu",
                          "gelu_fast",
                          "gelu_new",
                          "gelu_pytorch_tanh",
                          "relu",
                          "silu"),
        ::testing::Values(200),          // in_features
        ::testing::Values(256, 1088)));  // out_features

INSTANTIATE_TEST_SUITE_P(
    ActivationCPUTest,
    ActivationTest,
    ::testing::Combine(::testing::Values(torch::kCPU),
                       ::testing::Values(torch::kFloat),
                       ::testing::Values("gelu",
                                         "gelu_fast",
                                         "gelu_new",
                                         "gelu_pytorch_tanh",
                                         "relu",
                                         "silu"),
                       ::testing::Values(200),          // in_features
                       ::testing::Values(256, 1088)));  // out_features

class ActivationKernelTest : public ActivationTest {};

TEST_P(ActivationKernelTest, KernelTest) {
  // skip the test if no GPU
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  const auto& [device, dtype, activation, in_features, out_features] =
      GetParam();

  // generate input with non-contiguous memory
  auto input = torch::rand({in_features, out_features * 2},
                           torch::dtype(dtype).device(device))
                   .chunk(/*chunks=*/2, /*dim=*/1)[1];
  EXPECT_TRUE(!input.is_contiguous());

  // use float result as baseline
  auto input_float = input.to(torch::kFloat);
  auto output = activations.at(activation)(input_float).to(dtype);
  // same dtype and device
  EXPECT_EQ(input.dtype(), output.dtype());
  EXPECT_EQ(input.device(), output.device());

  auto kernel_output = activation_kernels.at(activation)(input);
  // same dtype and device
  EXPECT_EQ(input.dtype(), kernel_output.dtype());
  EXPECT_EQ(input.device(), kernel_output.device());

  EXPECT_TRUE(torch::allclose(output,
                              kernel_output,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));
}

const std::map<std::string, ActFunc> fused_activations = {
    {"gelu", detail::gelu_with_mul},
    {"gelu_fast", detail::gelu_fast_with_mul},
    {"gelu_new", detail::gelu_new_with_mul},
    {"gelu_pytorch_tanh", detail::gelu_pytorch_tanh_with_mul},
    {"relu", detail::relu_with_mul},
    {"silu", detail::silu_with_mul},
};
const std::map<std::string, ActFunc> fused_activation_kernels = {
    {"gelu_fast", kernel::gelu_fast_with_mul},
    {"gelu_new", kernel::gelu_new_with_mul},
    {"silu", kernel::silu_with_mul},
};

TEST_P(ActivationKernelTest, FusedKernelTest) {
  // skip the test if no GPU
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  const auto& [device, dtype, activation, in_features, out_features] =
      GetParam();

  // generate input with non-contiguous memory
  auto input = torch::rand({in_features, out_features * 2},
                           torch::dtype(dtype).device(device));

  // use float result as baseline
  auto input_float = input.to(torch::kFloat);
  auto output = fused_activations.at(activation)(input_float).to(dtype);
  // same dtype and device
  EXPECT_EQ(input.dtype(), output.dtype());
  EXPECT_EQ(input.device(), output.device());

  auto kernel_output = fused_activation_kernels.at(activation)(input);
  // same dtype and device
  EXPECT_EQ(input.dtype(), kernel_output.dtype());
  EXPECT_EQ(input.device(), kernel_output.device());

  EXPECT_TRUE(torch::allclose(output,
                              kernel_output,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));
}

INSTANTIATE_TEST_SUITE_P(
    ActivationKernelTest,
    ActivationKernelTest,
    ::testing::Combine(
        ::testing::Values(torch::kCUDA),
        ::testing::Values(torch::kFloat, torch::kHalf, torch::kBFloat16),
        ::testing::Values("gelu_fast", "gelu_new", "silu"),
        ::testing::Values(200),          // in_features
        ::testing::Values(256, 1088)));  // out_features

}  // namespace llm
