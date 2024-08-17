#include "linear_impl.h"

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>

#include "model_loader/state_dict.h"
#include "model_loader/tensor_utils.h"
#include "model_parallel/model_parallel.h"

namespace llm {

// Linear layer with column parallelism.
ColumnParallelLinearImpl::ColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : gather_output_(gather_output), parallel_args_(parallel_args) {
  const auto world_size = parallel_args_.world_size();
  CHECK(out_features % world_size == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size;
  const int64_t out_features_per_partition = out_features / world_size;

  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  weight_ = register_parameter(
      "weight",
      torch::empty({out_features_per_partition, in_features}, options),
      /*requires_grad=*/false);

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) const {
  namespace F = torch::nn::functional;
  auto output = F::linear(input, weight_, bias_);
  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output;
}

// load the weight from the checkpoint
void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  // call load_state_dict with identity transform
  load_state_dict(state_dict,
                  [](const torch::Tensor& tensor) { return tensor; });
}

void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict,
                                               TensorTransform transform_func) {
  CHECK(transform_func != nullptr) << "transform_func must be provided";
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();
  weight_is_loaded_ = TensorUtils::load_sharded_weights(state_dict,
                                                        "weight",
                                                        transform_func,
                                                        /*dim=*/0,
                                                        rank,
                                                        world_size,
                                                        weight_);

  if (bias_.defined()) {
    bias_is_loaded_ = TensorUtils::load_sharded_weights(state_dict,
                                                        "bias",
                                                        transform_func,
                                                        /*dim=*/0,
                                                        rank,
                                                        world_size,
                                                        bias_);
  }
}

// special load_state_dict for fused cases
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  // load and merge the weights from multiple prefixes
  TensorUtils::load_fused_weights(state_dict,
                                  prefixes,
                                  "weight",
                                  /*dim=*/0,
                                  parallel_args_.rank(),
                                  parallel_args_.world_size(),
                                  weight_list_,
                                  weight_,
                                  weight_is_loaded_);

  if (bias_.defined()) {
    TensorUtils::load_fused_weights(state_dict,
                                    prefixes,
                                    "bias",
                                    /*dim=*/0,
                                    parallel_args_.rank(),
                                    parallel_args_.world_size(),
                                    bias_list_,
                                    bias_,
                                    bias_is_loaded_);
  }
}

// Linear layer with row parallelism.
RowParallelLinearImpl::RowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : input_is_parallelized_(input_is_parallelized),
      parallel_args_(parallel_args) {
  const auto world_size = parallel_args_.world_size();
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;
  // Allocate the transpose since linear performs XA^T.
  weight_ = register_parameter(
      "weight",
      torch::empty({out_features, in_features_per_partition}, options),
      /*requires_grad=*/false);

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) const {
  namespace F = torch::nn::functional;
  if (!input_is_parallelized_) {
    input = scatter_to_model_parallel_region(input, parallel_args_);
  }
  auto output = F::linear(input, weight_);
  if (parallel_args_.world_size() > 1) {
    output = reduce_from_model_parallel_region(output, parallel_args_);
  }
  // N.B. need to apply bias after the reduce
  if (bias_.defined()) {
    output.add_(bias_);
  }
  return output;
}

// load the weight from the checkpoint
void RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();
  weight_is_loaded_ = TensorUtils::load_sharded_weights(state_dict,
                                                        "weight",
                                                        /*dim=*/1,
                                                        rank,
                                                        world_size,
                                                        weight_);

  if (bias_.defined()) {
    bias_is_loaded_ = TensorUtils::load_sharded_weights(state_dict,
                                                        "bias",
                                                        /*dim=*/1,
                                                        rank,
                                                        world_size,
                                                        bias_);
  }
}

}  // namespace llm
