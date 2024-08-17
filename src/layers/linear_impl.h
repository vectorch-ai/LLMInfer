#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "linear.h"
#include "model_loader/state_dict.h"

namespace llm {
namespace detail {

void load_weights(const StateDict& state_dict,
                  const std::string& name,
                  torch::Tensor& weight,
                  bool& weight_is_loaded);

void load_fused_weights(const StateDict& state_dict,
                        const std::vector<std::string>& prefixes,
                        const std::string& name,
                        int64_t dim,
                        int32_t rank,
                        int32_t world_size,
                        std::vector<torch::Tensor>& accumulated_tensors,
                        torch::Tensor& weight,
                        bool& weight_is_loaded);

void load_weights(const StateDict& state_dict,
                  const std::string& name,
                  int64_t dim,
                  int32_t rank,
                  int32_t world_size,
                  torch::Tensor& weight,
                  bool& weight_is_loaded);

void load_weights_with_transform(const StateDict& state_dict,
                                 const std::string& name,
                                 TensorTransform transform_func,
                                 int64_t dim,
                                 int32_t rank,
                                 int32_t world_size,
                                 torch::Tensor& weight,
                                 bool& weight_is_loaded);               

// helper function to merge fused weights
void merge_weights(const std::string& tensor_name,
                   std::vector<torch::Tensor> weight_list,
                   int64_t dim,  // dim to cat
                   bool clone,   // wheather to make a colne for accumulating
                   std::vector<torch::Tensor>& accumulated_weight_list,
                   torch::Tensor& weight,
                   bool& weight_is_loaded);

}  // namespace detail

// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelLinearImpl : public ParallelLinearImpl {
 public:
  ColumnParallelLinearImpl(int64_t in_features,
                           int64_t out_features,
                           bool bias,
                           bool gather_output,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input) const override;

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // load state dict with a transform function
  void load_state_dict(const StateDict& state_dict,
                       TensorTransform transform_func) override;

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes) override;

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix) const override {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!bias_.defined() || bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  torch::Tensor weight_{nullptr};
  torch::Tensor bias_{nullptr};

  bool weight_is_loaded_ = false;
  bool bias_is_loaded_ = false;
  std::vector<torch::Tensor> weight_list_;
  std::vector<torch::Tensor> bias_list_;

  // whether to gather the output
  bool gather_output_;

  // parallel args
  ParallelArgs parallel_args_;
};

// Linear layer with row parallelism.
//     The linear layer is defined as Y = XA + b. A is parallelized along
//     its first dimension and X along its second dimension as:
//                -   -
//               | A_1 |
//               | .   |
//           A = | .   |       X = [X_1, ..., X_p]
//               | .   |
//               | A_p |
//                -   -
class RowParallelLinearImpl : public ParallelLinearImpl {
 public:
  RowParallelLinearImpl(int64_t in_features,
                        int64_t out_features,
                        bool bias,
                        bool input_is_parallelized,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input) const override;

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const override {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!bias_.defined() || bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features_per_partition]
  torch::Tensor weight_{nullptr};
  torch::Tensor bias_{nullptr};

  // whether the weight is loaded
  bool weight_is_loaded_ = false;
  bool bias_is_loaded_ = false;

  // whether the input is already parallelized
  bool input_is_parallelized_;

  // parallel args
  ParallelArgs parallel_args_;
};
}  // namespace llm
