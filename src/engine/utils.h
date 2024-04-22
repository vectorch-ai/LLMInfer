#pragma once

#include <torch/torch.h>

#include <vector>

#include "models/parameters.h"

namespace llm {

class Utils {
 public:
  static void prepare_profile_inputs(int64_t max_num_tokens,
                                     int64_t max_num_seqs,
                                     torch::Tensor* flatten_token_ids,
                                     torch::Tensor* flatten_positions,
                                     InputParameters* input_params);
};

}  // namespace llm
