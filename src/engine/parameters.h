#pragma once

#include <torch/torch.h>

#include "models/parameters.h"
#include "sampling/parameters.h"

namespace llm {

// input for the model that encapsulates all the necessary
// input information.
struct ModelInput {
  // flatten token ids
  torch::Tensor token_ids;
  // flatten positions
  torch::Tensor positions;
  // input parameters, mainly for attention
  InputParameters input_params;
  // sampling parameters, mainly for sampling
  SamplingParameters sampling_params;
};

// output for the model that encapsulates all the necessary
// output information.
struct ModelOutput {
  // sample parameters carried over from input, used for speculative decoding
  torch::Tensor do_sample;

  // whether to return logprobs
  bool logprobs = false;

  // max number of top logprobs in the batch
  int64_t max_top_logprobs = 0;

  // output of sampling
  SampleOutput sample_output;

  // logits for selected indices
  torch::Tensor logits;

  // hidden states for selected indices
  torch::Tensor hidden_states;
};

}  // namespace llm