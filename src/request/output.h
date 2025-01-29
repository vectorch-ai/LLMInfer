#pragma once

#include <glog/logging.h>

#include <optional>
#include <string>
#include <vector>

#include "status.h"

namespace llm {

// Priority of the request.
// The higher the priority, the sooner the request is processed.
enum class Priority { HIGH = 0, NORMAL, LOW };

// "stop" - the model hit a natural stop point or a provided stop sequence.
// "length" - the maximum number of tokens specified in the request was reached.
// "function_call" - the model called a function.
enum class FinishReason {
  NONE = 0,
  STOP = 1,
  LENGTH,
  FUNCTION_CALL,
};

struct Usage {
  // the number of tokens in the prompt.
  size_t num_prompt_tokens = 0;

  // the number of tokens in the generated completion.
  size_t num_generated_tokens = 0;

  // the total number of tokens used in the request (prompt + completion).
  size_t num_total_tokens = 0;
};

struct LogProbData {
  // the text of the token.
  std::string token;
  // the token id.
  int32_t token_id;
  // the log probability of the token.
  float logprob = -9999.0f;
  // whether the token is finished.
  bool finished_token = true;
};

struct LogProb : public LogProbData {
  // the top log probabilities.
  std::optional<std::vector<LogProbData>> top_logprobs;
};

struct SequenceOutput {
  // the index of the sequence in the request.
  size_t index;

  // the generated/delta text.
  // delta text is the text generated since the last response for streaming.
  std::string text;

  // the token ids of the generated text.
  std::vector<int32_t> token_ids;

  // the reason the sequence finished.
  std::optional<std::string> finish_reason;

  // log probabilities of the generated tokens.
  std::optional<std::vector<LogProb>> logprobs;

  // the embeddings of the sequence, only used for embedding generation.
  std::optional<std::vector<float>> embeddings;
};

struct RequestOutput {
  RequestOutput() = default;

  RequestOutput(Status&& _status) : status(std::move(_status)) {}

  // the prompt text for the request.
  std::optional<std::string> prompt;

  // the status of the request.
  std::optional<Status> status;

  // the output for each sequence in the request.
  std::vector<SequenceOutput> outputs;

  // the statistics for the request.
  std::optional<Usage> usage;

  // whether the request is finished.
  bool finished = false;
};

inline std::optional<std::string> to_string(FinishReason reason) {
  switch (reason) {
    case FinishReason::NONE:
      return std::nullopt;
    case FinishReason::STOP:
      return "stop";
    case FinishReason::LENGTH:
      return "length";
    case FinishReason::FUNCTION_CALL:
      return "function_call";
    default:
      LOG(WARNING) << "Unknown finish reason: " << static_cast<int>(reason);
  }
  return std::nullopt;
}

}  // namespace llm