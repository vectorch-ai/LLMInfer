#include "chat_handler.h"

#include <absl/strings/escaping.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <torch/torch.h>
#include <uuid.h>

#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <string>

#include "chat_template/chat_template.h"
#include "chat_template/jinja_chat_template.h"
#include "engine/engine.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "request/request.h"
#include "scheduler/scheduler.h"
#include "utils.h"

DEFINE_bool(enable_jinja_chat_template, false, "Enable Jinja chat template");

DECLARE_int32(num_speculative_tokens);

namespace llm {

namespace {

std::string generate_request_id() {
  return "chatcmpl-" + uuids::to_string(uuids::uuid_system_generator{}());
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool verify_request_arguments(ChatCallData* call_data) {
  const auto& request = call_data->request();
  if (request.messages().empty()) {
    call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                 "messages is empty");
    return false;
  }

  // up to 4 stop sequences
  if (request.stop_size() > 4) {
    call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                 "stop size is too large");
    return false;
  }
  // temperature between [0.0, 2.0]
  if (request.has_temperature()) {
    if (request.temperature() < 0.0 || request.temperature() > 2.0) {
      call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                   "temperature must be between 0.0 and 2.0");
      return false;
    }
  }
  // top_p between [0.0, 1.0]
  if (request.has_top_p()) {
    if (request.top_p() < 0.0 || request.top_p() > 1.0) {
      call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                   "top_p must be between 0.0 and 1.0");
      return false;
    }
  }

  // presence_penalty between [-2.0, 2.0]
  if (request.has_presence_penalty()) {
    if (request.presence_penalty() < -2.0 || request.presence_penalty() > 2.0) {
      call_data->finish_with_error(
          grpc::StatusCode::INVALID_ARGUMENT,
          "presence_penalty must be between -2.0 and 2.0");
      return false;
    }
  }
  // frequency_penalty between [0.0, 2.0]
  if (request.has_frequency_penalty()) {
    if (request.frequency_penalty() < 0.0 ||
        request.frequency_penalty() > 2.0) {
      call_data->finish_with_error(
          grpc::StatusCode::INVALID_ARGUMENT,
          "frequency_penalty must be between 0.0 and 2.0");
      return false;
    }
  }
  return true;
}

bool send_delta_to_client(ChatCallData* call_data,
                          Request* request,
                          std::vector<bool>* first_message,
                          const RequestOutput& output) {
  // send delta to client
  for (const auto& seq_output : output.outputs) {
    proto::ChatResponse response;
    response.set_object("chat.completion.chunk");
    response.set_id(request->id);
    response.set_created(request->created_time);
    // response.set_model(request->model);
    auto* choice = response.add_choices();
    const auto& index = seq_output.index;
    choice->set_index(index);
    // add message
    auto* message = choice->mutable_delta();
    // only set role for first message
    if ((*first_message)[index]) {
      message->set_role("assistant");
      (*first_message)[index] = false;
    }
    message->set_content(seq_output.text);
    if (seq_output.finish_reason.has_value()) {
      choice->set_finish_reason(seq_output.finish_reason.value());
    }
    if (!call_data->write(std::move(response))) {
      return false;
    }
  }
  return true;
}

bool send_result_to_client(ChatCallData* call_data,
                           Request* request,
                           const RequestOutput& req_output) {
  if (req_output.outputs.empty()) {
    // TODO: mapping status to grpc status
    return call_data->finish();
  }

  proto::ChatResponse response;
  response.set_object("chat.completion");
  response.set_id(request->id);
  response.set_created(request->created_time);
  // response.set_model(request->model);

  for (const auto& output : req_output.outputs) {
    // add choices into response
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    auto* message = choice->mutable_message();
    message->set_role("assistant");
    message->set_content(output.text);
    if (output.finish_reason.has_value()) {
      choice->set_finish_reason(output.finish_reason.value());
    }
  }

  // add usage statistics
  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
  }

  // TODO: combine write and finish
  call_data->write(response);
  return call_data->finish();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::unique_ptr<Request> grpc_request_to_request(ChatCallData* call_data,
                                                 ChatTemplate* chat_template,
                                                 const Tokenizer& tokenizer,
                                                 const ModelArgs& model_args) {
  const auto& grpc_request = call_data->request();
  const int64_t max_context_len = model_args.max_position_embeddings();

  // construct prompt from dialog messages
  if (chat_template == nullptr) {
    call_data->finish_with_error(
        grpc::StatusCode::INVALID_ARGUMENT,
        "Chat template has not configured, please use /completion API");
    LOG(ERROR) << "Failed to get dialog factory for model type: "
               << model_args.model_type();
    return nullptr;
  }

  std::vector<Message> messages;
  for (const auto& message : grpc_request.messages()) {
    messages.push_back({message.role(), message.content()});
  }
  auto prompt = chat_template->apply(messages);
  if (!prompt.has_value()) {
    call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                 "Failed to construct prompt from messages");
    LOG(ERROR) << "Failed to construct prompt from messages";
    return nullptr;
  }

  std::vector<int> prompt_tokens;
  if (!tokenizer.encode(prompt.value(), &prompt_tokens)) {
    call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                 "Failed to encode prompt");
    LOG(ERROR) << "Failed to encode prompt: " << prompt.value();
    return nullptr;
  }
  if (prompt_tokens.size() >= max_context_len) {
    call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                 "Prompt is too long");
    LOG(ERROR) << "Prompt is too long, prompt_len:" << prompt_tokens.size()
               << ", max_context_len: " << max_context_len;
    return nullptr;
  }

  uint32_t max_tokens = 0;
  if (grpc_request.has_max_tokens()) {
    max_tokens = grpc_request.max_tokens();
  } else {
    const uint32_t kDefaultMaxTokens = 16;
    max_tokens = kDefaultMaxTokens;
  }

  const uint32_t num_seqs = grpc_request.has_n() ? grpc_request.n() : 1;
  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens
  const size_t capacity = prompt_tokens.size() + max_tokens +
                          FLAGS_num_speculative_tokens + /*bouns_token*/ 1;
  auto request = std::make_unique<Request>(
      generate_request_id(), "", prompt_tokens, capacity, num_seqs);

  // construct sampling parameters
  auto& sampling_param = request->sampling_param;
  if (grpc_request.has_frequency_penalty()) {
    sampling_param.frequency_penalty = grpc_request.frequency_penalty();
  }
  if (grpc_request.has_presence_penalty()) {
    sampling_param.presence_penalty = grpc_request.presence_penalty();
  }
  if (grpc_request.has_temperature()) {
    sampling_param.temperature = grpc_request.temperature();
  }
  if (grpc_request.has_top_p()) {
    sampling_param.top_p = grpc_request.top_p();
  }
  // TODO: add support for following extended parameters
  // sampling_param.repetition_penalty = grpc_request.repetition_penalty();
  // sampling_param.top_k = grpc_request.top_k();
  // sampling_param.do_sample = grpc_request.do_sample();
  // sampling_param.seed = grpc_request.seed();

  // construct stopping criteria
  auto& stopping_criteria = request->stopping_criteria;
  stopping_criteria.max_tokens = max_tokens;
  stopping_criteria.max_context_len =
      max_context_len - FLAGS_num_speculative_tokens;
  // stopping_criteria.ignore_eos_token = false;
  stopping_criteria.eos_token_id = model_args.eos_token_id();

  // use stop token ids if specified in the request
  if (grpc_request.stop_token_ids_size() > 0) {
    const auto& stop_token_ids = grpc_request.stop_token_ids();
    stopping_criteria.stop_token_ids.insert(stop_token_ids.begin(),
                                            stop_token_ids.end());
  } else {
    // otherwise use stop token ids from model args
    stopping_criteria.stop_token_ids = model_args.stop_token_ids();
  }

  // construct stop sequences
  for (const auto& stop_seq : grpc_request.stop()) {
    // encode stop sequence
    std::vector<int> token_ids;
    if (!tokenizer.encode(stop_seq, &token_ids)) {
      call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                   "Failed to encode stop sequence");
      LOG(ERROR) << "Failed to encode stop sequence: " << stop_seq;
      return nullptr;
    }
    stopping_criteria.stop_sequences.push_back(token_ids);
  }

  if (grpc_request.has_stream()) {
    request->stream = grpc_request.stream();
  }
  if (grpc_request.has_priority()) {
    request->priority = grpc_priority_to_priority(grpc_request.priority());
  }
  // disable echo for chat completion
  request->echo = false;

  // set callback for outputs
  request->on_output = [call_data,
                        request = request.get(),
                        first_message = std::vector<bool>(num_seqs, true)](
                           const RequestOutput& req_output) mutable -> bool {
    if (req_output.finished) {
      return send_result_to_client(call_data, request, req_output);
    }
    // send delta to client
    return send_delta_to_client(call_data, request, &first_message, req_output);
  };

  // add one sequence, the rest will be expanded by scheduler
  request->add_sequence();
  return request;
}

}  // namespace

ChatHandler::ChatHandler(Scheduler* scheduler, const Engine* engine)
    : scheduler_(scheduler) {
  CHECK(scheduler_ != nullptr);
  tokenizer_ = engine->tokenizer();
  model_args_ = engine->model_args();

  // construct chat template
  auto factory = ModelRegistry::get_default_chat_template_factory(
      model_args_.model_type());
  if (!FLAGS_enable_jinja_chat_template && factory) {
    LOG(INFO) << "Using default chat template for model type: "
              << model_args_.model_type();
    chat_template_ = factory();
  } else {
    const auto& tokenizer_args = engine->tokenizer_args();
    if (!tokenizer_args.chat_template().empty()) {
      LOG(INFO) << "Using jinja chat template: "
                << absl::CEscape(tokenizer_args.chat_template());
      chat_template_ = std::make_unique<JinjaChatTemplate>(
          tokenizer_args.chat_template(), /*add_generation_prompt=*/true);
    }
  }
}

void ChatHandler::chat_async(ChatCallData* call_data) {
  converter_threadpool_.schedule([this, call_data = call_data]() {
    if (!verify_request_arguments(call_data)) {
      // request is not valid, finish with error
      return;
    }

    auto request = grpc_request_to_request(
        call_data, chat_template_.get(), *tokenizer_, model_args_);
    if (request == nullptr) {
      return;
    }

    // schedule the request
    if (!scheduler_->schedule(request)) {
      call_data->finish_with_error(grpc::StatusCode::RESOURCE_EXHAUSTED,
                                   "Out of capacity");
    }
  });
}

}  // namespace llm
