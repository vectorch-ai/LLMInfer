#pragma once
#include <absl/strings/match.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <string>

#include "chat_template/coded_chat_template.h"
#include "layers/activation.h"
#include "layers/attention/attention.h"
#include "layers/attention/handler.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "layers/qkv_linear.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "models/parameters.h"

// gemma model compatible with huggingface weight
namespace llm::hf {

class GemmaMLPImpl : public torch::nn::Module {
 public:
  GemmaMLPImpl(const ModelArgs& args,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               const torch::TensorOptions& options) {
    act_with_mul_ =
        Activation::get_act_with_mul_func(args.hidden_act(), options.device());
    CHECK(act_with_mul_ != nullptr);

    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    // register the weight parameter
    gate_up_proj_ =
        register_module("gate_up_proj",
                        ColumnParallelLinear(hidden_size,
                                             intermediate_size * 2,
                                             /*bias=*/false,
                                             /*gather_output=*/false,
                                             quant_args,
                                             parallel_args,
                                             options));
    down_proj_ =
        register_module("down_proj",
                        RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          /*bias=*/false,
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          options));
  }

  torch::Tensor forward(torch::Tensor x) {
    return down_proj_(act_with_mul_(gate_up_proj_(x)));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    gate_up_proj_->load_state_dict(state_dict, {"gate_proj.", "up_proj."});
    down_proj_->load_state_dict(state_dict.select("down_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    gate_up_proj_->verify_loaded_weights(prefix + "[gate_proj,up_proj].");
    down_proj_->verify_loaded_weights(prefix + "down_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};

  // calculate act(x) * y
  ActFunc act_with_mul_{nullptr};
};
TORCH_MODULE(GemmaMLP);

class GemmaAttentionImpl : public torch::nn::Module {
 public:
  GemmaAttentionImpl(const ModelArgs& args,
                     const QuantArgs& quant_args,
                     const ParallelArgs& parallel_args,
                     const torch::TensorOptions& options,
                     AttentionHandler* handler) {
    const int32_t world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t n_kv_heads = args.n_kv_heads().value_or(n_heads);
    const int64_t head_dim = args.head_dim();
    const int64_t n_local_heads = n_heads / world_size;
    const int64_t n_local_kv_heads =
        std::max<int64_t>(1, n_kv_heads / world_size);

    // size for q, k, v
    qkv_sizes_ = {n_local_heads * head_dim,
                  n_local_kv_heads * head_dim,
                  n_local_kv_heads * head_dim};

    // register submodules
    qkv_proj_ = register_module("qkv_proj",
                                QKVColumnParallelLinear(hidden_size,
                                                        n_heads,
                                                        n_kv_heads,
                                                        head_dim,
                                                        /*bias=*/false,
                                                        /*gather_output=*/false,
                                                        quant_args,
                                                        parallel_args,
                                                        options));

    o_proj_ = register_module("o_proj",
                              RowParallelLinear(n_heads * head_dim,
                                                hidden_size,
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                options));

    // initialize attention
    atten_ = register_module(
        "atten", Attention(n_local_heads, n_local_kv_heads, head_dim, handler));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_local_heads * head_dim)
    // => (num_tokens, n_local_heads * head_dim)
    auto qkv = qkv_proj_(x).split(/*split_size=*/qkv_sizes_, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);

    // calculate attention,
    // output: (num_tokens, n_local_heads*head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return o_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    qkv_proj_->load_state_dict(
        state_dict, {"q_proj.", "k_proj.", "v_proj."}, {"k_proj.", "v_proj."});
    o_proj_->load_state_dict(state_dict.select("o_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    qkv_proj_->verify_loaded_weights(prefix + "[q_proj,k_proj,v_proj].");
    o_proj_->verify_loaded_weights(prefix + "o_proj.");
  }

 private:
  // parameter members, must be registered
  QKVColumnParallelLinear qkv_proj_{nullptr};

  RowParallelLinear o_proj_{nullptr};

  // module members without parameters
  Attention atten_{nullptr};

  // size for q, k, v
  std::vector<int64_t> qkv_sizes_;
};
TORCH_MODULE(GemmaAttention);

class GemmaDecoderLayerImpl : public torch::nn::Module {
 public:
  GemmaDecoderLayerImpl(const ModelArgs& args,
                        const QuantArgs& quant_args,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options,
                        AttentionHandler* handler) {
    // register submodules
    self_attn_ = register_module(
        "self_attn",
        GemmaAttention(args, quant_args, parallel_args, options, handler));

    mlp_ = register_module("mlp",
                           GemmaMLP(args, quant_args, parallel_args, options));

    input_layernorm_ = register_module(
        "input_layernorm",
        RMSNormResidual(args.hidden_size(), args.rms_norm_eps(), options));

    post_attention_layernorm_ = register_module(
        "post_attention_layernorm",
        RMSNormResidual(args.hidden_size(), args.rms_norm_eps(), options));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params,
                        torch::Tensor& residual) {
    auto hidden_states = input_layernorm_(x, residual);

    hidden_states =
        self_attn_(hidden_states, positions, kv_cache, input_params);

    // fully connected
    hidden_states = post_attention_layernorm_(hidden_states, residual);

    return mlp_(hidden_states);
  }
  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    input_layernorm_->load_state_dict((state_dict.select_with_transform(
        "input_layernorm.",
        [](const std::string_view& /*name*/, const torch::Tensor& tensor) {
          return tensor + 1.0f;
        })));
    mlp_->load_state_dict(state_dict.select("mlp."));
    post_attention_layernorm_->load_state_dict(
        (state_dict.select_with_transform(
            "post_attention_layernorm.",
            [](const std::string_view& /*name*/, const torch::Tensor& tensor) {
              return tensor + 1.0f;
            })));
    self_attn_->load_state_dict(state_dict.select("self_attn."));
  }
  void verify_loaded_weights(const std::string& prefix) const {
    self_attn_->verify_loaded_weights(prefix + "self_attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    input_layernorm_->verify_loaded_weights(prefix + "input_layernorm.");
    post_attention_layernorm_->verify_loaded_weights(
        prefix + "post_attention_layernorm.");
  }

 private:
  GemmaAttention self_attn_{nullptr};

  GemmaMLP mlp_{nullptr};

  RMSNormResidual input_layernorm_{nullptr};

  RMSNormResidual post_attention_layernorm_{nullptr};
};
TORCH_MODULE(GemmaDecoderLayer);

class GemmaModelImpl : public torch::nn::Module {
 public:
  GemmaModelImpl(const ModelArgs& args,
                 const QuantArgs& quant_args,
                 const ParallelArgs& parallel_args,
                 const torch::TensorOptions& options) {
    modelArgs_ = args;
    // register submodules
    embed_tokens_ = register_module(
        "embed_tokens",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    // normalize the embedding by sqrt(hidden_size)
    // N.B. the data type of the normalizer should be the same as the embedding
    // ref to:
    // https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/gemma_causal_lm.py#L426
    const float normalizer = std::sqrt(args.hidden_size());
    normalizer_ =
        register_buffer("normalizer", torch::tensor({normalizer}, options));

    norm_ = register_module(
        "norm",
        RMSNormResidual(args.hidden_size(), args.rms_norm_eps(), options));

    handler_ = AttentionHandler::create_handler_with_rope(
        args, /*interleaved=*/false, options);

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = GemmaDecoderLayer(
          args, quant_args, parallel_args, options, handler_.get());
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    // embedding tokens
    auto h = embed_tokens_(tokens) * normalizer_;

    torch::Tensor residual;
    for (int32_t i = 0; i < modelArgs_.n_layers(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params, residual);
    }

    return norm_(h, residual);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(state_dict.select("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
    // GemmaRMSNorm is different from Llama's in that it multiplies
    // (1 + weight) to the output, instead of just weight.
    norm_->load_state_dict((state_dict.select_with_transform(
        "norm.",
        [](const std::string_view& /*name*/, const torch::Tensor& tensor) {
          return tensor + 1.0f;
        })));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

 private:
  ModelArgs modelArgs_;

  // parameter members, must be registered
  // embedding module
  ParallelEmbedding embed_tokens_{nullptr};

  // embedding normalizer
  torch::Tensor normalizer_{nullptr};

  RMSNormResidual norm_{nullptr};
  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<GemmaDecoderLayer> layers_;
};
TORCH_MODULE(GemmaModel);

class GemmaForCausalLMImpl : public torch::nn::Module {
 public:
  GemmaForCausalLMImpl(const ModelArgs& args,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options) {
    // register submodules

    model_ = register_module(
        "model", GemmaModel(args, quant_args, parallel_args, options));

    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
                                                    /*bias=*/false,
                                                    /*gather_output=*/true,
                                                    parallel_args,
                                                    options));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // return: [num_tokens,hidden_size]
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    // select tokens if provided
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }

    return lm_head_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict.select("model."));

    // Share the embedding weights with the final llm_head layer.
    lm_head_->load_state_dict(state_dict.select("model.embed_tokens."));
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights("model.");
    lm_head_->verify_loaded_weights("model.embed_tokens.");
  }

 private:
  // parameter members, must be registered
  GemmaModel model_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(GemmaForCausalLM);

class GemmaChatTemplate final : public CodedChatTemplate {
 public:
  std::optional<std::string> get_prompt(
      const std::string_view& /*system_message*/,
      const std::vector<std::string_view>& messages) const override {
    // at least one user message
    if (messages.size() % 2 == 0) {
      return std::nullopt;
    }
    // prompt:https://huggingface.co/google/gemma-2b-it
    /*
     * <bos><start_of_turn>user
     * Write a hello world program<end_of_turn>
     * <start_of_turn>model
     */
    std::stringstream ss;
    // then user message
    for (size_t i = 0; i < messages.size(); i++) {
      ss << "\n<start_of_turn> user\n" << messages[i] << "<end_of_turn>";
    }

    ss << "\n<start_of_turn> model";
    return ss.str();
  }
};

// register the model to make it available
REGISTER_CAUSAL_MODEL(gemma, GemmaForCausalLM);

REGISTER_DEFAULT_CHAT_TEMPLATE(gemma, GemmaChatTemplate);

REGISTER_MODEL_ARGS(gemma, [&] {
  // example config from
  // https://huggingface.co/google/gemma-2b/blob/main/config.json
  LOAD_ARG_OR(model_type, "model_type", "gemma");
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 2);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 16384);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 8192);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 8);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 18);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 1);  // MQA
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "vocab_size", 256000);

  LOAD_ARG_OR_FUNC(hidden_act, "hidden_activation", [&] {
    const auto hidden_act = json.value<std::string>("hidden_act");
    if (hidden_act.has_value()) {
      LOG(WARNING) << "Gemma's activation function was initially released with "
                      "an incorrect setting. Override the "
                      "activation function from '"
                   << hidden_act.value() << "' to 'gelu_pytorch_tanh'";
    }
    return "gelu_pytorch_tanh";
  });

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

}  // namespace llm::hf