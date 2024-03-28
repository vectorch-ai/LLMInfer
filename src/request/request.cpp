#include "request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "sequence.h"

namespace llm {

Request::Request(const std::string& id,
                 const std::string_view& prompt,
                 size_t n,
                 const std::vector<int32_t>& prompt_tokens)
    : id(id),
      prompt(prompt),
      num_seqs(n),
      created_time(absl::ToUnixSeconds(absl::Now())),
      prompt_tokens(prompt_tokens) {}

Request::Request(const std::string& id,
                 const std::vector<int32_t>& prompt_tokens)
    : Request(id, /*prompt=*/"", /*n=*/1, prompt_tokens) {}

void Request::add_sequence() {
  OnDelta on_delta = nullptr;

  if (stream) {
    CHECK(on_stream_delta);
    on_delta = [this, index = sequences.size(), first_message = true](
                   const std::string& delta, FinishReason reason) mutable {
      bool ret = this->on_stream_delta(index, first_message, delta, reason);
      first_message = false;
      return ret;
    };
  }
  sequences.emplace_back(this->prompt,
                         this->prompt_tokens,
                         this->sampling_param,
                         this->stopping_criteria,
                         this->echo,
                         on_delta);
}

bool Request::is_finished() const {
  // still need to generate more sequences
  if (sequences.size() < num_seqs) {
    return false;
  }

  for (const auto& seq : sequences) {
    if (!seq.is_finished()) {
      return false;
    }
  }
  return true;
}

bool Request::should_expand_sequences() const {
  if (sequences.size() < num_seqs) {
    CHECK(!sequences.empty());
    const auto& first_sequence = sequences.front();
    // if all prompt tokens are in kv cache, then expand
    return first_sequence.num_tokens_in_kv_cache() >=
           first_sequence.num_prompt_tokens();
  }
  return false;
}

void Request::expand_sequences() {
  while (sequences.size() < num_seqs) {
    add_sequence();
  }
}

}  // namespace llm
