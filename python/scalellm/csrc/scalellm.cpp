#include <folly/init/Init.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_llm_engine.h"
#include "llm.h"
#include "sampling_params.h"

namespace llm {
namespace py = pybind11;

PYBIND11_MODULE(PY_MODULE_NAME, m) {
  // glog and glfag will be initialized in folly::init
  //   int argc = 0;
  //   char** argv = nullptr;
  //   folly::Init init(&argc, &argv);

  // class SamplingParameter
  py::class_<SamplingParams>(m, "SamplingParams")
      .def(py::init())
      .def_readwrite("frequency_penalty", &SamplingParams::frequency_penalty)
      .def_readwrite("presence_penalty", &SamplingParams::presence_penalty)
      .def_readwrite("repetition_penalty", &SamplingParams::repetition_penalty)
      .def_readwrite("temperature", &SamplingParams::temperature)
      .def_readwrite("top_p", &SamplingParams::top_p)
      .def_readwrite("top_k", &SamplingParams::top_k);

  py::class_<Statistics>(m, "Statistics")
      .def(py::init())
      .def_readwrite("num_prompt_tokens", &Statistics::num_prompt_tokens)
      .def_readwrite("num_generated_tokens", &Statistics::num_generated_tokens)
      .def_readwrite("num_total_tokens", &Statistics::num_total_tokens);

  py::class_<SequenceOutput>(m, "SequenceOutput")
      .def(py::init())
      .def_readwrite("index", &SequenceOutput::index)
      .def_readwrite("text", &SequenceOutput::text);
  //   .def_readwrite("finish_reason", &SequenceOutput::finish_reason);

  py::class_<RequestOutput>(m, "RequestOutput")
      .def(py::init())
      .def_readwrite("outputs", &RequestOutput::outputs)
      .def_readwrite("stats", &RequestOutput::stats)
      .def_readwrite("finished", &RequestOutput::finished);

  py::class_<_LLMEngine>(m, "_LLMEngine")
      .def(py::init<const std::string&, const std::string&>())
      .def("schedule_async",
           &_LLMEngine::schedule_async,
           py::call_guard<py::gil_scoped_release>())
      .def("run_forever", &_LLMEngine::run_forever)
      .def("stop", &_LLMEngine::stop)
      .def("run_until_complete", &_LLMEngine::run_until_complete);

  // class LLM
  py::class_<LLM, std::shared_ptr<LLM>>(m, "LLM")
      .def(py::init<const std::string&,
                    const SamplingParams&,
                    int64_t,
                    const std::string>())
      .def("generate", &LLM::generate);
}

}  // namespace llm