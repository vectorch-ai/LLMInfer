#include <absl/strings/str_split.h>
#include <c10/core/Device.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <csignal>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <thread>

#include "common/metrics.h"
#include "engine/llm_engine.h"
#include "grpc_server.h"
#include "handlers/chat_handler.h"
#include "handlers/completion_handler.h"
#include "handlers/models_handler.h"
#include "http_server.h"
#include "scheduler/continuous_scheduler.h"

using namespace llm;

DEFINE_string(model_id, "", "hf model name.");

DEFINE_string(model_path, "", "hf model path to the model file.");

DEFINE_string(device,
              "auto",
              "Device to run the model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or "
              "auto to use all available gpus.");

DEFINE_int32(http_port, 9999, "Port for http server.");
DEFINE_int32(grpc_port, 8888, "Port for grpc server.");

// NOLINTNEXTLINE
static std::atomic<uint32_t> signal_received{0};
void shutdown_handler(int signal) {
  // force exit after receiving third signal
  if (signal_received.fetch_add(1, std::memory_order_relaxed) >= 2) {
    LOG(ERROR) << "Received too many signals, force aborting...";
    exit(1);
  }
  LOG(WARNING) << "Received signal " << signal << ", stopping server...";
}

std::vector<torch::Device> parse_devices(const std::string& device_str) {
  std::vector<torch::Device> devices;
  if (device_str == "auto") {
    // use all available gpus if any
    const auto num_gpus = torch::cuda::device_count();
    if (num_gpus == 0) {
      LOG(INFO) << "no gpus found, using cpu.";
      return {torch::kCPU};
    }
    devices.reserve(num_gpus);
    for (int i = 0; i < num_gpus; ++i) {
      devices.emplace_back(torch::kCUDA, i);
    }
    return devices;
  }

  // parse device string
  const std::vector<std::string> device_strs = absl::StrSplit(device_str, ',');
  std::set<torch::DeviceType> device_types;
  devices.reserve(device_strs.size());
  for (const auto& device_str : device_strs) {
    devices.emplace_back(device_str);
    device_types.insert(devices.back().type());
  }
  CHECK(!devices.empty()) << "No devices specified.";
  CHECK(device_types.size() == 1)
      << "All devices must be of the same type. Got: " << FLAGS_device;
  return devices;
}

std::string to_string(const std::vector<torch::Device>& devices) {
  std::stringstream ss;
  for (size_t i = 0; i < devices.size(); ++i) {
    const auto& device = devices[i];
    if (i == 0) {
      ss << device;
    } else {
      ss << "," << device;
    }
  }
  return ss.str();
}

int main(int argc, char** argv) {
  // glog and glfag will be initialized in folly::init
  folly::Init init(&argc, &argv);
  google::InstallFailureSignalHandler();

  // check if model path exists
  if (!std::filesystem::exists(FLAGS_model_path)) {
    LOG(FATAL) << "Model path " << FLAGS_model_path << " does not exist.";
  }

  if (FLAGS_model_id.empty()) {
    // use last part of the path as model id
    FLAGS_model_id = std::filesystem::path(FLAGS_model_path).filename();
  }

  HttpServer http_server;
  http_server.register_uri("/gflags",
                           [](HttpServer::Transport& transport) -> bool {
                             auto gflags = nlohmann::json::array();
                             std::vector<google::CommandLineFlagInfo> flags;
                             google::GetAllFlags(&flags);
                             for (const auto& flag : flags) {
                               nlohmann::json gflag;
                               gflag["name"] = flag.name;
                               gflag["type"] = flag.type;
                               gflag["description"] = flag.description;
                               gflag["value"] = flag.current_value;
                               gflag["default"] = flag.default_value;
                               gflags.push_back(gflag);
                             }
                             return transport.send_string(
                                 gflags.dump(/*indent=*/2), "application/json");
                           });
  http_server.register_uri(
      "/metrics", [](HttpServer::Transport& transport) -> bool {
        return transport.send_string(Metrics::Instance().GetString());
      });
  http_server.register_uri(
      "/health", [](HttpServer::Transport& transport) -> bool {
        if (signal_received.load(std::memory_order_relaxed) == 0) {
          return transport.send_string("Ok\n");
        }
        // 503 Service Unavailable
        return transport.send_status(503);
      });

  // parse devices
  const auto devices = parse_devices(FLAGS_device);
  LOG(INFO) << "Using devices: " << to_string(devices);

  // create engine
  auto engine = std::make_unique<LLMEngine>(devices);
  CHECK(engine->init(FLAGS_model_path));

  // create scheduler and grpc handlers
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get());
  auto completion_handler =
      std::make_unique<CompletionHandler>(scheduler.get(), engine.get());
  auto chat_handler =
      std::make_unique<ChatHandler>(scheduler.get(), engine.get());
  auto models_handler = std::make_unique<ModelsHandler>(FLAGS_model_id);

  // start grpc server
  GrpcServer grpc_server(std::move(completion_handler),
                         std::move(chat_handler),
                         std::move(models_handler));
  GrpcServer::Options options;
  options.address = "0.0.0.0";
  options.port = FLAGS_grpc_port;

  if (!grpc_server.start(options)) {
    LOG(ERROR) << "failed to start grpc server on port " << FLAGS_grpc_port;
    return -1;
  }

  if (!http_server.start(FLAGS_http_port, /*num_threads=*/2)) {
    LOG(ERROR) << "Failed to start http server on port " << FLAGS_http_port;
    return -1;
  }

  // install graceful shutdown handler
  (void)signal(SIGINT, shutdown_handler);
  (void)signal(SIGTERM, shutdown_handler);

  const auto timeout = absl::Milliseconds(500);
  while (signal_received.load(std::memory_order_relaxed) == 0) {
    // move scheduler forward
    scheduler->step(timeout);
  }

  // stop grpc server and http server
  grpc_server.stop();
  http_server.stop();

  return 0;
}
