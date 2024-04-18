#include "process_group.h"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

namespace llm {
namespace {

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NCCLCHECK(cmd)                                               \
  do {                                                               \
    ncclResult_t r = cmd;                                            \
    if (r != ncclSuccess) {                                          \
      LOG(FATAL) << "Failed, NCCL error :" << ncclGetErrorString(r); \
    }                                                                \
  } while (0)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CUDACHECK(cmd)                                                 \
  do {                                                                 \
    cudaError_t err = cmd;                                             \
    if (err != cudaSuccess) {                                          \
      LOG(FATAL) << "Failed, Cuda error :" << cudaGetErrorString(err); \
    }                                                                  \
  } while (0)

at::Tensor flatten_for_scatter_gather(std::vector<at::Tensor>& tensors) {
  auto& t = tensors[0];
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}

ncclDataType_t to_nccl_data_type(const torch::Tensor& input) {
  const auto type = input.scalar_type();
  switch (type) {
    case at::kFloat:
      return ncclDataType_t::ncclFloat;
    case at::kHalf:
      return ncclDataType_t::ncclHalf;
    case at::kDouble:
      return ncclDataType_t::ncclDouble;
    case at::kLong:
      return ncclDataType_t::ncclInt64;
    case at::kInt:
      return ncclDataType_t::ncclInt;
    case at::kChar:
      return ncclDataType_t::ncclChar;
    case at::kByte:
      return ncclDataType_t::ncclUint8;
    case at::kBool:
      return ncclDataType_t::ncclUint8;
    case at::kBFloat16:
      return ncclDataType_t::ncclBfloat16;
    default:
      TORCH_CHECK(false, "Unconvertible NCCL type ", type);
  }
}

void check_input(torch::Tensor input) {
  CHECK(input.is_cuda()) << "input should be cuda tensor";
  CHECK(input.is_contiguous()) << "input should be contiguous";
  CHECK(!input.is_sparse()) << "input have to be cuda dense tensor";
}

}  // namespace

std::vector<std::unique_ptr<ProcessGroup>> ProcessGroup::create_process_groups(
    const std::vector<torch::Device>& devices) {
  CHECK(!devices.empty()) << "devices should not be empty";
  // all devices should be cuda devices
  for (const auto& device : devices) {
    CHECK(device.is_cuda()) << "device should be cuda device";
  }

  std::vector<int> device_idxs;
  device_idxs.reserve(devices.size());
  for (const auto& device : devices) {
    device_idxs.push_back(device.index());
  }

  std::vector<ncclComm_t> comms(devices.size());
  const int world_size = static_cast<int>(devices.size());
  NCCLCHECK(ncclCommInitAll(comms.data(), world_size, device_idxs.data()));

  std::vector<std::unique_ptr<ProcessGroup>> process_groups;
  process_groups.reserve(devices.size());
  for (int i = 0; i < world_size; ++i) {
    process_groups.emplace_back(std::make_unique<ProcessGroupNCCL>(
        /*rank=*/i, world_size, devices[i], comms[i]));
  }
  return process_groups;
}

// Constructor.
ProcessGroupNCCL::ProcessGroupNCCL(int rank,
                                   int world_size,
                                   const torch::Device& device,
                                   ncclComm_t comm)
    : ProcessGroup(rank, world_size, device), comm_(comm) {}

// Destructor.
ProcessGroupNCCL::~ProcessGroupNCCL() { NCCLCHECK(ncclCommDestroy(comm_)); }

void ProcessGroupNCCL::allreduce(torch::Tensor& input) {
  DCHECK(input.device() == device())
      << "input should be on the same device as the process group";
  check_input(input);

  // inplace all reduce
  const auto count = input.numel();
  const auto data_type = to_nccl_data_type(input);

  auto stream = at::cuda::getCurrentCUDAStream();
  torch::DeviceGuard device_guard(device());
  NCCLCHECK(ncclAllReduce(
      /*sendbuff=*/input.data_ptr(),
      /*recvbuff=*/input.data_ptr(),
      /*count=*/count,
      /*datatype=*/data_type,
      /*op=*/ncclSum,
      /*comm=*/comm_,
      /*stream=*/stream));
}

void ProcessGroupNCCL::allgather(torch::Tensor input,
                                 std::vector<torch::Tensor>& outputs) {
  check_input(input);
  CHECK(outputs.size() == world_size())
      << "outputs should have the same size as world_size";
  DCHECK(input.device() == device())
      << "input should be on the same device as the process group";

  torch::DeviceGuard device_guard(device());
  torch::Tensor flattened_output = flatten_for_scatter_gather(outputs);

  const auto count = input.numel();
  const auto data_type = to_nccl_data_type(input);

  auto stream = at::cuda::getCurrentCUDAStream();
  NCCLCHECK(ncclAllGather(
      /*sendbuff=*/input.data_ptr(),
      /*recvbuff=*/flattened_output.data_ptr(),
      /*sendcount=*/count,
      /*datatype=*/data_type,
      /*comm=*/comm_,
      /*stream=*/stream));

  // copy the flattened output tensors to the outputs.
  for (int i = 0; i < outputs.size(); ++i) {
    outputs[i].copy_(flattened_output[i], /*non_blocking=*/true);
  }
}

}  // namespace llm
