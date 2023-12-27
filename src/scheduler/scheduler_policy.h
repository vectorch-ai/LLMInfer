#pragma once

#include <folly/MPMCQueue.h>

#include <cstdint>
#include <memory>

namespace llm {

class Request;
class Sequence; 
class SchedulerPolicy {
 public:
  virtual ~SchedulerPolicy() = default;

  virtual bool schedule(std::unique_ptr<Request>& request) = 0;
  virtual std::vector<Sequence*> build_batch() = 0;
};

class BlockManager;
class ResponseHandler;
class FCFSSchedulerPolicy final : public SchedulerPolicy {
 public:
  explicit FCFSSchedulerPolicy(ResponseHandler* response_handler,
                               BlockManager* block_manager);
  ~FCFSSchedulerPolicy() override;

  bool schedule(std::unique_ptr<Request>& request) override;
  std::vector<Sequence*> build_batch() override;

 private:
  ResponseHandler* response_handler_;
  BlockManager* block_manager_;

  folly::MPMCQueue<Request*> waiting_queue_;
  std::vector<Request*> blocking_queue_;
  std::vector<Request*> running_queue_;

  std::vector<Sequence*> running_batch_;
};

}  // namespace llm
