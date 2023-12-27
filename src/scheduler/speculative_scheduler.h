#pragma once

#include <absl/time/time.h>
#include <folly/MPMCQueue.h>

#include <cstdint>
#include <memory>
#include <queue>

#include "engine/engine.h"
#include "memory/block_manager.h"
#include "request/request.h"
#include "scheduler.h"

namespace llm {

class SpeculativeScheduler final : public Scheduler {
 public:
  SpeculativeScheduler(Engine* llm_engine, Engine* ssm_engine);

  ~SpeculativeScheduler() override;

  // schedule a request, thread safe and non-blocking
  // may return false if the queue is full
  bool schedule(std::unique_ptr<Request>& request) override;

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step(const absl::Duration& timeout) override;
 
 private:
  // get a batch of requests from the priority queue
  void build_sequence_batch();

  void on_request_finish(Request* request);

  void on_sequence_stream(Sequence* seq);

 private:
  // the engine to run ssm
  Engine* ssm_engine_;

  // the engine to run llm
  Engine* llm_engine_;

  // the llm block manager to manage the cache blocks
  BlockManager* llm_block_manager_;

  // the ssm block manager to manage the cache blocks
  BlockManager* ssm_block_manager_; 

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  // a thread safe queue of requests, bounded by kRequestQueueSize
  // the schedule owns the requests and manages their lifetimes.
  folly::MPMCQueue<Request*> request_queue_;

  // Requests with HIGH priority are processed first, followed by MEDIUM
  // priority requests, and finally LOW priority requests. Within each priority
  // level, requests are handled on First-Come-First-Served (FCFS) basis.
  using MinHeap =
      std::priority_queue<Request*, std::vector<Request*>, RequestPtrGreater>;
  MinHeap priority_queue_;

  // a batch of requests to be processed, sorted by priority from high to low.
  std::vector<Request*> request_batch_;

  // a batch of sequence to be processed.
  std::vector<Sequence*> sequences_batch_;

  // preemptable requests that hold cache slots, sorted by priority from high to
  // low.
  std::deque<Request*> preemptable_candidates_;

  // the threadpool to handle responses
  ThreadPool response_threadpool_;
};

}  // namespace llm
