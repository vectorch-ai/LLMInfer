#include "block_manager.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "block_allocator.h"
#include "request/request.h"

namespace llm {
namespace {
// get the number of cache blocks to allocate for the sequence
size_t num_blocks_to_allocate(const Sequence& sequence, int32_t block_size) {
  if (sequence.is_finished()) {
    // no need to allocate more blocks for a finished sequence
    return 0;
  }
  const size_t num_tokens = sequence.num_tokens();
  const size_t num_blocks = sequence.num_blocks();
  // round up to the nearest block number
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed <= num_blocks) {
    // enough slots, don't need to allocate more
    return 0;
  }
  return num_blocks_needed - num_blocks;
}
}  // namespace
BlockManager::BlockManager(uint32_t num_blocks, int32_t block_size)
    : block_size_(block_size), block_allocator_(num_blocks, block_size) {}

// try to allocat slots for the request
bool BlockManager::allocate_slots_for_request(Request* request) {
  DCHECK(request != nullptr);
  uint32_t num_additional_blocks = 0;
  for (const auto& sequence : request->sequences) {
    num_additional_blocks += num_blocks_to_allocate(sequence, block_size_);
  }

  if (num_additional_blocks == 0) {
    // no need to allocate more blocks
    return true;
  }

  if (num_additional_blocks > block_allocator_.free_block_count()) {
    // not enough blocks in the block allocator
    // TODO: evict some blocks from the prefix cache then retry
    return false;
  }
  for (auto& sequence : request->sequences) {
    const uint32_t n_blocks = num_blocks_to_allocate(sequence, block_size_);
    const auto block_ids = block_allocator_.allocate(n_blocks);
    sequence.append_blocks(block_ids);
  }
  return true;
}

bool BlockManager::allocate_slots_for_sequence(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  const uint32_t num_additional_blocks =
      num_blocks_to_allocate(*sequence, block_size_);
  if (num_additional_blocks == 0) {
    // no need to allocate more blocks
    return true;
  }

  if (num_additional_blocks > block_allocator_.free_block_count()) {
    // not enough blocks
    return false;
  }
  const auto block_ids = block_allocator_.allocate(num_additional_blocks);
  sequence->append_blocks(block_ids);
  return true;
}

bool BlockManager::allocate_slots_for_sequences(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    if (!allocate_slots_for_sequence(sequence)) {
      // should we gurantee the atomicity of the allocation? all or nothing?
      return false;
    }
  }
  return true;
}

void BlockManager::release_slots_for_request(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences) {
    release_slots_for_sequence(&sequence);
  }
}

void BlockManager::release_slots_for_sequences(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    release_slots_for_sequence(sequence);
  }
}

void BlockManager::release_slots_for_sequence(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  const auto block_ids = sequence->release_blocks();
  // TODO: add the block to prefix cache
}

}  // namespace llm
