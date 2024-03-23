#include "block_manager.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "block_allocator.h"
#include "request/request.h"

DEFINE_bool(enable_prefix_cache,
            true,
            "enable the prefix cache for the block manager");

namespace llm {

BlockManager::BlockManager(uint32_t num_blocks, int32_t block_size)
    : block_size_(block_size),
      block_allocator_(num_blocks, block_size),
      prefix_cache_(block_size) {}

bool BlockManager::allocate_blocks_for(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  return allocate_blocks_for(sequence, sequence->num_tokens());
}

bool BlockManager::allocate_blocks_for(Sequence* sequence, size_t num_tokens) {
  DCHECK(sequence != nullptr);
  // first try to allocate shared blocks
  allocate_shared_blocks(sequence);

  const size_t num_blocks = sequence->num_blocks();
  // round up to the nearest block number
  const size_t num_blocks_needed = (num_tokens + block_size_ - 1) / block_size_;
  if (num_blocks_needed <= num_blocks) {
    return true;
  }

  const uint32_t num_additional_blocks = num_blocks_needed - num_blocks;
  if (!has_enough_blocks(num_additional_blocks)) {
    // not enough blocks
    return false;
  }

  const auto block_ids = block_allocator_.allocate(num_additional_blocks);
  sequence->append_blocks(block_ids);
  return true;
}

bool BlockManager::allocate_blocks_for(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    if (!allocate_blocks_for(sequence)) {
      // should we gurantee the atomicity of the allocation? all or nothing?
      return false;
    }
  }
  return true;
}

void BlockManager::release_blocks_for(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences) {
    release_blocks_for(&sequence);
  }
}

void BlockManager::release_blocks_for(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    release_blocks_for(sequence);
  }
}

void BlockManager::release_blocks_for(Sequence* sequence) {
  DCHECK(sequence != nullptr);

  if (FLAGS_enable_prefix_cache) {
    // only insert tokens in kv cache to the prefix cache
    const auto tokens_ids = sequence->tokens_in_kv_cache();
    const auto blocks = sequence->blocks();
    // Add the kv cache to the prefix cache
    prefix_cache_.insert(tokens_ids, blocks);
  }
  // release the blocks after prefix cache insertion
  sequence->release_blocks();
}

bool BlockManager::has_enough_blocks(uint32_t num_blocks) {
  // still have enough blocks
  if (num_blocks <= block_allocator_.free_block_count()) {
    return true;
  }

  // prefix cache is disabled, no way to evict blocks
  if (!FLAGS_enable_prefix_cache) {
    return false;
  }

  // try to evict some blocks from the prefix cache
  const uint32_t n_blocks_to_evict =
      num_blocks - block_allocator_.free_block_count();
  const uint32_t n_blocks_evicted = prefix_cache_.evict(n_blocks_to_evict);
  if (n_blocks_evicted < n_blocks_to_evict) {
    return false;
  }

  if (block_allocator_.free_block_count() >= num_blocks) {
    return true;
  }

  LOG(WARNING) << "Potential block leak, free blocks in allocator: "
               << block_allocator_.free_block_count()
               << " blocks in prefix cache: " << prefix_cache_.num_blocks();
  return false;
}

void BlockManager::allocate_shared_blocks(Sequence* sequence) {
  // only allocate shared blocks for prefill sequences
  if (FLAGS_enable_prefix_cache && sequence->num_blocks() == 0) {
    const auto& tokens_ids = sequence->token_ids();
    std::vector<Block> shared_blocks = prefix_cache_.match(tokens_ids);
    sequence->append_shared_blocks(shared_blocks);
  }
}

}  // namespace llm
