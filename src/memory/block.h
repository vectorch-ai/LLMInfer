#pragma once

#include <cstdint>

namespace llm {

// forward declaration
class BlockAllocator;

// Memory block represents a contiguous memory region.
// It is used to track memory usage. the block will be released when the
// reference count drops to zero.
class Block final {
 public:
  ~Block();

  // add default constructor to allow resizing with std::vector
  Block() = default;

  // used for testing
  Block(int32_t id);
  Block(int32_t id, uint32_t size);

  Block(int32_t id, BlockAllocator* allocator);

  // copy constructor and assignment operator
  Block(const Block& other);
  Block& operator=(const Block& other);

  // move related operations
  Block(Block&& other) noexcept;
  Block& operator=(Block&& other) noexcept;

  // get the block id
  int32_t id() const { return id_; }

  // get the block size
  uint32_t size() const { return size_; }

  // get the reference count, 0 if the block is invalid after move
  uint32_t ref_count() const { return ref_count_ == nullptr ? 0 : *ref_count_; }

  // check if the block is shared
  bool is_shared() const { return ref_count() > 1; }

  // check if the block is valid
  bool is_valid() const { return id_ >= 0 && ref_count_ != nullptr; }

 private:
  // increase reference count
  void inc_ref_count();

  // decrease reference count
  void dec_ref_count();

  // block id
  int32_t id_ = -1;

  // block size
  uint32_t size_ = 0;

  // reference count
  uint32_t* ref_count_ = nullptr;

  // allocator that manages this block
  BlockAllocator* allocator_ = nullptr;
};

// equeal operator, mainly used for testing
inline bool operator==(const Block& lhs, const Block& rhs) {
  return lhs.id() == rhs.id();
}

}  // namespace llm