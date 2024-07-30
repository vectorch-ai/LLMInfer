/*
 * Copyright (C) 2024 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All
 * Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Adapted from https://github.com/IST-DASLab/Sparse-Marlin

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace marlin {

namespace {
constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

// Instances of `Vec` are used to organize groups of >>registers<<, as needed
// for instance as inputs to tensor core operations. Consequently, all
// corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee
// this.
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

// Matrix fragments for tensor core instructions; their precise layout is
// documented here:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragM = Vec<uint, 1>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>;  // quantization scales

// Predicated asynchronous global->shared copy; used for inputs A where we apply
// predication to handle batchsizes that are not multiples of 16.
__device__ inline void cp_async4_pred(void* smem_ptr,
                                      const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem),
      "l"(glob_ptr),
      "n"(BYTES));
}

// Asynchronous global->shared copy with a cache hint indicating that the values
// may be evicted immediately; used for quantized weights B, which are only
// accessed precisely once and should thus not pollute the L2 cache which we
// need for inputs A and outputs C.
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .b64 p;\n"
      "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
      "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr),
      "n"(BYTES));
}

// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
__device__ inline void mma(const FragA& a_frag,
                           const FragB& frag_b,
                           FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]),
        "r"(a[1]),
        "r"(a[2]),
        "r"(a[3]),
        "r"(b[0]),
        "r"(b[1]),
        "f"(c[0]),
        "f"(c[1]),
        "f"(c[2]),
        "f"(c[3]));
}

// m16n8k32 sparse tensor core mma instruction with fp16 inputs and fp32
// output/accumulation.
__device__ inline void mma_sp(const FragB& a_frag0,
                              const FragB& a_frag1,
                              const FragA& frag_b,
                              FragC& frag_c,
                              FragM& frag_m,
                              const int psel) {
  const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a_frag0);
  const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a_frag1);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* e = reinterpret_cast<const uint32_t*>(&frag_m);
  float* c = reinterpret_cast<float*>(&frag_c);
  if (psel == 0) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
        "{%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a0[0]),
          "r"(a1[0]),
          "r"(a0[1]),
          "r"(a1[1]),
          "r"(b[0]),
          "r"(b[2]),
          "r"(b[4]),
          "r"(b[6]),
          "f"(c[0]),
          "f"(c[1]),
          "f"(c[2]),
          "f"(c[3]),
          "r"(e[0]));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
        "{%12,%13,%14,%15}, %16, 0x0;\n"
        : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
        : "r"(a0[0]),
          "r"(a1[0]),
          "r"(a0[1]),
          "r"(a1[1]),
          "r"(b[1]),
          "r"(b[3]),
          "r"(b[5]),
          "r"(b[7]),
          "f"(c[4]),
          "f"(c[5]),
          "f"(c[6]),
          "f"(c[7]),
          "r"(e[0]));
  } else {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
        "{%12,%13,%14,%15}, %16, 0x1;\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a0[0]),
          "r"(a1[0]),
          "r"(a0[1]),
          "r"(a1[1]),
          "r"(b[0]),
          "r"(b[2]),
          "r"(b[4]),
          "r"(b[6]),
          "f"(c[0]),
          "f"(c[1]),
          "f"(c[2]),
          "f"(c[3]),
          "r"(e[0]));
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10,%11}, "
        "{%12,%13,%14,%15}, %16, 0x1;\n"
        : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
        : "r"(a0[0]),
          "r"(a1[0]),
          "r"(a0[1]),
          "r"(a1[1]),
          "r"(b[1]),
          "r"(b[3]),
          "r"(b[5]),
          "r"(b[7]),
          "f"(c[4]),
          "f"(c[5]),
          "f"(c[6]),
          "f"(c[7]),
          "r"(e[0]));
  }
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

__device__ inline void ldsm4_m(FragM& frag_m, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_m);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  /* asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  ); */
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(a[0]), "=r"(a[1])
               : "r"(smem));
}

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

__device__ __forceinline__ uint2 to_half4(float c0,
                                          float c1,
                                          float c2,
                                          float c3) {
  uint2 r;
  asm("{\n\t"
      ".reg .f16 a, b, c, d; \n\t"
      "cvt.rn.f16.f32 a, %2; \n\t"
      "cvt.rn.f16.f32 b, %3; \n\t"
      "cvt.rn.f16.f32 c, %4; \n\t"
      "cvt.rn.f16.f32 d, %5; \n\t"
      "mov.b32 %0, {a, b};   \n\t"
      "mov.b32 %1, {c, d};   \n\t"
      "}"
      : "=r"(r.x), "=r"(r.y)
      : "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  return r;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}

template <const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks = -1  // number of consecutive 16x16 blocks
                                       // with a separate quantization scale
          >
__global__ void Marlin_2_4(
    const int4* __restrict__ A,     // fp16 input matrix of shape mxk
    const int4* __restrict__ B,     // 4bit quantized weight matrix of shape kxn
    const int4* __restrict__ meta,  // 2bit metadata information about 2:4
                                    // format on B
    int4* __restrict__ C,           // fp16 output buffer of shape mxn
    const int4* __restrict__ s,     // fp16 quantization scales of shape
                                    // (k/groupsize)xn
    int prob_m,                     // batch dimension m
    int prob_n,                     // output dimension n
    int prob_k,                     // reduction dimension k
    int* locks  // extra global storage for barrier synchronization
) {
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.

  // For larger GEMMs we run multiple batchsize 64 versions in parallel for a
  // better partitioning with less reductions
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

  int k_tiles =
      prob_k / 32 / thread_k_blocks;  // number of thread_k_blocks in k-dim
  int n_tiles =
      prob_n / 16 / thread_n_blocks;  // number of thread_n_blocks in n-dim
  int iters = ceildiv(k_tiles * n_tiles * parallel,
                      gridDim.x);  // iters needeed to cover all slices
  // Ensure that the number of tiles in each stripe is a multiple of the
  // groupsize; this avoids an annoying special case where a stripe starts in
  // the middle of group.
  if (group_blocks != -1)
    iters = (group_blocks / thread_k_blocks) *
            ceildiv(iters, (group_blocks / thread_k_blocks));

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;  // number of threadblock tiles in the current slice
  int slice_count =
      0;          // total number of active threadblocks in the current slice
  int slice_idx;  // index of threadblock in current slice; numbered bottom to
                  // top

  // We can easily implement parallel problem execution by just remapping
  // indices and advancing global pointers
  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
  }

  // Compute all information about the current slice which is required for
  // synchronization.
  auto init_slice = [&]() {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;
    }
  };
  init_slice();

  // RLC: 8 is vec_size -> 128-bit instructions, 8 fp16 elements
  int a_gl_stride = prob_k / 8;  // stride of the A matrix in global memory
  // We typically use `constexpr` to indicate that this value is a compile-time
  // constant
  constexpr int a_sh_stride =
      32 * thread_k_blocks / 8;  // stride of an A matrix tile in shared memory
  constexpr int a_gl_rd_delta_o =
      32 * thread_k_blocks /
      8;  // delta between subsequent A tiles in global memory
  int a_gl_rd_delta_i =
      a_gl_stride *
      (threads / a_gl_rd_delta_o);  // between subsequent accesses within a tile
  constexpr int a_sh_wr_delta =
      a_sh_stride *
      (threads / a_gl_rd_delta_o);  // between shared memory writes
  constexpr int a_sh_rd_delta_o =
      4 * ((threads / 32) /
           (thread_n_blocks /
            4));  // between shared memory tile reads //RLC: 2 * #warps k-dim
  constexpr int a_sh_rd_delta_i =
      a_sh_stride * 16;  // within a shared memory tile
  constexpr int a_sh_stage =
      a_sh_stride * (16 * thread_m_blocks);  // overall size of a tile
  constexpr int a_sh_wr_iters =
      ceildiv(a_sh_stage,
              a_sh_wr_delta);  // number of shared write iterations for a tile

  int b_gl_stride = 16 * prob_n / 32;
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  constexpr int b_sh_wr_delta = threads;
  constexpr int b_sh_rd_delta = threads;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  int m_gl_stride = 2 * prob_n / 8;  // (16*2*4 / 8) = 16
  constexpr int m_sh_stride =
      (16 * thread_n_blocks) / 4;  // #warps n-dim * threads/warp
  int m_gl_rd_delta_o = m_gl_stride * thread_k_blocks;
  int m_gl_rd_delta_i = m_gl_stride * (threads / m_sh_stride);
  constexpr int m_sh_wr_delta = threads / 2;
  constexpr int m_sh_rd_delta = threads / 2;
  constexpr int m_sh_stage = m_sh_stride * thread_k_blocks;
  constexpr int m_sh_iters = ceildiv(m_sh_stage, m_sh_wr_delta);

  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  constexpr int s_sh_stage = s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 4 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd =
      b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x;
  int b_sh_rd = threadIdx.x;

  int m_gl_rd = m_gl_stride * (threadIdx.x / (m_sh_stride)) +
                (threadIdx.x % (m_sh_stride));
  m_gl_rd += (m_sh_stride)*slice_col;
  m_gl_rd += m_gl_rd_delta_o * slice_row;
  int m_sh_wr = threadIdx.x;
  int m_sh_rd = threadIdx.x % 16 + (threadIdx.x / 32) * 16;

  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                s_sh_stride * slice_col + threadIdx.x;
  int s_sh_wr = threadIdx.x;
  int s_sh_rd;
  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  if (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) / 4;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) / 4;

  // Precompute which thread should not read memory in which iterations; this is
  // needed if there are more threads than required for a certain tilesize or
  // when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++) {
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  }
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[2][b_sh_wr_iters][thread_m_blocks];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
    for (int j = 0; j < thread_m_blocks; j++) {
      a_sh_rd_trans[0][i][j] =
          transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
      a_sh_rd_trans[1][i][j] =
          transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd + 2);
    }
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependicies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.
  const int4* B_ptr[b_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  bool m_sh_wr_pred = threadIdx.x < m_sh_wr_delta;
  const int4* meta_ptr[m_sh_iters];
#pragma unroll
  for (int i = 0; i < m_sh_iters; i++)
    meta_ptr[i] = meta + m_gl_rd_delta_i * i + m_gl_rd;

  extern __shared__ int4 sh[];
  // Shared memory storage for global fetch pipelines.
  int4* sh_a = sh;
  int4* sh_b = sh_a + (stages * a_sh_stage);
  int4* sh_s = sh_b + (stages * b_sh_stage);
  int4* sh_m = sh_s + (stages * s_sh_stage);
  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks][2];
  I4 frag_b_quant[2];
  FragM frag_m[2][2];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];

  // Zero accumulators.
  auto zero_accums = [&]() {
#pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
        B_ptr[i] += b_gl_rd_delta_o;
      }
      int4* sh_meta_stage = sh_m + m_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < m_sh_iters; i++) {
        if (m_sh_wr_pred)
          cp_async4_stream(&sh_meta_stage[m_sh_wr_delta * i + m_sh_wr],
                           meta_ptr[i]);
        meta_ptr[i] += m_gl_rd_delta_o;
      }
      // Only fetch scales if this tile starts a new group
      if (group_blocks != -1 && pipe % (group_blocks / thread_k_blocks) == 0) {
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;
        if (s_sh_wr_pred) cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
        s_gl_rd += s_gl_rd_delta;
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    // It may seem inefficient that we reload the groups for every sub-tile;
    // however, this does not seem to be a significant bottleneck, while some
    // theoretically better attempts have lead to bad instruction ordering by
    // the compiler and correspondingly a noticable drop in performance.
    if (group_blocks != -1) {
      int4* sh_s_stage =
          sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) *
                               (pipe / (group_blocks / thread_k_blocks)));
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
    }
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
      ldsm4(frag_a[k % 2][i][0],
            &sh_a_stage[a_sh_rd_trans[0][k % b_sh_wr_iters][i]]);
      ldsm4(frag_a[k % 2][i][1],
            &sh_a_stage[a_sh_rd_trans[1][k % b_sh_wr_iters][i]]);
    }

    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(
        &sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);

    // Load meta with ldsm4
    int4* sh_m_stage = sh_m + m_sh_stage * pipe;
    ldsm4_m(frag_m[k % 2][0],
            &sh_m_stage[m_sh_rd_delta * (k % m_sh_iters) + m_sh_rd]);
  };

  // Execute the actual tensor core matmul of a sub-tile.
  auto matmul = [&](int k) {
// We have the m dimension as the inner loop in order to encourage overlapping
// dequantization and matmul operations.
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int b_quant = frag_b_quant[k % 2][j];
      int b_quant_shift = b_quant >> 8;
      FragB frag_b0 = dequant(b_quant);
      // If there are no groups, we can just scale the final output once and can
      // avoid doing so for each weight.
      if (group_blocks != -1) scale(frag_b0, frag_s[k % 2][j], 0);
      FragB frag_b1 = dequant(b_quant_shift);
      if (group_blocks != -1) scale(frag_b1, frag_s[k % 2][j], 1);
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma_sp(frag_b0,
               frag_b1,
               frag_a[k % 2][i][0],
               frag_c[i][j][0],
               frag_m[k % 2][j / 2],
               j % 2);
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) +
                      (threadIdx.x % b_sh_stride);

      // Parallel logarithmic shared memory reduction. We make sure to avoid any
      // unnecessary read or write iterations, e.g., for two warps we write only
      // once by warp 1 and read only once by warp 0.
#pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
#pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd =
                    reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
#pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] =
                  reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
#pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd =
                reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
#pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] +=
                  c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we
  // finally have to globally reduce over the results. As the striped partioning
  // minimizes the number of such reductions and our outputs are usually rather
  // small, we perform this reduction serially in L2 cache.
  auto global_reduce = [&](bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to
    // maximize L2 cache utilization in this step. To do this, we write out
    // results in FP16 (but still reduce with FP32 compute).
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 2 * 4 * c_gl_stride;
      int c_gl_wr_delta_i =
          c_gl_stride;  // 8 threads (e.g., 0,4,8,12,16,20,24,28)
      int c_gl_wr = 2 * c_gl_stride * (threadIdx.x % 4) +
                    8 * (threadIdx.x / 32) + (threadIdx.x % 32) / 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      int c_sh_wr = threadIdx.x;

      int col = 2 * ((threadIdx.x % 32) % 4);

      if (!first) {
// Interestingly, doing direct global accesses here really seems to mess up the
// compiler and lead to slowdowns, hence we also use async-copies even though
// these fetches are not actually asynchronous.
#pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred(&sh[c_sh_wr + c_sh_wr_delta * i],
                         &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                            c_gl_wr_delta_i * (i % 2)],
                         i < (thread_m_blocks - 1) * 4 ||
                             8 * (i / 2) + col + (i % 2) < prob_m);
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

#pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 ||
            8 * (i / 2) + col + (i % 2) < prob_m) {
          if (!first) {
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
#pragma unroll
            for (int j2 = 0; j2 < 2; j2++) {
#pragma unroll
              for (int j1 = 0; j1 < 4; j1++) {
                reinterpret_cast<float*>(
                    &frag_c)[4 * 2 * 4 * (i / 4) + 8 * j1 + 2 * j2 +
                             4 * ((i % 4) / 2) + i % 2] +=
                    __half2float(
                        reinterpret_cast<__half*>(&c_red)[(j2 * 4 + j1)]);
              }
            }
          }
          if (!last) {
            int4 c;
#pragma unroll
            for (int j2 = 0; j2 < 2; j2++) {
#pragma unroll
              for (int j1 = 0; j1 < 4; j1++) {
                reinterpret_cast<__half*>(&c)[(j2 * 4 + j1)] =
                    __float2half(reinterpret_cast<float*>(
                        &frag_c)[4 * 2 * 4 * (i / 4) + 8 * j1 + 2 * j2 +
                                 4 * ((i % 4) / 2) + i % 2]);
              }
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] =
                c;
          }
        }
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  auto write_result = [&]() {
    int c_gl_stride = prob_n / 8;

    constexpr int c_sh_stride = 2 * thread_n_blocks;              // RLC:
    constexpr int c_sh_stride_2 = 2 * c_sh_stride + 2;            // RLC:
    constexpr int c_sh_stride_3 = 2 * (2 * thread_n_blocks) + 2;  // RLC:

    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;

    int c_sh_wr = c_sh_stride_2 * ((threadIdx.x % 32) % 4) +
                  ((threadIdx.x % 32) / 4);  // RLC:
    c_sh_wr += 8 * (threadIdx.x / 32);       // 128/4(half4)

    constexpr int c_sh_rd_delta =
        c_sh_stride_3 * (threads / (2 * 2 * thread_n_blocks));  // RLC:
    int c_sh_rd = c_sh_stride_3 * (threadIdx.x / (2 * 2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * 2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    int col = 2 * ((threadIdx.x % 32) % 4);

    auto write = [&](int idx,
                     float c0,
                     float c1,
                     float c2,
                     float c3,
                     FragS& s0,
                     float c4,
                     float c5,
                     float c6,
                     float c7,
                     FragS& s1) {
      uint2 res[2];
      res[0] = to_half4(c0, c1, c2, c3);
      res[1] = to_half4(c4, c5, c6, c7);
      half2* tmp = (half2*)&res;
      if (group_blocks ==
          -1) {  // for per-column quantization we finally apply the scale here
        tmp[0] = __hmul2(tmp[0], s0[0]);
        tmp[1] = __hmul2(tmp[1], s0[1]);
        tmp[2] = __hmul2(tmp[2], s1[0]);
        tmp[3] = __hmul2(tmp[3], s1[1]);
      }
      ((int4*)sh)[idx] = *((int4*)&res[0]);
    };

    if (threadIdx.x / 32 <
        thread_n_blocks / 4) {  // RLC:  only warp 0 and 1 baseline example
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        int wr = c_sh_wr;
        write(wr,
              frag_c[i][0][0][0],
              frag_c[i][1][0][0],
              frag_c[i][2][0][0],
              frag_c[i][3][0][0],
              frag_s[0][0],
              frag_c[i][0][0][2],
              frag_c[i][1][0][2],
              frag_c[i][2][0][2],
              frag_c[i][3][0][2],
              frag_s[0][2]);
        // if((col+1)<prob_m){
        write(wr + c_sh_stride,
              frag_c[i][0][0][1],
              frag_c[i][1][0][1],
              frag_c[i][2][0][1],
              frag_c[i][3][0][1],
              frag_s[0][0],
              frag_c[i][0][0][3],
              frag_c[i][1][0][3],
              frag_c[i][2][0][3],
              frag_c[i][3][0][3],
              frag_s[0][2]);
        // if((col+8)<prob_m){
        write(wr + 4 * c_sh_stride_2,
              frag_c[i][0][1][0],
              frag_c[i][1][1][0],
              frag_c[i][2][1][0],
              frag_c[i][3][1][0],
              frag_s[0][0],
              frag_c[i][0][1][2],
              frag_c[i][1][1][2],
              frag_c[i][2][1][2],
              frag_c[i][3][1][2],
              frag_s[0][2]);
        // if((col+9)<prob_m){
        write(wr + 4 * c_sh_stride_2 + c_sh_stride,
              frag_c[i][0][1][1],
              frag_c[i][1][1][1],
              frag_c[i][2][1][1],
              frag_c[i][3][1][1],
              frag_s[0][0],
              frag_c[i][0][1][3],
              frag_c[i][1][1][3],
              frag_c[i][2][1][3],
              frag_c[i][3][1][3],
              frag_s[0][2]);
        //}
        //}
        //}
        c_sh_wr += 8 * c_sh_stride_2;
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0;
         i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {
#pragma unroll
    for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  start_pipes();

  // Main loop.
  while (slice_iters) {
// We unroll over both the global fetch and the register load pipeline to ensure
// all shared memory accesses are static. Note that both pipelines have even
// length meaning that the next iteration will always start at index 0.
#pragma unroll
    for (int pipe = 0; pipe < stages;) {
      // cp_async_wait<0>();
      fetch_to_shared(
          (pipe + stages - 1) % stages, pipe, slice_iters >= stages);
      wait_for_stage();

      fetch_to_registers(pipe + 1, (pipe + 1) % stages);
      matmul(pipe);

      pipe++;
      slice_iters--;
      if (slice_iters == 0) break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compliation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before
      // write-out
      if (group_blocks == -1 && last) {
        if (s_sh_wr_pred) cp_async4_stream(&sh_s[s_sh_wr], &s[s_gl_rd]);
        cp_async_fence();
      }
      thread_block_reduce();
      if (group_blocks == -1 && last) {
        cp_async_wait<0>();
        __syncthreads();
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
          *(float4*)(frag_s) = *(float4*)(&sh_s[s_sh_rd]);
        }
      }
      if (slice_count > 1) {  // only globally reduce if there is more than one
                              // block in a slice
        barrier_acquire(&locks[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[slice_col], last);
      }
      if (last)  // only the last block in a slice actually writes the result
        write_result();

      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
#pragma unroll
        for (int i = 0; i < m_sh_iters; i++)
          meta_ptr[i] += (m_sh_stride)-m_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
#pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] -= b_gl_stride;
#pragma unroll
          for (int i = 0; i < m_sh_iters; i++) meta_ptr[i] -= m_gl_stride;
        }
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
const int THREADS = 256;
const int STAGES = 4;  // 4 pipeline stages fit into shared memory
const int SHARED_MEM =
    96 * 1024;  // max shared memory on compute capability 8.6 (< 8.0)

#define CALL_IF_2_4(                                                          \
    THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS)          \
  else if (thread_m_blocks == THREAD_M_BLOCKS &&                              \
           thread_n_blocks == THREAD_N_BLOCKS &&                              \
           thread_k_blocks == THREAD_K_BLOCKS &&                              \
           group_blocks == GROUP_BLOCKS) {                                    \
    cudaFuncSetAttribute(Marlin_2_4<THREADS,                                  \
                                    THREAD_N_BLOCKS,                          \
                                    THREAD_M_BLOCKS,                          \
                                    THREAD_K_BLOCKS,                          \
                                    STAGES,                                   \
                                    GROUP_BLOCKS>,                            \
                         cudaFuncAttributeMaxDynamicSharedMemorySize,         \
                         SHARED_MEM);                                         \
    Marlin_2_4<THREADS,                                                       \
               THREAD_N_BLOCKS,                                               \
               THREAD_M_BLOCKS,                                               \
               THREAD_K_BLOCKS,                                               \
               STAGES,                                                        \
               GROUP_BLOCKS><<<blocks, THREADS, SHARED_MEM, stream>>>(        \
        A_ptr, B_ptr, meta_ptr, C_ptr, s_ptr, prob_n, prob_m, prob_k, locks); \
  }
}  // namespace

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

int marlin_sparse(const void* A,
                  const void* B,
                  const void* meta,
                  void* C,
                  void* s,
                  int prob_m,
                  int prob_n,
                  int prob_k,
                  void* workspace,
                  int groupsize = -1,
                  int dev = 0,
                  cudaStream_t stream = 0,
                  int thread_k = -1,
                  int thread_m = -1,
                  int sms = -1,
                  int max_par = 16) {
  int tot_n = prob_n;
  int tot_n_blocks = ceildiv(tot_n, 16);
  int pad = 16 * tot_n_blocks - tot_n;

  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  if (thread_k == -1 || thread_m == -1) {
    if (prob_n <= 16) {
      // For small batchizes, better partioning is slightly more important than
      // better compute utilization
      thread_k = 128;
      thread_m = 128;
    } else {
      thread_k = 64;
      thread_m = 256;
    }
  }

  int thread_k_blocks = thread_k / 32;  // 2:4 version with m16n8k32 instruction
  int thread_m_blocks = thread_m / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;
  if (prob_m % thread_m != 0 || prob_k % thread_k != 0 ||
      (group_blocks != -1 && (prob_k / 2) % group_blocks != 0)) {
    return ERR_PROB_SHAPE;
  }
  if (prob_m == 0 || prob_n == 0 || prob_k == 0) {
    return 0;
  }
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  const int4* meta_ptr = (const int4*)meta;
  int4* C_ptr = (int4*)C;
  const int4* s_ptr = (const int4*)s;

  int cols = prob_m / thread_m;
  int* locks = (int*)workspace;
  int ret = 0;
  for (int i = 0; i < tot_n_blocks; i += 4) {
    int thread_n_blocks = tot_n_blocks - i;
    prob_n = tot_n - 16 * i;
    int par = 1;
    if (thread_n_blocks > 4) {
      // Note that parallel > 1 currently only works for inputs without any
      // padding
      par = (16 * thread_n_blocks - pad) / 64;
      if (par > max_par) par = max_par;
      prob_n = 64 * par;
      i += 4 * (par - 1);
      thread_n_blocks = 4;
    }

    // For compilation speed, we only define the kernel configurations that have
    // seemed useful (in terms of performance) in our testing, however many more
    // are, in principle, possible.
    if (false) {
    }                          //       BMxBNxBK,   group
    CALL_IF_2_4(8, 1, 4, -1)   // e.g., 16x128x128
    CALL_IF_2_4(8, 1, 4, 4)    // e.g., 16x128x128, 64
    CALL_IF_2_4(16, 1, 2, -1)  // e.g., 16x256x64
    CALL_IF_2_4(16, 1, 2, 4)   // e.g., 16x256x64,  64
    CALL_IF_2_4(16, 2, 2, -1)  // e.g.. 32x256x64
    CALL_IF_2_4(16, 2, 2, 4)
    CALL_IF_2_4(16, 3, 2, -1)
    CALL_IF_2_4(16, 3, 2, 4)
    CALL_IF_2_4(16, 4, 2, -1)
    CALL_IF_2_4(16, 4, 2, 4)
    else {
      ret = ERR_KERN_SHAPE;
    }

    A_ptr += 16 * thread_n_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_n_blocks * (prob_m / 8) * par;
  }

  return ret;
}

}  // namespace marlin