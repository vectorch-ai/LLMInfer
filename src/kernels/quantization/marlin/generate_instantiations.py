#!/usr/bin/env python3

# This file is run to generate the kernel instantiations for the marlin kernels
# They are written to several files in order to speed up compilation
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

# mapping from dtype to the corresponding CUDA type
DTYPE_MAP = {
    "fp16": "half",
    "bf16": "nv_bfloat16",
}

KERNEL_IMPL_TEMPLATE = """
// Splitting the different head dimensions to different files to speed up
// compilation. This file is auto-generated. See "generate_instantiations.py"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "gemm_kernel.cuh"

namespace marlin {{

template <>
void Marlin<{DTYPE},
            /*num_bits=*/{NUM_BITS},
            /*threads=*/{THREADS},
            /*thread_m_blocks=*/{M_BLOCKS},
            /*thread_n_blocks=*/{N_BLOCKS},
            /*thread_k_blocks=*/{K_BLOCKS},
            /*stages=*/{STAGES},
            /*has_act_order=*/{HAS_ACT_ORDER},
            /*has_zp=*/{HAS_ZP},
            /*group_blocks=*/{GROUP_BLOCKS}>(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int num_groups,       // number of scale groups per output channel
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
);

}}  // namespace marlin
"""


@dataclass
class Kernel:
    dtype: str
    num_bits: int
    threads: int
    m_blocks: int
    n_blocks: int
    k_blocks: int
    stages: int
    has_act_order: bool
    has_zp: bool
    group_blocks: int

    @property
    def template(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            DTYPE=DTYPE_MAP[self.dtype],
            NUM_BITS=self.num_bits,
            THREADS=self.threads,
            M_BLOCKS=self.m_blocks,
            N_BLOCKS=self.n_blocks,
            K_BLOCKS=self.k_blocks,
            STAGES=self.stages,
            HAS_ACT_ORDER="true" if self.has_act_order else "false",
            HAS_ZP="true" if self.has_zp else "false",
            GROUP_BLOCKS=self.group_blocks,
        )

    @property
    def filename(self) -> str:
        return f"marlin_{self.dtype}_b{self.num_bits}_t{self.threads}_m{self.m_blocks}_n{self.n_blocks}_k{self.k_blocks}_s{self.stages}_{self.has_act_order}_{self.has_zp}_g{self.group_blocks}.cu"


def get_gptq_kernels() -> List[Kernel]:
    for dtype in DTYPE_MAP.keys():
        for num_bits in [4, 8]:
            for m_blocks in [1, 2, 3, 4]:
                for n_blocks, k_blocks, threads in [
                    (16, 4, 256),
                    (8, 8, 256),
                    (8, 4, 128),
                    (4, 8, 128),
                ]:
                    for stages in [4]:
                        for has_act_order, group_blocks in [
                            (True, 0),
                            (False, -1),
                            (False, 2),
                            (False, 4),
                            (False, 8),
                        ]:
                            for has_zp in [False]:
                                yield Kernel(
                                    dtype=dtype,
                                    num_bits=num_bits,
                                    threads=threads,
                                    m_blocks=m_blocks,
                                    n_blocks=n_blocks,
                                    k_blocks=k_blocks,
                                    stages=stages,
                                    has_act_order=has_act_order,
                                    has_zp=has_zp,
                                    group_blocks=group_blocks,
                                )


def get_awq_kernels() -> List[Kernel]:
    for dtype in DTYPE_MAP.keys():
        for num_bits in [4, 8]:
            for m_blocks in [1, 2, 3, 4]:
                for n_blocks, k_blocks, threads in [
                    (16, 4, 256),
                    (8, 8, 256),
                    (8, 4, 128),
                    (4, 8, 128),
                ]:
                    for stages in [4]:
                        for group_blocks in [-1, 2, 4, 8]:
                            for has_act_order in [False]:
                                for has_zp in [False]:
                                    yield Kernel(
                                        dtype=dtype,
                                        num_bits=num_bits,
                                        threads=threads,
                                        m_blocks=m_blocks,
                                        n_blocks=n_blocks,
                                        k_blocks=k_blocks,
                                        stages=stages,
                                        has_act_order=has_act_order,
                                        has_zp=has_zp,
                                        group_blocks=group_blocks,
                                    )


def get_all_kernels() -> List[Kernel]:
    yield from get_gptq_kernels()
    yield from get_awq_kernels()


def write_kernel(kernel: Kernel, output_dir: Path) -> None:
    (output_dir / kernel.filename).write_text(kernel.template)


if __name__ == "__main__":
    output_dir = Path.cwd() / "generated"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)
