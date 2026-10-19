/*
 * Axis X5, private CUDA storage conversion and bounded launch primitives.
 * Contract: BF16 converts with round-to-nearest-even, FP32 is unchanged; explicit
 * storage rounding must survive operation fusion. Grid sizing takes positive
 * work counts, uses checked-by-preparation extents and never overflows ceil-div.
 * No allocation, transfers, waits or owned state. Device reductions require all
 * 256 lanes of a block to call with initialized shared values.
 * Algorithm responsibilities:
 * - Storage conversion: widen BF16 to FP32 for arithmetic. To round an
 *   intermediate, narrow FP32 to BF16 with round-to-nearest-even, then widen
 *   it again. For FP32 storage these conversions leave the value unchanged.
 * - Block sum: accept one FP32 partial sum from each of 256 threads. Write
 *   the partials to shared scratch, then repeatedly add the upper half into
 *   the lower half until scratch[0] contains the total. Synchronize after
 *   each stage; return the same total to every participating thread.
 * - Launch sizing: divide work into groups using integer ceil division and
 *   cap the grid at 65535 blocks. Kernel loops process work beyond that cap.
 * Overlay: docs/architecture/transformer-execution.md, "Model computation".
 * Cost: constant conversions; reduction O(log 256) barriers with 256 floats.
 * Optimization: shared helpers preserve explicit rounding without global
 * intermediate tensors or duplicated launch policy.
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): explicit rounding and
 * participation preconditions; unsupported dtype never silently falls back.
 * cpp-perf-guidelines GPU.2: contiguous lane reduction storage. Source:
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.

 * Static-analysis design/implementation, 2026-10-19:
 * The native shared-memory reduction bounds each index by the 256-lane block.
 * Its narrow indexing exception preserves the fixed reduction order.
 * cpp-guidelines SL.con.3 (#rsl-bounds), I.30 (#ri-encapsulate): make checked
 * host access explicit and confine necessary native API operations.
 * cpp-perf-guidelines COPY.7: borrow spans/references without hidden payload copies.
 * Source: guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 */
#pragma once

#include "kernels/cuda/launch.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace dolittle::kernels::cuda::detail {
inline constexpr unsigned threads = 256;
inline constexpr unsigned max_blocks = 65535;

inline std::size_t ceil_div(std::size_t value, std::size_t divisor) {
    return value / divisor + (value % divisor != 0);
}

inline unsigned grid_blocks(std::size_t tasks) {
    return static_cast<unsigned>(tasks < max_blocks ? tasks : max_blocks);
}

inline bool is_bfloat16(tensors::ElementType type) {
    if (type == tensors::ElementType::bfloat16) {
        return true;
    }
    if (type == tensors::ElementType::float32) {
        return false;
    }
    throw std::invalid_argument("unsupported CUDA floating storage type");
}

template <class T> __device__ float widen(T value) { return static_cast<float>(value); }

template <> __device__ inline float widen(__nv_bfloat16 value) { return __bfloat162float(value); }

template <class T> __device__ T narrow(float value) { return static_cast<T>(value); }

template <> __device__ inline __nv_bfloat16 narrow(float value) { return __float2bfloat16_rn(value); }

template <class T> __device__ float storage_round(float value) { return widen(narrow<T>(value)); }

// All 256 lanes participate; scratch has 256 elements and stride halves.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
__device__ inline float block_sum(float value, float *scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = threads / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
} // namespace dolittle::kernels::cuda::detail
