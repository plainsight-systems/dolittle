/*
 * Axis X5: residual addition and the MLP's SiLU gate multiplication.
 * Inputs: two equally shaped dense arrays, flattened to C elements.
 * Output: an equally shaped array in the same BF16 or FP32 storage type.
 *
 * Algorithm for Add, used at each residual connection:
 * 1. Read left[i] and right[i] as FP32, for each i in [0,C).
 * 2. Add them, round the sum to storage and write output[i].
 *
 * Algorithm for SiluMultiply, used between the MLP's up and down projections:
 * 1. Read a = gate[i] and b = up[i] as FP32.
 * 2. Compute sigmoid(a). For a >= 0 use 1/(1+exp(-a)); otherwise use
 *    exp(a)/(1+exp(a)). This avoids exponentiating a large positive value.
 * 3. Compute activated = a * sigmoid(a), then round it to the storage type.
 * 4. Multiply the rounded activation by b in FP32; round the product to
 *    storage and write output[i].
 * In BF16 mode the intermediate activation rounding is part of the result.
 *
 * CUDA implementation: each thread computes independent elements in a
 * strided loop over the flat array. Add versus SiluMultiply is selected at
 * compile time; neither path needs shared memory or synchronization.
 * Cost: O(C) work, reading two arrays and writing one.
 * Optimization: compute the SiLU activation and product in the same thread
 * without storing an intermediate activation array; retain its rounding step.
 * Adjacent threads access adjacent elements, with bounds checks at the tail.
 *
 * Overlay: docs/architecture/transformer-execution.md,
 * "One Qwen3 transformer block".
 *
 * Contract: preparation validates shapes, byte/launch extents and device limits.
 * Inputs and the separate output buffer are borrowed until queue completion;
 * output must not overlap an input. Launches allocate no buffers and perform
 * no host readback or host wait. The caller checks immediate launch status;
 * asynchronous device errors surface at the explicit queue wait. See
 * kernels/operations.h for the public operation and failure contracts.
 *
 * Corpus consultation, comment design, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): state the output and rounding guarantees.
 * cpp-perf-guidelines GPU.2: contiguous features map to adjacent threads.
 * Source: guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 * Static-analysis design/implementation, 2026-10-19: kernel arguments use
 * named fields to distinguish operand and dimension roles. Core I.30
 * (#ri-encapsulate) confines native pointer/array access to the guarded kernel;
 * GDSA.2 preserves the exact partial ownership and arithmetic order. Source:
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 */
#include "kernels/cuda/common.cuh"

namespace dolittle::kernels::cuda {
namespace {
template <class T> struct ElementwiseArguments {
    const T *left;
    const T *right;
    T *output;
    std::size_t count;
};

// Device-only indexing: preparation proves allocation extents; the loops
// below guard tails and index shared/register arrays within their declared sizes.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
template <class T, bool Silu> __global__ void elementwise(ElementwiseArguments<T> args) {
    const auto [left, right, output, count] = args;
    const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;) {
        const auto a = detail::widen(left[index]);
        const auto b = detail::widen(right[index]);
        float result = 0.0F;
        if constexpr (Silu) {
            const auto exponential = expf(a >= 0.0F ? -a : a);
            const auto sigmoid = a >= 0.0F ? 1.0F / (1.0F + exponential) : exponential / (1.0F + exponential);
            const auto activated = detail::storage_round<T>(__fmul_rn(a, sigmoid));
            result = __fmul_rn(activated, b);
        } else {
            result = __fadd_rn(a, b);
        }
        output[index] = detail::narrow<T>(result);
        if (count - index <= stride) {
            break;
        }
        index += stride;
    }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

template <class T, bool Silu>
void submit(const void *left, const void *right, void *output, std::size_t count, cudaStream_t stream) {
    elementwise<T, Silu>
        <<<detail::grid_blocks(detail::ceil_div(count, detail::threads)), detail::threads, 0, stream>>>(
            ElementwiseArguments<T>{.left = static_cast<const T *>(left),
                                    .right = static_cast<const T *>(right),
                                    .output = static_cast<T *>(output),
                                    .count = count});
}
} // namespace

void launch_add(tensors::ElementType type, const void *left, const void *right, void *output,
                std::size_t count, cudaStream_t stream) {
    if (detail::is_bfloat16(type)) {
        submit<__nv_bfloat16, false>(left, right, output, count, stream);
    } else {
        submit<float, false>(left, right, output, count, stream);
    }
}

void launch_silu_multiply(tensors::ElementType type, const void *gate, const void *up, void *output,
                          std::size_t count, cudaStream_t stream) {
    if (detail::is_bfloat16(type)) {
        submit<__nv_bfloat16, true>(gate, up, output, count, stream);
    } else {
        submit<float, true>(gate, up, output, count, stream);
    }
}
} // namespace dolittle::kernels::cuda
