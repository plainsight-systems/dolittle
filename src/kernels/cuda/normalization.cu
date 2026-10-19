/*
 * Axis X5: RMS normalization over each vector's final dimension.
 * Role: normalize residual vectors before attention/MLP, or individual Q/K
 * head vectors before rotary embedding.
 *
 * Inputs: X contains R independent vectors of D features; W contains D learned
 * scales shared by all vectors; epsilon is positive. For residuals R is the
 * number of tokens. For Q/K, R is tokens times heads and D is the head width.
 * Output: Y has the same shape and storage type as X.
 *
 * Algorithm: for each vector r:
 * 1. Convert its features to FP32 and compute
 *    mean_square = sum_j(X[r,j] * X[r,j]) / D, for j in [0,D).
 * 2. Compute inverse_rms = 1 / sqrt(mean_square + epsilon).
 * 3. Normalize each feature: normalized = X[r,j] * inverse_rms.
 * 4. Round normalized to the input storage type, then multiply it by W[j]
 *    in FP32. Round that product to storage when writing Y[r,j].
 * Vectors are independent. In BF16 mode, step 4 preserves the intermediate
 * rounding before the learned scale is applied; rounding only the final
 * product would compute a different result.
 *
 * CUDA implementation: a 256-thread block processes one vector at a time.
 * Threads accumulate squares for their assigned features, then sum those
 * partials through 256 shared FP32 values. They reread X to write normalized
 * features. All threads reach the barriers, including those without a feature.
 * Blocks advance through additional vectors when R exceeds the grid size.
 * Cost: O(R*D) work and 256 shared floats per block.
 * Optimization: adjacent threads read/write adjacent features; the reduction
 * uses bounded scratch instead of a full tensor of squared values.
 * FP32 reduction order is not a promise of bitwise agreement across backends.
 *
 * Overlay: docs/architecture/transformer-execution.md,
 * "One Qwen3 transformer block".
 * Reference: transformers v4.57.3, Qwen3RMSNorm.forward:
 * https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3/modeling_qwen3.py.
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
 */
#include "kernels/cuda/common.cuh"

namespace dolittle::kernels::cuda {
namespace {
template <class T>
__global__ void rms_norm(const T* input, const T* weight, T* output,
                         std::size_t rows, std::size_t width, float epsilon) {
    __shared__ float scratch[detail::threads];
    for (std::size_t row = blockIdx.x; row < rows;) {
        float sum = 0.0F;
        for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
            const auto value = detail::widen(input[row * width + column]);
            sum = fmaf(value, value, sum);
        }
        const auto total = detail::block_sum(sum, scratch);
        const auto inverse = rsqrtf(__fadd_rn(total / static_cast<float>(width), epsilon));
        for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
            const auto normalized = detail::storage_round<T>(
                __fmul_rn(detail::widen(input[row * width + column]), inverse));
            output[row * width + column] = detail::narrow<T>(
                __fmul_rn(normalized, detail::widen(weight[column])));
        }
        __syncthreads();
        if (rows - row <= gridDim.x) break;
        row += gridDim.x;
    }
}

template <class T>
void submit(const void* input, const void* weight, void* output, std::size_t rows,
            std::size_t width, float epsilon, cudaStream_t stream) {
    rms_norm<<<detail::grid_blocks(rows), detail::threads, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(weight),
        static_cast<T*>(output), rows, width, epsilon);
}
}  // namespace

void launch_rms_norm(tensors::ElementType type, const void* input, const void* weight,
                     void* output, std::size_t rows, std::size_t width,
                     float epsilon, cudaStream_t stream) {
    if (detail::is_bfloat16(type)) {
        submit<__nv_bfloat16>(input, weight, output, rows, width, epsilon, stream);
    } else {
        submit<float>(input, weight, output, rows, width, epsilon, stream);
    }
}
}  // namespace dolittle::kernels::cuda
