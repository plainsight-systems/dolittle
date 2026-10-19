/*
 * Axis X5: dense projection without a bias term.
 * Role: compute the learned attention, MLP and vocabulary projections.
 *
 * Inputs: X[R,I] has R rows of I input features; W[O,I] has one learned weight
 * vector for each of O output features. Trailing head/feature axes are treated
 * as one feature axis; this does not copy or transpose the input.
 * Output: Y[R,O], using the same BF16 or FP32 storage type as X and W.
 *
 * Algorithm: for each row r and output feature o:
 * 1. Start an FP32 sum at zero.
 * 2. For each input feature k in [0,I), accumulate X[r,k] * W[o,k] into
 *    the sum, converting both operands to FP32 before multiplication.
 * 3. Round the completed sum to storage and write Y[r,o].
 * This computes Y = X * transpose(W). W remains stored as [O,I].
 *
 * CUDA implementation: a 16-by-16-thread block computes a 16-by-16 output
 * tile, with one output sum per thread. It advances through input features
 * 16 at a time, loading X and W into shared memory. W is transposed within
 * shared memory for consumption; its global representation does not change.
 * Out-of-range tile entries load as zero. Every thread participates in both
 * barriers per input tile; only valid outputs are stored. Accumulation uses
 * FP32 fused multiply-add, without TF32 substitution or a matrix library.
 * Cost: approximately 2*R*I*O arithmetic operations; 528 shared floats per
 * block, plus one FP32 sum per thread.
 * Optimization: threads load contiguous input features, then reuse each
 * shared tile across multiple output sums.
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
 */
#include "kernels/cuda/common.cuh"

namespace dolittle::kernels::cuda {
namespace {
template <class T>
__global__ void projection(const T* input, const T* weight, T* output,
                           std::size_t rows, std::size_t input_width,
                           std::size_t output_width, std::size_t output_tiles,
                           std::size_t tasks) {
    __shared__ float x_tile[16][16];
    __shared__ float w_tile[16][17];
    const auto x = threadIdx.x;
    const auto y = threadIdx.y;
    for (std::size_t task = blockIdx.x; task < tasks;) {
        const auto row = (task / output_tiles) * 16 + y;
        const auto column = (task % output_tiles) * 16 + x;
        const auto weight_row = (task % output_tiles) * 16 + y;
        float sum = 0.0F;
        const auto reduction_tiles = input_width / 16 + (input_width % 16 != 0);
        for (std::size_t tile = 0; tile < reduction_tiles; ++tile) {
            const auto reduction = tile * 16 + x;
            x_tile[y][x] = row < rows && reduction < input_width
                ? detail::widen(input[row * input_width + reduction]) : 0.0F;
            w_tile[x][y] = weight_row < output_width && reduction < input_width
                ? detail::widen(weight[weight_row * input_width + reduction]) : 0.0F;
            __syncthreads();
            for (unsigned k = 0; k < 16; ++k) {
                sum = fmaf(x_tile[y][k], w_tile[k][x], sum);
            }
            __syncthreads();
        }
        if (row < rows && column < output_width) {
            output[row * output_width + column] = detail::narrow<T>(sum);
        }
        if (tasks - task <= gridDim.x) break;
        task += gridDim.x;
    }
}

template <class T>
void submit(const void* input, const void* weight, void* output,
            std::size_t rows, std::size_t input_width, std::size_t output_width,
            cudaStream_t stream) {
    const auto output_tiles = detail::ceil_div(output_width, 16);
    const auto tasks = detail::ceil_div(rows, 16) * output_tiles;
    projection<<<detail::grid_blocks(tasks), dim3(16, 16), 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(weight),
        static_cast<T*>(output), rows, input_width, output_width, output_tiles, tasks);
}
}  // namespace

void launch_projection(tensors::ElementType type, const void* input, const void* weight,
                       void* output, std::size_t rows, std::size_t input_width,
                       std::size_t output_width, cudaStream_t stream) {
    if (detail::is_bfloat16(type)) {
        submit<__nv_bfloat16>(input, weight, output, rows, input_width, output_width, stream);
    } else {
        submit<float>(input, weight, output, rows, input_width, output_width, stream);
    }
}
}  // namespace dolittle::kernels::cuda
