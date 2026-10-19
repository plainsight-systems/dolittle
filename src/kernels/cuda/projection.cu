/*
 * Axis X5: bias-free dense projection for attention, MLP and vocabulary.
 * Inputs: X[R,I] and W[O,I]; output Y[R,O], all in BF16 or FP32 storage.
 * Algorithm: Y[r,o] = sum_k X[r,k]*W[o,k], with W stored as [O,I].
 * 1. Use P=16 for I<=4096 and O>1024; otherwise P=32. P depends only
 *    on feature widths, never on the number of rows or their scheduling.
 * 2. Partial j visits features j,j+P,j+2P,... in ascending order. Each
 *    update is one round-to-nearest FP32 fused multiply-add, starting at zero.
 * 3. Add upper partials into lower partials at offsets P/2,P/4,...,1.
 *    Each merge rounds to FP32. Round the completed sum to output storage.
 * CUDA implementation: 16-by-16 threads compute a 16-by-16 output tile.
 * Shared tiles stage 16 input features; the weight tile is transposed only
 * in shared memory. Each output thread carries P private FP32 partials.
 * For P=32, consecutive tiles update alternating halves of those partials.
 * Missing entries load zero; all threads reach both barriers per input tile.
 * Cost: about 2*R*I*O FLOPs, 528 shared floats and P registers per output.
 * Optimization: shared tiles reuse contiguous loads across outputs; constant
 * partial indices and merge bounds keep the accumulator array in registers.
 *
 * Overlay: docs/architecture/transformer-execution.md, "One Qwen3 transformer
 * block" and "Attention and positional semantics".
 * Contract: preparation validates shapes, extents and device limits. Inputs
 * and the separate output buffer are borrowed until queue completion; output
 * cannot overlap inputs. Launches allocate no buffers, read nothing to the
 * host and perform no host wait. The caller checks immediate launch status;
 * asynchronous errors surface at the explicit queue wait.
 * Same-build/device repeatability and row/chunk invariance are required.
 * Cross-GPU bit equality is unverified. No TF32 or FP64 tensor accumulation.
 * Corpus consultation, header design and implementation, 2026-10-19:
 * cpp-guidelines I.7 (#ri-post): specify intermediate and final rounding.
 * cpp-perf-guidelines GDSA.2: fix partial ownership and combination order
 * independently of batch size, scheduling and cache chunks.
 * Source: guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 * cpp-perf-guidelines GDSA.16: bound on-chip state and account for rereads.
 * Source: guidelines/gpu-dsa/GDSA.16-tile-through-on-chip-memory.md.
 */
#include "kernels/cuda/common.cuh"

namespace dolittle::kernels::cuda {
namespace {
template <class T, unsigned Partials>
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
        float partials[Partials]{};
        const auto reduction_tiles = input_width / 16 + (input_width % 16 != 0);
        for (std::size_t tile = 0; tile < reduction_tiles; ++tile) {
            const auto reduction = tile * 16 + x;
            x_tile[y][x] = row < rows && reduction < input_width
                ? detail::widen(input[row * input_width + reduction]) : 0.0F;
            w_tile[x][y] = weight_row < output_width && reduction < input_width
                ? detail::widen(weight[weight_row * input_width + reduction]) : 0.0F;
            __syncthreads();
            if constexpr (Partials == 32) {
                if (tile % 2 != 0) {
                    #pragma unroll
                    for (unsigned k = 0; k < 16; ++k) {
                        partials[k + 16] = __fmaf_rn(x_tile[y][k], w_tile[k][x], partials[k + 16]);
                    }
                } else {
                    #pragma unroll
                    for (unsigned k = 0; k < 16; ++k) {
                        partials[k] = __fmaf_rn(x_tile[y][k], w_tile[k][x], partials[k]);
                    }
                }
            } else {
                #pragma unroll
                for (unsigned k = 0; k < 16; ++k) {
                    partials[k] = __fmaf_rn(x_tile[y][k], w_tile[k][x], partials[k]);
                }
            }
            __syncthreads();
        }
        if constexpr (Partials == 32) {
            #pragma unroll
            for (unsigned k = 0; k < 16; ++k) {
                partials[k] = __fadd_rn(partials[k], partials[k + 16]);
            }
        }
        #pragma unroll
        for (unsigned k = 0; k < 8; ++k) {
            partials[k] = __fadd_rn(partials[k], partials[k + 8]);
        }
        #pragma unroll
        for (unsigned k = 0; k < 4; ++k) {
            partials[k] = __fadd_rn(partials[k], partials[k + 4]);
        }
        #pragma unroll
        for (unsigned k = 0; k < 2; ++k) {
            partials[k] = __fadd_rn(partials[k], partials[k + 2]);
        }
        partials[0] = __fadd_rn(partials[0], partials[1]);
        if (row < rows && column < output_width) {
            output[row * output_width + column] = detail::narrow<T>(partials[0]);
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
    if (input_width <= 4096 && output_width > 1024) {
        projection<T, 16><<<detail::grid_blocks(tasks), dim3(16, 16), 0, stream>>>(
            static_cast<const T*>(input), static_cast<const T*>(weight),
            static_cast<T*>(output), rows, input_width, output_width, output_tiles, tasks);
    } else {
        projection<T, 32><<<detail::grid_blocks(tasks), dim3(16, 16), 0, stream>>>(
            static_cast<const T*>(input), static_cast<const T*>(weight),
            static_cast<T*>(output), rows, input_width, output_width, output_tiles, tasks);
    }
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
