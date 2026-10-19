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
 * Static-analysis design/implementation, 2026-10-19: kernel arguments use
 * named fields to distinguish operand and dimension roles. Core I.30
 * (#ri-encapsulate) confines native pointer/array access to the guarded kernel;
 * GDSA.2 preserves the exact partial ownership and arithmetic order. Source:
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 */
#include "kernels/cuda/common.cuh"

namespace dolittle::kernels::cuda {
namespace {
inline constexpr unsigned tile_width = 16;
inline constexpr unsigned padded_tile_width = 17;
inline constexpr unsigned wide_partials = 32;
inline constexpr unsigned half_tile = 8;
inline constexpr std::size_t small_input_limit = 4096;
inline constexpr std::size_t narrow_output_limit = 1024;

template <class T> struct ProjectionArguments {
    const T *input;
    const T *weight;
    T *output;
    std::size_t rows;
    std::size_t input_width;
    std::size_t output_width;
    std::size_t output_tiles;
    std::size_t tasks;
};

// Device-only indexing: preparation proves allocation extents; the loops
// below guard tails and index shared/register arrays within their declared sizes.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-constant-array-index)
template <class T, unsigned Partials> __global__ void projection(ProjectionArguments<T> args) {
    const auto [input, weight, output, rows, input_width, output_width, output_tiles, tasks] = args;
    __shared__ float x_tile[tile_width][tile_width];
    __shared__ float w_tile[tile_width][padded_tile_width];
    const auto x = threadIdx.x;
    const auto y = threadIdx.y;
    for (std::size_t task = blockIdx.x; task < tasks;) {
        const auto row = (task / output_tiles) * tile_width + y;
        const auto column = (task % output_tiles) * tile_width + x;
        const auto weight_row = (task % output_tiles) * tile_width + y;
        float partials[Partials]{};
        const auto reduction_tiles = input_width / tile_width + (input_width % tile_width != 0);
        for (std::size_t tile = 0; tile < reduction_tiles; ++tile) {
            const auto reduction = tile * tile_width + x;
            x_tile[y][x] = row < rows && reduction < input_width
                               ? detail::widen(input[row * input_width + reduction])
                               : 0.0F;
            w_tile[x][y] = weight_row < output_width && reduction < input_width
                               ? detail::widen(weight[weight_row * input_width + reduction])
                               : 0.0F;
            __syncthreads();
            if constexpr (Partials == wide_partials) {
                if (tile % 2 != 0) {
#pragma unroll
                    for (unsigned k = 0; k < tile_width; ++k) {
                        partials[k + tile_width] =
                            __fmaf_rn(x_tile[y][k], w_tile[k][x], partials[k + tile_width]);
                    }
                } else {
#pragma unroll
                    for (unsigned k = 0; k < tile_width; ++k) {
                        partials[k] = __fmaf_rn(x_tile[y][k], w_tile[k][x], partials[k]);
                    }
                }
            } else {
#pragma unroll
                for (unsigned k = 0; k < tile_width; ++k) {
                    partials[k] = __fmaf_rn(x_tile[y][k], w_tile[k][x], partials[k]);
                }
            }
            __syncthreads();
        }
        if constexpr (Partials == wide_partials) {
#pragma unroll
            for (unsigned k = 0; k < tile_width; ++k) {
                partials[k] = __fadd_rn(partials[k], partials[k + tile_width]);
            }
        }
#pragma unroll
        for (unsigned k = 0; k < half_tile; ++k) {
            partials[k] = __fadd_rn(partials[k], partials[k + half_tile]);
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
        if (tasks - task <= gridDim.x) {
            break;
        }
        task += gridDim.x;
    }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-constant-array-index)

template <class T>
void submit(const void *input, const void *weight, void *output, std::size_t rows, std::size_t input_width,
            std::size_t output_width, cudaStream_t stream) {
    const auto output_tiles = detail::ceil_div(output_width, tile_width);
    const auto tasks = detail::ceil_div(rows, tile_width) * output_tiles;
    if (input_width <= small_input_limit && output_width > narrow_output_limit) {
        projection<T, tile_width><<<detail::grid_blocks(tasks), dim3(tile_width, tile_width), 0, stream>>>(
            ProjectionArguments<T>{.input = static_cast<const T *>(input),
                                   .weight = static_cast<const T *>(weight),
                                   .output = static_cast<T *>(output),
                                   .rows = rows,
                                   .input_width = input_width,
                                   .output_width = output_width,
                                   .output_tiles = output_tiles,
                                   .tasks = tasks});
    } else {
        projection<T, wide_partials><<<detail::grid_blocks(tasks), dim3(tile_width, tile_width), 0, stream>>>(
            ProjectionArguments<T>{.input = static_cast<const T *>(input),
                                   .weight = static_cast<const T *>(weight),
                                   .output = static_cast<T *>(output),
                                   .rows = rows,
                                   .input_width = input_width,
                                   .output_width = output_width,
                                   .output_tiles = output_tiles,
                                   .tasks = tasks});
    }
}
} // namespace

void launch_projection(tensors::ElementType type, const void *input, const void *weight, void *output,
                       std::size_t rows, std::size_t input_width, std::size_t output_width,
                       cudaStream_t stream) {
    if (detail::is_bfloat16(type)) {
        submit<__nv_bfloat16>(input, weight, output, rows, input_width, output_width, stream);
    } else {
        submit<float>(input, weight, output, rows, input_width, output_width, stream);
    }
}
} // namespace dolittle::kernels::cuda
