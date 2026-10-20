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
 * CUDA implementation: single-row decode assigns P adjacent lanes to one
 * output. Lane j owns partial j; shuffle-down offsets P/2,...,1 reproduce
 * the tree above. A 128-thread block owns 128/P output columns.
 * For multi-row prefill with I divisible by 128, each block stages eight
 * prompt rows by 128 features in 4096 bytes of shared memory. Lane j loads
 * W[o,k+j] once and applies it to eight separate row accumulators, using
 * staged X[r,k+j]. Advance k by P within each stage, then stage the next
 * 128 features. Two block barriers protect each stage. Reduce each row's
 * partials with the same shuffle tree and cast once to output storage.
 * Adjacent lanes read adjacent features; repeated half-warp reads of the
 * same staged input are broadcasts. Output row/column tails are guarded;
 * every lane still participates in barriers and shuffles. Device-owned
 * output_position offsets KV stores without host feedback.
 * Ragged input widths retain the 16-by-16 tiled kernel, with its original
 * padding to 16 features. The aligned path adds no zero FMAs. Partial count
 * depends only on feature widths, so row tiles and chunks preserve values.
 * Cost: about 2*R*I*O FLOPs; aligned weights are read once per eight rows,
 * while staged inputs are reused across 128/P columns. Row accumulators stay
 * in registers. No scratch allocation, readback or weight repacking.
 * Optimization: stage prompt inputs to avoid per-output global reloads,
 * reuse weights in registers, and reduce feature-stage barriers eightfold.
 * Measurements on the 3090 Ti selected eight rows and 128 features; larger
 * shared stages and register-only input loads lost on representative shapes.
 * Prefill design/implementation consultation, 2026-10-20: cpp-guidelines I.7
 * (#ri-post) preserves stored-value postconditions. cpp-perf-guidelines GPU.2
 * maps lanes to adjacent features, GPU.5 requires measured shared reuse and
 * barrier costs, and GDSA.2 fixes partial ownership and merge order. Both
 * corpora queried and rules read in each phase. Sources:
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md;
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md;
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 * References: llama.cpp ggml/src/ggml-cuda/mmvf.cu (cooperative GEMV),
 * https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvf.cu;
 * Charlotte src/core/kernels/matmul/matmul.wgsl (separate decode/prefill).
 * These inform lane mapping; Dolittle retains its own reduction contract.
 * Optimization corpus consultation, header design and implementation,
 * 2026-10-19: cpp-guidelines I.7 (#ri-post): preserve stored results across
 * dispatches. cpp-perf-guidelines GPU.2: adjacent lanes load adjacent weight
 * features. Source:
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 * cpp-perf-guidelines GPU.5: use shuffles instead of per-tile block barriers.
 * Source:
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md.
 * Optimization, design and implementation, 2026-10-20: aligned BF16 decode
 * with 4096<I<=UINT_MAX, I divisible by 32 and O<=4096 uses P/2 lanes per
 * output. Each lane
 * loads an adjacent BF16 pair and owns two original FP32 partials, even and
 * odd. Reduce the even and odd chains separately at offsets P/4,...,1,
 * then add even+odd. This is the same P-partial tree, with half as many
 * load/address instructions and twice as many outputs per block. Scalar
 * decode remains for other shapes/alignment. Full-model traces on the 3090 Ti
 * favored pairs for the wide MLP down projection, but scalar loads for
 * attention projections and the large MLP/vocabulary outputs.
 * No weight repacking, scratch allocation or numerical change.
 * Core Per.11 (#rper-comp): specialize the fixed partial count at compile time.
 * Performance GPU.2 (source above) and GDSA.2 (source below): coalesced pair
 * loads retain each partial's feature order and the original merge tree.
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
 * Device-loop extension (design/implementation, 2026-10-20): a prepared
 * prefill/decode body borrows an immutable-for-this-iteration device position.
 * The graph advances it only after every reader completes. RoPE/attention use
 * that absolute position; positioned output bindings address cache rows
 * S..S+B-1. Arithmetic and reduction order are unchanged. Graph storage is
 * admitted for the maximum position; the device loop enforces S+B <= capacity.
 * No per-token host argument update, copy of position, or host launch is
 * required. Core I.7/CP.3: explicit bounds and ordered borrowing. Performance
 * GPU.1/GPU.6: resident control and stable graph arguments
 * (guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md). Both corpora queried and rules
 * read during header design and implementation.
 */
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kernels/cuda/common.cuh"

namespace dolittle::kernels::cuda {
namespace {
inline constexpr unsigned tile_width = 16;
inline constexpr unsigned padded_tile_width = 17;
inline constexpr unsigned wide_partials = 32;
inline constexpr unsigned half_tile = 8;
inline constexpr unsigned decode_threads = 128;
inline constexpr unsigned full_warp_mask = 0xffffffffU;
inline constexpr std::size_t small_input_limit = 4096;
inline constexpr std::size_t narrow_output_limit = 1024;
inline constexpr std::size_t paired_output_limit = 4096;

template <class T>
struct ProjectionArguments {
  const T* input;
  const T* weight;
  T* output;
  std::size_t rows;
  std::size_t input_width;
  std::size_t output_width;
  std::size_t output_tiles;
  std::size_t tasks;
  const std::size_t* output_position;
};

// Device-only indexing: preparation proves allocation extents; the loops
// below guard tails and index shared/register arrays within their declared
// sizes. NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-pro-bounds-constant-array-index)
template <class T, unsigned Partials>
__global__ void projection(ProjectionArguments<T> args) {
  const auto [input, weight, output, rows, input_width, output_width,
              output_tiles, tasks, output_position] = args;
  const auto output_base =
      output_position ? *output_position * output_width : 0;
  __shared__ float x_tile[tile_width][tile_width];
  __shared__ float w_tile[tile_width][padded_tile_width];
  const auto x = threadIdx.x;
  const auto y = threadIdx.y;
  for (std::size_t task = blockIdx.x; task < tasks;) {
    const auto row = (task / output_tiles) * tile_width + y;
    const auto column = (task % output_tiles) * tile_width + x;
    const auto weight_row = (task % output_tiles) * tile_width + y;
    float partials[Partials]{};
    const auto reduction_tiles =
        input_width / tile_width + (input_width % tile_width != 0);
    for (std::size_t tile = 0; tile < reduction_tiles; ++tile) {
      const auto reduction = tile * tile_width + x;
      x_tile[y][x] = row < rows && reduction < input_width
                         ? detail::widen(input[row * input_width + reduction])
                         : 0.0F;
      w_tile[x][y] =
          weight_row < output_width && reduction < input_width
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
      output[output_base + row * output_width + column] =
          detail::narrow<T>(partials[0]);
    }
    if (tasks - task <= gridDim.x) {
      break;
    }
    task += gridDim.x;
  }
}

// Stage prompt operands once per block; weights remain coalesced global loads.
template <class T, unsigned Partials, unsigned RowTile>
__global__ void projection_shared(ProjectionArguments<T> args) {
  constexpr unsigned stage = 128;
  static_assert(stage % Partials == 0 && decode_threads % Partials == 0);
  constexpr unsigned outputs_per_block = decode_threads / Partials;
  __shared__ float input_tile[RowTile][stage];
  const auto lane = threadIdx.x % Partials;
  const auto group = threadIdx.x / Partials;
  const auto output_base =
      args.output_position ? *args.output_position * args.output_width : 0;
  for (std::size_t task = blockIdx.x; task < args.tasks;) {
    const auto first_row = (task / args.output_tiles) * RowTile;
    const auto column = (task % args.output_tiles) * outputs_per_block + group;
    float partials[RowTile]{};
    for (std::size_t base = 0; base < args.input_width; base += stage) {
#pragma unroll
      for (unsigned r = 0; r < RowTile; ++r) {
        input_tile[r][threadIdx.x] =
            first_row + r < args.rows
                ? detail::widen(args.input[(first_row + r) * args.input_width +
                                           base + threadIdx.x])
                : 0.0F;
      }
      __syncthreads();
      if (column < args.output_width) {
#pragma unroll
        for (unsigned k = 0; k < stage; k += Partials) {
          const auto weight = detail::widen(
              args.weight[column * args.input_width + base + k + lane]);
#pragma unroll
          for (unsigned r = 0; r < RowTile; ++r) {
            partials[r] =
                __fmaf_rn(input_tile[r][k + lane], weight, partials[r]);
          }
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (unsigned r = 0; r < RowTile; ++r) {
      float value = partials[r];
#pragma unroll
      for (unsigned offset = Partials / 2; offset != 0; offset /= 2) {
        value = __fadd_rn(
            value, __shfl_down_sync(full_warp_mask, value, offset, Partials));
      }
      if (lane == 0 && column < args.output_width &&
          first_row + r < args.rows) {
        args.output[output_base + (first_row + r) * args.output_width +
                    column] = detail::narrow<T>(value);
      }
    }
    if (args.tasks - task <= gridDim.x) {
      break;
    }
    task += gridDim.x;
  }
}

template <class T, unsigned Partials>
__global__ void projection_decode(ProjectionArguments<T> args) {
  constexpr unsigned outputs_per_block = decode_threads / Partials;
  const auto lane = threadIdx.x % Partials;
  const auto group = threadIdx.x / Partials;
  for (std::size_t tile = blockIdx.x; tile < args.output_tiles;
       tile += gridDim.x) {
    const auto column = tile * outputs_per_block + group;
    float partial = 0.0F;
    // Complete subwarps participate in every shuffle, including output tails.
    if (column < args.output_width) {
      for (std::size_t feature = lane; feature < args.input_width;
           feature += Partials) {
        partial = __fmaf_rn(
            detail::widen(args.input[feature]),
            detail::widen(args.weight[column * args.input_width + feature]),
            partial);
      }
      // The tiled path zero-pads to 16 features. Preserve its final FMA
      // even for a partial whose last feature lies in that padding.
      const auto padded_width = (args.input_width / tile_width +
                                 (args.input_width % tile_width != 0)) *
                                tile_width;
      if (padded_width != args.input_width &&
          lane >= args.input_width % Partials &&
          lane < (padded_width - 1) % Partials + 1) {
        partial = __fmaf_rn(0.0F, 0.0F, partial);
      }
    }
#pragma unroll
    for (unsigned offset = Partials / 2; offset != 0; offset /= 2) {
      partial = __fadd_rn(
          partial, __shfl_down_sync(full_warp_mask, partial, offset, Partials));
    }
    if (lane == 0 && column < args.output_width) {
      args.output[(args.output_position
                       ? *args.output_position * args.output_width
                       : 0) +
                  column] = detail::narrow<T>(partial);
    }
  }
}

template <unsigned Partials>
__global__ void projection_decode_pairs(
    ProjectionArguments<__nv_bfloat16> args) {
  constexpr unsigned lanes = Partials / 2;
  constexpr unsigned outputs_per_block = decode_threads / lanes;
  const auto lane = threadIdx.x % lanes;
  const auto group = threadIdx.x / lanes;
  // Dispatch proves 4-byte alignment and complete 32-feature groups. These
  // CUDA vector loads preserve the two BF16 objects' bits.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  const auto* input = reinterpret_cast<const __nv_bfloat162*>(args.input);
  const auto* weight = reinterpret_cast<const __nv_bfloat162*>(args.weight);
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  const auto width = static_cast<unsigned>(args.input_width / 2);
  for (std::size_t tile = blockIdx.x; tile < args.output_tiles;
       tile += gridDim.x) {
    const auto column = tile * outputs_per_block + group;
    float even = 0.0F;
    float odd = 0.0F;
    if (column < args.output_width) {
      for (std::size_t feature = lane; feature < width; feature += lanes) {
        const auto x = __bfloat1622float2(input[feature]);
        const auto w = __bfloat1622float2(weight[column * width + feature]);
        even = __fmaf_rn(x.x, w.x, even);
        odd = __fmaf_rn(x.y, w.y, odd);
      }
    }
#pragma unroll
    for (unsigned offset = lanes / 2; offset != 0; offset /= 2) {
      even = __fadd_rn(even,
                       __shfl_down_sync(full_warp_mask, even, offset, lanes));
      odd =
          __fadd_rn(odd, __shfl_down_sync(full_warp_mask, odd, offset, lanes));
    }
    if (lane == 0 && column < args.output_width) {
      args.output[(args.output_position
                       ? *args.output_position * args.output_width
                       : 0) +
                  column] = detail::narrow<__nv_bfloat16>(__fadd_rn(even, odd));
    }
  }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-pro-bounds-constant-array-index)

template <class T>
void submit(const void* input, const void* weight, void* output,
            std::size_t rows, std::size_t input_width, std::size_t output_width,
            cudaStream_t stream, const std::size_t* output_position) {
  if (rows == 1) {
    const bool narrow =
        input_width <= small_input_limit && output_width > narrow_output_limit;
    const auto partials = narrow ? tile_width : wide_partials;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      // Natural BF16 operand alignment is only two bytes; packed access is
      // optional and must also admit views starting between adjacent pairs.
      // Pointer-to-integer conversion only tests the packed-load precondition.
      // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
      const bool aligned =
          reinterpret_cast<std::uintptr_t>(input) % alignof(__nv_bfloat162) ==
              0 &&
          reinterpret_cast<std::uintptr_t>(weight) % alignof(__nv_bfloat162) ==
              0;
      // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
      if (aligned && input_width > small_input_limit &&
          input_width <= std::numeric_limits<unsigned>::max() &&
          input_width % wide_partials == 0 &&
          output_width <= paired_output_limit) {
        const auto tiles = detail::ceil_div(
            output_width, decode_threads / (wide_partials / 2));
        const ProjectionArguments<__nv_bfloat16> args{
            .input = static_cast<const __nv_bfloat16*>(input),
            .weight = static_cast<const __nv_bfloat16*>(weight),
            .output = static_cast<__nv_bfloat16*>(output),
            .rows = rows,
            .input_width = input_width,
            .output_width = output_width,
            .output_tiles = tiles,
            .tasks = tiles,
            .output_position = output_position};
        projection_decode_pairs<wide_partials>
            <<<detail::grid_blocks(tiles), decode_threads, 0, stream>>>(args);
        return;
      }
    }
    const auto tiles =
        detail::ceil_div(output_width, decode_threads / partials);
    const ProjectionArguments<T> args{.input = static_cast<const T*>(input),
                                      .weight = static_cast<const T*>(weight),
                                      .output = static_cast<T*>(output),
                                      .rows = rows,
                                      .input_width = input_width,
                                      .output_width = output_width,
                                      .output_tiles = tiles,
                                      .tasks = tiles,
                                      .output_position = output_position};
    if (narrow) {
      projection_decode<T, tile_width>
          <<<detail::grid_blocks(tiles), decode_threads, 0, stream>>>(args);
    } else {
      projection_decode<T, wide_partials>
          <<<detail::grid_blocks(tiles), decode_threads, 0, stream>>>(args);
    }
    return;
  }
  // The wider input stage must not add zero FMAs to ragged feature tails.
  constexpr unsigned stage = 128;
  constexpr unsigned row_tile = 8;
  if (input_width % stage == 0) {
    const bool narrow =
        input_width <= small_input_limit && output_width > narrow_output_limit;
    const auto partials = narrow ? tile_width : wide_partials;
    const auto output_tiles =
        detail::ceil_div(output_width, decode_threads / partials);
    const auto tasks = detail::ceil_div(rows, row_tile) * output_tiles;
    const ProjectionArguments<T> args{.input = static_cast<const T*>(input),
                                      .weight = static_cast<const T*>(weight),
                                      .output = static_cast<T*>(output),
                                      .rows = rows,
                                      .input_width = input_width,
                                      .output_width = output_width,
                                      .output_tiles = output_tiles,
                                      .tasks = tasks,
                                      .output_position = output_position};
    if (narrow) {
      projection_shared<T, tile_width, row_tile>
          <<<detail::grid_blocks(tasks), decode_threads, 0, stream>>>(args);
    } else {
      projection_shared<T, wide_partials, row_tile>
          <<<detail::grid_blocks(tasks), decode_threads, 0, stream>>>(args);
    }
    return;
  }
  const auto output_tiles = detail::ceil_div(output_width, tile_width);
  const auto tasks = detail::ceil_div(rows, tile_width) * output_tiles;
  if (input_width <= small_input_limit && output_width > narrow_output_limit) {
    projection<T, tile_width><<<detail::grid_blocks(tasks),
                                dim3(tile_width, tile_width), 0, stream>>>(
        ProjectionArguments<T>{.input = static_cast<const T*>(input),
                               .weight = static_cast<const T*>(weight),
                               .output = static_cast<T*>(output),
                               .rows = rows,
                               .input_width = input_width,
                               .output_width = output_width,
                               .output_tiles = output_tiles,
                               .tasks = tasks,
                               .output_position = output_position});
  } else {
    projection<T, wide_partials><<<detail::grid_blocks(tasks),
                                   dim3(tile_width, tile_width), 0, stream>>>(
        ProjectionArguments<T>{.input = static_cast<const T*>(input),
                               .weight = static_cast<const T*>(weight),
                               .output = static_cast<T*>(output),
                               .rows = rows,
                               .input_width = input_width,
                               .output_width = output_width,
                               .output_tiles = output_tiles,
                               .tasks = tasks,
                               .output_position = output_position});
  }
}
}  // namespace

void launch_projection(tensors::ElementType type, const void* input,
                       const void* weight, void* output, std::size_t rows,
                       std::size_t input_width, std::size_t output_width,
                       cudaStream_t stream,
                       const std::size_t* output_position) {
  if (detail::is_bfloat16(type)) {
    submit<__nv_bfloat16>(input, weight, output, rows, input_width,
                          output_width, stream, output_position);
  } else {
    submit<float>(input, weight, output, rows, input_width, output_width,
                  stream, output_position);
  }
}
}  // namespace dolittle::kernels::cuda
