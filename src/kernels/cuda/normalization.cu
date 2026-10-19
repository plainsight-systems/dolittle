/*
 * Axis X5: RMS normalization of residual vectors or individual Q/K heads.
 * Inputs: R vectors X[r,0..D), learned scales W[0..D), positive epsilon.
 * Output: Y has X's shape and BF16 or FP32 storage type.
 * Algorithm for each vector:
 * 1. Square each widened feature in FP32. Thirty-two lanes each own four
 *    sums. For D>128, lane t/component c visits 4*t+c+128*k; for
 *    D<=128 it visits t+32*c. Squares and adds round separately to FP32.
 *    Both layouts are fixed by vector width, independent of row count.
 * 2. Combine each lane's four sums left to right. Reduce lanes with shuffle
 *    offsets 1,2,4,8,16, adding the higher lane into the lower lane.
 * 3. Multiply the total by FP32(1/D); add epsilon and compute FP32 rsqrt.
 * 4. Multiply X by that inverse RMS and round to storage. Multiply this
 *    rounded value by W in FP32, then round again when writing Y.
 * The intermediate storage rounding precedes the learned scale.
 * CUDA implementation: one 32-thread warp processes a vector; each lane
 * loads neighboring groups for wide vectors and strided features for heads.
 * Lane zero broadcasts the sum. Grid-stride iteration changes scheduling,
 * never reduction order.
 * Cost: O(R*D) work, four FP32 partials per lane, no shared-memory scratch.
 * Optimization: fuse squaring/reduction; avoid a tensor of squared values.
 * Reference: PyTorch Reduce.cuh vectorized four-component reduction and
 * Transformers v4.57.3 Qwen3RMSNorm storage boundaries. Fixed warp geometry
 * preserves batch invariance instead of inheriting library launch selection.
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
inline constexpr unsigned warp_lanes = 32;
inline constexpr unsigned feature_group = 128;
inline constexpr unsigned full_warp_mask = 0xffffffffU;

template <class T>
struct RmsArguments {
  const T* input;
  const T* weight;
  T* output;
  std::size_t rows;
  std::size_t width;
  float epsilon;
};

// Device-only indexing: preparation proves allocation extents; the loops
// below guard tails and index shared/register arrays within their declared
// sizes. NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-pro-bounds-constant-array-index)
template <class T>
__global__ void rms_norm(RmsArguments<T> args) {
  const auto [input, weight, output, rows, width, epsilon] = args;
  for (std::size_t row = blockIdx.x; row < rows;) {
    float partials[4]{};
    if (width <= feature_group) {
#pragma unroll
      for (unsigned c = 0; c < 4; ++c) {
        const auto column = threadIdx.x + warp_lanes * c;
        if (column < width) {
          const auto value = detail::widen(input[row * width + column]);
          partials[c] = __fmul_rn(value, value);
        }
      }
    } else {
      for (std::size_t base = 4 * std::size_t{threadIdx.x}; base < width;
           base += feature_group) {
#pragma unroll
        for (unsigned c = 0; c < 4; ++c) {
          if (base + c < width) {
            const auto value = detail::widen(input[row * width + base + c]);
            partials[c] = __fadd_rn(partials[c], __fmul_rn(value, value));
          }
        }
      }
    }
    float sum = partials[0];
#pragma unroll
    for (unsigned c = 1; c < 4; ++c) {
      sum = __fadd_rn(sum, partials[c]);
    }
#pragma unroll
    for (unsigned offset = 1; offset < warp_lanes; offset *= 2) {
      sum = __fadd_rn(sum, __shfl_down_sync(full_warp_mask, sum, offset));
    }
    const auto total = __shfl_sync(full_warp_mask, sum, 0);
    const auto mean = __fmul_rn(total, 1.0F / static_cast<float>(width));
    const auto inverse = rsqrtf(__fadd_rn(mean, epsilon));
    for (std::size_t column = threadIdx.x; column < width;
         column += blockDim.x) {
      const auto normalized = detail::storage_round<T>(
          __fmul_rn(detail::widen(input[row * width + column]), inverse));
      output[row * width + column] = detail::narrow<T>(
          __fmul_rn(normalized, detail::widen(weight[column])));
    }
    if (rows - row <= gridDim.x) {
      break;
    }
    row += gridDim.x;
  }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-pro-bounds-constant-array-index)

template <class T>
void submit(const void* input, const void* weight, void* output,
            std::size_t rows, std::size_t width, float epsilon,
            cudaStream_t stream) {
  rms_norm<<<detail::grid_blocks(rows), warp_lanes, 0, stream>>>(
      RmsArguments<T>{.input = static_cast<const T*>(input),
                      .weight = static_cast<const T*>(weight),
                      .output = static_cast<T*>(output),
                      .rows = rows,
                      .width = width,
                      .epsilon = epsilon});
}
}  // namespace

void launch_rms_norm(tensors::ElementType type, const void* input,
                     const void* weight, void* output, std::size_t rows,
                     std::size_t width, float epsilon, cudaStream_t stream) {
  if (detail::is_bfloat16(type)) {
    submit<__nv_bfloat16>(input, weight, output, rows, width, epsilon, stream);
  } else {
    submit<float>(input, weight, output, rows, width, epsilon, stream);
  }
}
}  // namespace dolittle::kernels::cuda
