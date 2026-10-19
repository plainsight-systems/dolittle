/*
 * Axis X5: unscaled, split-half rotary position embedding (RoPE).
 * Role: encode position in each normalized Q or K head before attention.
 *
 * Inputs: X[N,H,d] contains N positions, H heads and an even head width d;
 * first_position is the absolute position of X[0]; theta is positive finite.
 * Output: Y has X's shape and BF16 or FP32 storage type. The model invokes
 * this operation separately for Q and K; V does not pass through it.
 *
 * Algorithm: for each position p, head h and feature i in [0,d/2):
 * 1. Set absolute position t = first_position + p. Compute exponent 2*i/d
 *    in FP32, then frequency = 1 / pow(theta, exponent) in double and
 *    convert frequency to FP32. Compute angle = FP32(t) * frequency.
 * 2. Compute c = cos(angle) and s = sin(angle) in FP32, rounding each to
 *    the storage type before use.
 * 3. Read the paired features a = X[p,h,i] and b = X[p,h,i+d/2]. Compute
 *    a*c, b*s, b*c and a*s in FP32; round each product to the storage type.
 * 4. Write Y[p,h,i] = a*c - b*s and Y[p,h,i+d/2] = b*c + a*s, using
 *    the rounded products from step 3 and rounding each result to storage.
 * Each pair is independent. The separate product and sum rounding steps
 * must not be replaced by a fused multiply-add.
 *
 * CUDA implementation: one thread handles a feature pair at a time; its
 * strided loop covers further pairs. No shared memory or barriers are needed.
 * Cost: O(N*H*d) work, with no stored trigonometric table.
 * Optimization: compute each angle once for both outputs of its pair;
 * adjacent threads read adjacent features within each half of a head.
 *
 * Overlay: docs/architecture/transformer-execution.md,
 * "Attention and positional semantics".
 * Reference: transformers v4.57.3, Qwen3RotaryEmbedding.forward and
 * apply_rotary_pos_emb:
 * https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3/modeling_qwen3.py.
 *
 * Contract: preparation validates shapes, byte/launch extents and device
 * limits. Inputs and the separate output buffer are borrowed until queue
 * completion; output must not overlap an input. Launches allocate no buffers
 * and perform no host readback or host wait. The caller checks immediate launch
 * status; asynchronous device errors surface at the explicit queue wait. See
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
template <class T>
struct RotaryArguments {
  const T* input;
  T* output;
  std::size_t pairs;
  std::size_t heads;
  std::size_t width;
  std::size_t first_position;
  double theta;
};

// Device-only indexing: preparation proves allocation extents; the loops
// below guard tails and index shared/register arrays within their declared
// sizes. NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
template <class T>
__global__ void rotary(RotaryArguments<T> args) {
  const auto [input, output, pairs, heads, width, first_position, theta] = args;
  const auto half = width / 2;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < pairs;) {
    const auto feature = index % half;
    const auto row = index / half;
    const auto position = first_position + row / heads;
    const auto exponent =
        static_cast<float>(2 * feature) / static_cast<float>(width);
    // Preserve finite double theta until the FP32 inverse-frequency boundary.
    const auto inverse_frequency =
        static_cast<float>(1.0 / pow(theta, static_cast<double>(exponent)));
    const auto angle =
        __fmul_rn(static_cast<float>(position), inverse_frequency);
    const auto cosine = detail::storage_round<T>(cosf(angle));
    const auto sine = detail::storage_round<T>(sinf(angle));
    const auto first = detail::widen(input[row * width + feature]);
    const auto second = detail::widen(input[row * width + feature + half]);
    const auto first_cosine =
        detail::storage_round<T>(__fmul_rn(first, cosine));
    const auto second_sine = detail::storage_round<T>(__fmul_rn(second, sine));
    const auto second_cosine =
        detail::storage_round<T>(__fmul_rn(second, cosine));
    const auto first_sine = detail::storage_round<T>(__fmul_rn(first, sine));
    output[row * width + feature] =
        detail::narrow<T>(__fsub_rn(first_cosine, second_sine));
    output[row * width + feature + half] =
        detail::narrow<T>(__fadd_rn(second_cosine, first_sine));
    if (pairs - index <= stride) {
      break;
    }
    index += stride;
  }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

template <class T>
void submit(const void* input, void* output, std::size_t positions,
            std::size_t heads, std::size_t width, std::size_t first_position,
            double theta, cudaStream_t stream) {
  const auto pairs = positions * heads * (width / 2);
  rotary<<<detail::grid_blocks(detail::ceil_div(pairs, detail::threads)),
           detail::threads, 0, stream>>>(
      RotaryArguments<T>{.input = static_cast<const T*>(input),
                         .output = static_cast<T*>(output),
                         .pairs = pairs,
                         .heads = heads,
                         .width = width,
                         .first_position = first_position,
                         .theta = theta});
}
}  // namespace

void launch_rotary(tensors::ElementType type, const void* input, void* output,
                   std::size_t positions, std::size_t heads, std::size_t width,
                   std::size_t first_position, double theta,
                   cudaStream_t stream) {
  if (detail::is_bfloat16(type)) {
    submit<__nv_bfloat16>(input, output, positions, heads, width,
                          first_position, theta, stream);
  } else {
    submit<float>(input, output, positions, heads, width, first_position, theta,
                  stream);
  }
}
}  // namespace dolittle::kernels::cuda
