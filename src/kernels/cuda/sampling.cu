/*
 * Axis R1/X5: CUDA implementation of device.h's prepared selector.
 * Contract: one queue, admitted scratch, no allocation in submit; completion
 * makes {token,status} visible. Input is BF16 or FP32; arithmetic for sampling
 * weights and the ordered CDF is FP64, independently of tensor precision.
 * Algorithm: four stable byte-radix passes sort descending float keys. Each
 * warp groups equal digits with match_any; warp histograms and block prefixes
 * give deterministic scatter ranks (never atomic arrival order). Parallel exp
 * computes weights; fixed 256-entry tiles scan in index order, then tile
 * totals scan in index order. Inclusive prefix/total locates top-p and
 * prefix/retained locates the draw. Integer minima select the first crossing.
 * FP64 grouping is fixed, run-to-run deterministic and independent of batch
 * or occupancy; it may differ from the CPU's serial fold in roundoff. The
 * probability law is unchanged; no bitwise cross-backend trajectory promise.
 * Greedy uses parallel maxima with lowest-ID ties. GDSA.2
 * (guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md)
 * queried/read for this scan design and implementation, 2026-10-20.
 * Cost: O(V) storage/work per pass, 8 bytes returned, no host vocabulary copy.
 * Core I.7 (#ri-post), GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md),
 * MEM.9 (guidelines/memory/MEM.9-embedded-allocate-at-init.md), GDSA.5
 * (guidelines/gpu-dsa/GDSA.5-aggregate-atomics-before-global-memory.md), both
 * corpora queried and full rules read for design/implementation, 2026-10-20.
 * GDSA.1 library-first advice is superseded by this project's explicit direct
 * computation requirement; stability and scratch admission remain mandatory.
 * Device-loop consultation, design/implementation, 2026-10-20:
 * Core I.7/CP.3: result_storage and terminal_state expose borrowed regions for
 * ordered graph readers, never host dereferencing. They may be bound before
 * the initial draw; seed/draw must execute before the graph reads them.
 * Performance GPU.1/GPU.6: device selection/latches control the conditional
 * graph without host readback. Both local corpora queried and full rules read;
 * sources guidelines/gpu/GPU.1-keep-data-on-device.md and
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md.
 * Live random state: start validates settings/stop IDs and seeds 312 uint64
 * MT19937-64 words once. A single device lane twists in-place every 312 draws,
 * tempers the next word and applies (word >> 11) * 2^-53. Selection consumes
 * that device value. A following kernel latches errors/stop IDs and preserves
 * the terminal result; later queued selections cannot advance that stream.
 * A failed distribution poisons the run; its RNG cannot be reused or resumed.
 * Diagnostic submit still accepts explicit bits for isolated comparisons.
 * Cost beyond sort storage: fixed RNG state and V stop-mask bytes; no per-step
 * allocations or uploads. Same-stream ordering protects state and selected ID.
 * current_token borrows the last result for a device-to-device embedding copy
 * before the next selection overwrites it. Do not mix diagnostic and live
 * calls.
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <limits>
#include <stdexcept>

#include "gpu/cuda/backend.h"
#include "inference/sampling/device.h"
#include "tensors/tensor.h"

namespace dolittle::inference::sampling {
namespace {
constexpr unsigned width = 256;
constexpr unsigned warp_width = 32;
constexpr unsigned warps = width / warp_width;
constexpr unsigned digit_bits = 8;
constexpr unsigned float_bits = 32;
constexpr unsigned sign_bit = 0x80000000U;
constexpr unsigned full_mask = 0xffffffffU;
constexpr unsigned random_shift = 11;
constexpr double random_scale = 0x1.0p-53;
struct WeightOutput {
  double* prefix;
  double* block_totals;
};
struct WeightOptions {
  unsigned n;
  double temperature;
};
struct ChoiceOptions {
  unsigned n;
  double top_p, draw;
  const double* device_draw;
};
struct ScanState {
  double total;
  unsigned cutoff, selected;
};
struct Candidate {
  unsigned key, id;
};
struct Result {
  unsigned token, status;
};
constexpr unsigned mt_size = 312;
constexpr unsigned mt_middle = 156;
struct RandomState {
  std::uint64_t words[mt_size];  // NOLINT(cppcoreguidelines-avoid-c-arrays)
  unsigned index, terminal;
  double draw;
  Result previous;
};
// Native device indexing: admitted extents, explicit tails and shared-array
// bounds below prove addresses; standard host containers cannot be used here.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index)
// MT19937-64 parameters are the standardized engine, not tunable constants.
// One lane owns the state; queue order serializes draws. This preserves the
// original seed sequence, including the in-place wrap during every twist.
// NOLINTBEGIN(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
__global__ void seed_random(RandomState* state, std::uint64_t seed) {
  state->words[0] = seed;
  for (unsigned i = 1; i < mt_size; ++i) {
    const auto previous = state->words[i - 1];
    state->words[i] =
        6364136223846793005ULL * (previous ^ (previous >> 62)) + i;
  }
  state->index = mt_size;
  state->terminal = 0;
  state->previous = {0, 0};
  state->draw = 0;
}
__global__ void next_random(RandomState* state, bool greedy) {
  if (state->terminal || greedy) {
    return;
  }
  if (state->index == mt_size) {
    for (unsigned i = 0; i < mt_size; ++i) {
      const auto x = (state->words[i] & 0xffffffff80000000ULL) |
                     (state->words[(i + 1) % mt_size] & 0x7fffffffULL);
      state->words[i] = state->words[(i + mt_middle) % mt_size] ^ (x >> 1) ^
                        ((x & 1) ? 0xb5026f5aa96619e9ULL : 0);
    }
    state->index = 0;
  }
  auto x = state->words[state->index++];
  x ^= (x >> 29) & 0x5555555555555555ULL;
  x ^= (x << 17) & 0x71d67fffeda60000ULL;
  x ^= (x << 37) & 0xfff7eee000000000ULL;
  x ^= x >> 43;
  state->draw = static_cast<double>(x >> random_shift) * random_scale;
}
// NOLINTEND(readability-magic-numbers,cppcoreguidelines-avoid-magic-numbers)
__global__ void commit_selection(RandomState* state, Result* result,
                                 const unsigned char* stops) {
  if (state->terminal) {
    *result = state->previous;
    return;
  }
  state->previous = *result;
  state->terminal = result->status || stops[result->token];
}
__device__ float load(const void* row, unsigned i, bool bf16) {
  return bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(row)[i])
              : static_cast<const float*>(row)[i];
}
__global__ void initialize(const void* row, bool bf16, unsigned n,
                           Candidate* keys, Result* result) {
  const unsigned i = blockIdx.x * width + threadIdx.x;
  if (i < n) {
    float v = load(row, i, bf16);
    if (isnan(v) || v == CUDART_INF_F) {
      atomicExch(&result->status, 1U);
    }
    if (v == 0.0F) {
      v = 0.0F;
    }
    const unsigned bits = __float_as_uint(v);
    keys[i] = {~((bits & sign_bit) ? ~bits : bits ^ sign_bit), i};
  }
}
__global__ void histogram(const Candidate* input, unsigned n, unsigned shift,
                          unsigned* counts) {
  __shared__ unsigned bins[warps][width];
  const unsigned t = threadIdx.x, lane = t % warp_width, warp = t / warp_width;
  for (unsigned w = 0; w < warps; ++w) {
    bins[w][t] = 0;
  }
  __syncthreads();
  const unsigned i = blockIdx.x * width + t;
  const unsigned digit = i < n ? (input[i].key >> shift) & (width - 1) : width;
  const unsigned peers = __match_any_sync(full_mask, digit);
  if (digit < width &&
      lane == static_cast<unsigned>(__ffs(static_cast<int>(peers)) - 1)) {
    bins[warp][digit] = static_cast<unsigned>(__popc(peers));
  }
  __syncthreads();
  unsigned total = 0;
  for (unsigned w = 0; w < warps; ++w) {
    total += bins[w][t];
  }
  counts[blockIdx.x * width + t] = total;
}
__global__ void prefix(unsigned* counts, unsigned blocks) {
  __shared__ unsigned totals[width];
  const unsigned digit = threadIdx.x;
  unsigned total = 0;
  for (unsigned b = 0; b < blocks; ++b) {
    const unsigned value = counts[b * width + digit];
    counts[b * width + digit] = total;
    total += value;
  }
  totals[digit] = total;
  __syncthreads();
  unsigned base = 0;
  for (unsigned d = 0; d < digit; ++d) {
    base += totals[d];
  }
  for (unsigned b = 0; b < blocks; ++b) {
    counts[b * width + digit] += base;
  }
}
__global__ void scatter(const Candidate* input, Candidate* output, unsigned n,
                        unsigned shift, const unsigned* offsets) {
  __shared__ unsigned bins[warps][width];
  const unsigned t = threadIdx.x, lane = t % warp_width, warp = t / warp_width;
  for (unsigned w = 0; w < warps; ++w) {
    bins[w][t] = 0;
  }
  __syncthreads();
  const unsigned i = blockIdx.x * width + t;
  const unsigned digit = i < n ? (input[i].key >> shift) & (width - 1) : width;
  const unsigned peers = __match_any_sync(full_mask, digit);
  if (digit < width &&
      lane == static_cast<unsigned>(__ffs(static_cast<int>(peers)) - 1)) {
    bins[warp][digit] = static_cast<unsigned>(__popc(peers));
  }
  __syncthreads();
  if (i < n) {
    unsigned rank = static_cast<unsigned>(__popc(peers & ((1U << lane) - 1U)));
    for (unsigned w = 0; w < warp; ++w) {
      rank += bins[w][digit];
    }
    output[offsets[blockIdx.x * width + digit] + rank] = input[i];
  }
}
__global__ void weights(const void* row, bool bf16, const Candidate* sorted,
                        WeightOutput output, WeightOptions options) {
  const auto [out, block_totals] = output;
  const auto [n, temperature] = options;
  const unsigned t = threadIdx.x, i = blockIdx.x * width + t;
  __shared__ double tile[width];
  double weight = 0.0;
  if (i < n) {
    const double best = static_cast<double>(load(row, sorted[0].id, bf16));
    const double v = static_cast<double>(load(row, sorted[i].id, bf16));
    if (isfinite(v)) {
      weight = exp((v - best) / temperature);
    }
  }
  tile[t] = weight;
  __syncthreads();
  if (t == 0) {
    double sum = 0.0;
    for (unsigned j = 0; j < width; ++j) {
      sum += tile[j];
      tile[j] = sum;
    }
    block_totals[blockIdx.x] = sum;
  }
  __syncthreads();
  if (i < n) {
    out[i] = tile[t];
  }
}
__global__ void scan_blocks(double* block_totals, unsigned n, ScanState* state,
                            Result* result) {
  const unsigned blocks = (n + width - 1) / width;
  double total = 0.0;
  for (unsigned b = 0; b < blocks; ++b) {
    const double value = block_totals[b];
    block_totals[b] = total;
    total += value;
  }
  state->total = total;
  state->cutoff = n - 1;
  state->selected = n - 1;
  if (!(total > 0.0)) {
    result->status = 2;
  }
}
// Integer minima have an exact associative result. Only one lane per block
// publishes its minimum; no arrival order affects ties or floating arithmetic.
__device__ unsigned block_min(unsigned value) {
  __shared__ unsigned minima[warps];
  const unsigned lane = threadIdx.x % warp_width,
                 warp = threadIdx.x / warp_width;
  for (unsigned stride = warp_width / 2; stride; stride >>= 1) {
    value = min(value, __shfl_down_sync(full_mask, value, stride));
  }
  if (lane == 0) {
    minima[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < warps ? minima[lane] : full_mask;
    for (unsigned stride = warp_width / 2; stride; stride >>= 1) {
      value = min(value, __shfl_down_sync(full_mask, value, stride));
    }
  }
  return value;
}
__global__ void cutoff(const double* cumulative, const double* offsets,
                       ChoiceOptions options, ScanState* state) {
  const unsigned i = blockIdx.x * width + threadIdx.x;
  unsigned candidate = full_mask;
  if (i < options.n &&
      (offsets[blockIdx.x] + cumulative[i]) / state->total >= options.top_p) {
    candidate = i;
  }
  const unsigned first = block_min(candidate);
  if (threadIdx.x == 0) {
    atomicMin(&state->cutoff, first);
  }
}
__global__ void select_prefix(const double* cumulative, const double* offsets,
                              ChoiceOptions options, ScanState* state) {
  const unsigned i = blockIdx.x * width + threadIdx.x;
  const unsigned last = state->cutoff;
  const double retained = offsets[last / width] + cumulative[last];
  unsigned candidate = full_mask;
  if (i <= last && (options.device_draw ? *options.device_draw : options.draw) <
                       (offsets[blockIdx.x] + cumulative[i]) / retained) {
    candidate = i;
  }
  const unsigned first = block_min(candidate);
  if (threadIdx.x == 0) {
    atomicMin(&state->selected, first);
  }
}
__global__ void finish_selection(const Candidate* sorted,
                                 const ScanState* state, Result* result) {
  if (!result->status) {
    result->token = sorted[min(state->selected, state->cutoff)].id;
  }
}
__global__ void greedy_blocks(const void* row, bool bf16, unsigned n,
                              Candidate* best, Result* result) {
  __shared__ float values[width];
  __shared__ unsigned ids[width];
  const unsigned t = threadIdx.x, i = blockIdx.x * width + t;
  const float v = i < n ? load(row, i, bf16) : -CUDART_INF_F;
  if (isnan(v) || v == CUDART_INF_F) {
    atomicExch(&result->status, 1U);
  }
  values[t] = v;
  ids[t] = i;
  __syncthreads();
  for (unsigned stride = width / 2; stride; stride >>= 1) {
    if (t < stride &&
        (values[t + stride] > values[t] ||
         (values[t + stride] == values[t] && ids[t + stride] < ids[t]))) {
      values[t] = values[t + stride];
      ids[t] = ids[t + stride];
    }
    __syncthreads();
  }
  if (t == 0) {
    best[blockIdx.x] = {__float_as_uint(values[0]), ids[0]};
  }
}
__global__ void greedy_finish(const Candidate* best, unsigned blocks,
                              Result* result) {
  if (result->status) {
    return;
  }
  float value = -CUDART_INF_F;
  unsigned id = full_mask;
  for (unsigned i = 0; i < blocks; ++i) {
    const float v = __uint_as_float(best[i].key);
    if (v > value || (v == value && best[i].id < id)) {
      value = v;
      id = best[i].id;
    }
  }
  if (!isfinite(value)) {
    result->status = 2;
  } else {
    result->token = id;
  }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index)
}  // namespace
struct DeviceCategorical::Impl {
  std::reference_wrapper<gpu::Device> device;
  std::reference_wrapper<gpu::Queue> queue;
  std::size_t vocabulary;
  tensors::ElementType type;
  gpu::DeviceBuffer arena;
  Settings settings{};
  bool started = false, has_result = false;
  std::size_t result_offset() const {
    return DeviceCategorical::required_bytes(vocabulary) - vocabulary -
           sizeof(RandomState) - sizeof(Result);
  }
  RandomState* random_state() {
    // Typed CUDA boundary, arena partition is aligned to eight bytes.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return reinterpret_cast<RandomState*>(
        static_cast<std::byte*>(
            gpu::detail::BackendAccess::pointer(arena, device)) +
        result_offset() + sizeof(Result));
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }
  unsigned char* stop_mask() {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return reinterpret_cast<unsigned char*>(random_state() + 1);
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }
};
std::size_t DeviceCategorical::required_bytes(std::size_t n) {
  if (!n || n > std::numeric_limits<unsigned>::max() / float_bits) {
    throw std::invalid_argument("unsupported sampling vocabulary");
  }
  return n * (2 * sizeof(Candidate) + sizeof(double)) +
         ((n + width - 1) / width) * width * sizeof(unsigned) +
         sizeof(ScanState) + sizeof(Result) + sizeof(RandomState) + n;
}
DeviceCategorical::DeviceCategorical(gpu::Device& device, gpu::Queue& queue,
                                     std::size_t n, tensors::ElementType type,
                                     std::size_t budget) {
  const auto bytes = required_bytes(n);
  if (bytes > budget) {
    throw std::length_error("device sampling exceeds budget");
  }
  if (type != tensors::ElementType::float32 &&
      type != tensors::ElementType::bfloat16) {
    throw std::invalid_argument("unsupported sampling storage");
  }
  impl_ = std::make_unique<Impl>(
      Impl{device, queue, n, type, device.allocate({bytes, width})});
}
DeviceCategorical::~DeviceCategorical() = default;
gpu::ConstDeviceRegion DeviceCategorical::submit(gpu::ConstDeviceRegion row,
                                                 DeviceDraw draw) {
  return submit_impl(row, draw, false);
}
void DeviceCategorical::start(Settings settings, std::uint64_t seed,
                              std::span<const std::uint32_t> stop_ids) {
  auto& p = *impl_;
  validate_settings(settings);
  if (p.started || settings.top_k > p.vocabulary) {
    throw std::invalid_argument("invalid or repeated device sampler start");
  }
  for (const auto id : stop_ids) {
    if (id >= p.vocabulary) {
      throw std::invalid_argument("stop ID outside vocabulary");
    }
  }
  using Access = gpu::detail::BackendAccess;
  const auto stream = Access::stream(p.queue, p.device);
  Access::check(p.device,
                cudaMemsetAsync(p.stop_mask(), 0, p.vocabulary, stream));
  for (const auto id : stop_ids) {
    // Validated vocabulary index into the device stop-mask partition.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    Access::check(p.device, cudaMemsetAsync(p.stop_mask() + id, 1, 1, stream));
  }
  seed_random<<<1, 1, 0, stream>>>(p.random_state(), seed);
  Access::check_launch(p.device);
  p.settings = settings;
  p.started = true;
}
gpu::ConstDeviceRegion DeviceCategorical::result_storage() const {
  const auto& p = *impl_;
  return {&p.arena, p.result_offset(), sizeof(Result)};
}
gpu::ConstDeviceRegion DeviceCategorical::terminal_state() const {
  const auto& p = *impl_;
  return {&p.arena,
          p.result_offset() + sizeof(Result) + offsetof(RandomState, terminal),
          sizeof(unsigned)};
}
gpu::ConstDeviceRegion DeviceCategorical::current_token() const {
  const auto& p = *impl_;
  if (!p.started || !p.has_result) {
    throw std::logic_error("device sampler has no preceding selection");
  }
  return {&p.arena, p.result_offset(), sizeof(std::uint32_t)};
}
gpu::ConstDeviceRegion DeviceCategorical::submit_next(
    gpu::ConstDeviceRegion row) {
  if (!impl_->started) {
    throw std::logic_error("device sampler not started");
  }
  const auto result = submit_impl(row, {impl_->settings, 0}, true);
  impl_->has_result = true;
  return result;
}
gpu::ConstDeviceRegion DeviceCategorical::submit_impl(
    gpu::ConstDeviceRegion row, DeviceDraw draw, bool owned) {
  auto& p = *impl_;
  validate_settings(draw.settings);
  if (!row.owner ||
      row.bytes != p.vocabulary * tensors::element_bytes(p.type) ||
      row.offset > row.owner->size_bytes() ||
      row.bytes > row.owner->size_bytes() - row.offset ||
      row.offset % tensors::element_bytes(p.type) ||
      draw.settings.top_k > p.vocabulary) {
    throw std::invalid_argument("invalid device sampling row");
  }
  using Access = gpu::detail::BackendAccess;
  const auto stream = Access::stream(p.queue, p.device);
  // CUDA address boundary: the admitted arena is aligned for each typed
  // partition; no device pointer is dereferenced on the host.
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-type-reinterpret-cast)
  const auto* input =
      static_cast<const std::byte*>(Access::pointer(*row.owner, p.device)) +
      row.offset;
  auto* first = static_cast<Candidate*>(Access::pointer(p.arena, p.device));
  auto* second = first + p.vocabulary;
  auto* probabilities = reinterpret_cast<double*>(second + p.vocabulary);
  auto* counts = reinterpret_cast<unsigned*>(probabilities + p.vocabulary);
  const auto n = static_cast<unsigned>(p.vocabulary),
             blocks = (n + width - 1) / width;
  auto* state = reinterpret_cast<ScanState*>(
      counts + static_cast<std::size_t>(blocks) * width);
  auto* result = reinterpret_cast<Result*>(state + 1);
  auto* block_totals = reinterpret_cast<double*>(counts);
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-type-reinterpret-cast)
  Access::check(p.device, cudaMemsetAsync(result, 0, sizeof(Result), stream));
  if (owned) {
    next_random<<<1, 1, 0, stream>>>(
        p.random_state(), draw.settings.mode == SelectionMode::greedy);
  }
  const bool bf16 = p.type == tensors::ElementType::bfloat16;
  if (draw.settings.mode == SelectionMode::greedy) {
    greedy_blocks<<<blocks, width, 0, stream>>>(input, bf16, n, first, result);
    greedy_finish<<<1, 1, 0, stream>>>(first, blocks, result);
  } else {
    initialize<<<blocks, width, 0, stream>>>(input, bf16, n, first, result);
    for (unsigned shift = 0; shift < float_bits; shift += digit_bits) {
      histogram<<<blocks, width, 0, stream>>>(first, n, shift, counts);
      prefix<<<1, width, 0, stream>>>(counts, blocks);
      scatter<<<blocks, width, 0, stream>>>(first, second, n, shift, counts);
      auto* temporary = first;
      first = second;
      second = temporary;
    }
    const auto keep =
        draw.settings.top_k ? static_cast<unsigned>(draw.settings.top_k) : n;
    const auto kept_blocks = (keep + width - 1) / width;
    const ChoiceOptions choices{
        keep, draw.settings.top_p,
        static_cast<double>(draw.bits >> random_shift) * random_scale,
        owned ? &p.random_state()->draw : nullptr};
    weights<<<kept_blocks, width, 0, stream>>>(
        input, bf16, first, {probabilities, block_totals},
        {keep, draw.settings.temperature});
    scan_blocks<<<1, 1, 0, stream>>>(block_totals, keep, state, result);
    cutoff<<<kept_blocks, width, 0, stream>>>(probabilities, block_totals,
                                              choices, state);
    select_prefix<<<kept_blocks, width, 0, stream>>>(
        probabilities, block_totals, choices, state);
    finish_selection<<<1, 1, 0, stream>>>(first, state, result);
  }
  if (owned) {
    commit_selection<<<1, 1, 0, stream>>>(p.random_state(), result,
                                          p.stop_mask());
  }
  Access::check_launch(p.device);
  return {&p.arena, p.result_offset(), sizeof(Result)};
}
}  // namespace dolittle::inference::sampling
