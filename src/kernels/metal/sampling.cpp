/*
 * Metal implementation of the shared target-generation operations.
 *
 * Axis R1/X5: Metal implementation of inference/sampling/device.h. One stable
 * owner borrows Device/Queue and preallocates the complete sampler arena.
 * Contract: BF16/FP32 logits stay on device; result is {uint32 token,status}
 * plus a device terminal latch. Live RNG is MT19937-64 with the existing seed
 * and 53-bit draw mapping. Greedy does not draw. No CPU random generation,
 * vocabulary readback, per-token native allocation or continuation callback.
 *
 * Algorithm: construct all sort/scan/selection pipelines and immutable options
 * during preparation. required_bytes(V) includes ping-pong {key,id} arrays,
 * radix histograms/offsets, binary64 prefix/tile storage, RNG, stop mask and
 * result. start uploads only caller settings/stop IDs, then GPU-seeds RNG.
 * submit_next records the fixed sampler stages into the current body. The
 * following decode consumes current_token on GPU before another selection.
 * Diagnostic supplied-draw submit uses explicit caller bits and a completed
 * isolated operation; it is not used by live generation. sampling.metal
 * defines the sorting, probability, scan and stop arithmetic in full.
 *
 * Parameters are immutable and owned through drain; borrowed logits/result
 * regions are checked before recording. A failed status is terminal and cannot
 * reuse RNG state. Constructing/replacing the sampler requires an idle queue.
 * Cost O(V) arena plus O(312) RNG state; required_bytes uses checked arithmetic
 * and covers every encoded pass. No transient grows with generated length.
 * Optimization: reuse the complete sampler command body and arena; greedy
 * skips sort/exp/scan while preserving lowest-ID ties and validation.
 *
 * Header-design consultation, 2026-10-21: both local corpora searched and
 * cited rules read. Core I.7 (#ri-post) fixes numerical/completion results;
 * SL.con.3 (#rsl-bounds) requires checked extents before native indexing;
 * R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.1: all sampling and stopping remain GPU-owned.
 * guidelines/gpu/GPU.1-keep-data-on-device.md
 * GPU.6: record sort and selection once per prepared body.
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md
 * GPU.9: one admitted persistent arena, no hot allocations.
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md
 */

#include <array>
#include <bit>
#include <functional>
#include <limits>
#include <stdexcept>

#include "gpu/metal/backend.h"
#include "gpu/metal/commands.h"
#include "inference/sampling/device.h"
#include "kernels/metal/launch.h"
#include "tensors/tensor.h"
namespace dolittle::inference::sampling {
namespace {
using Access = gpu::detail::BackendAccess;
constexpr std::size_t tile_width = 256, simd_lanes = 32, mt_words = 312;
constexpr std::size_t candidate_bytes = 2 * sizeof(std::uint32_t);
constexpr std::size_t random_bytes = mt_words * sizeof(std::uint64_t) +
                                     2 * sizeof(std::uint32_t) +
                                     sizeof(std::uint64_t) + candidate_bytes;
constexpr std::size_t result_bytes = candidate_bytes,
                      scan_bytes =
                          sizeof(std::uint64_t) + 2 * sizeof(std::uint32_t);
constexpr std::size_t histogram_bytes =
    (tile_width / simd_lanes) * tile_width * sizeof(std::uint32_t);
constexpr std::size_t prefix_bytes = tile_width * sizeof(std::uint32_t);
constexpr std::size_t weight_tile_bytes = tile_width * sizeof(std::uint64_t);
constexpr std::size_t minima_bytes =
    (tile_width / simd_lanes) * sizeof(std::uint32_t);
constexpr std::size_t greedy_shared_bytes =
    tile_width * (sizeof(float) + sizeof(std::uint32_t));
constexpr std::size_t temperature_slot = 6, top_p_slot = 7, radix_bits = 8;
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
           random_bytes - result_bytes;
  }
  void dispatch(const char* name, kernels::metal::OperationArgs args,
                std::size_t groups, std::size_t threads, std::size_t shared = 0,
                const gpu::metal::BufferBinding* row = nullptr) {
    const auto binding = Access::binding(
        gpu::ConstDeviceRegion{&arena, 0, arena.size_bytes()}, device);
    args.address.at(1) = binding.address;
    args.dimensions.at(0) = vocabulary;
    args.type = type == tensors::ElementType::bfloat16 ? 1 : 0;
    const std::array bindings{binding, row ? *row : binding};
    auto pipeline = gpu::metal::pipeline(device, DOLITTLE_METAL_LIBRARY, name);
    if (threads > pipeline->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error("sampler threadgroup exceeds pipeline limit");
    }
    Access::dispatch(queue, device, *pipeline,
                     std::span(bindings).first(row ? 2 : 1),
                     {groups * threads, 1, 1}, {threads, 1, 1},
                     std::as_bytes(std::span(&args, 1)), shared);
  }
};
std::size_t DeviceCategorical::required_bytes(std::size_t n) {
  if (!n || n > std::numeric_limits<unsigned>::max() / simd_lanes) {
    throw std::invalid_argument("unsupported sampling vocabulary");
  }
  return n * (2 * candidate_bytes + sizeof(std::uint64_t)) +
         ((n + (tile_width - 1)) / tile_width) * tile_width * 4 + scan_bytes +
         result_bytes + random_bytes + n;
}
DeviceCategorical::DeviceCategorical(gpu::Device& d, gpu::Queue& q,
                                     std::size_t n, tensors::ElementType type,
                                     std::size_t budget) {
  const auto bytes = required_bytes(n);
  if (bytes > budget) throw std::length_error("device sampling exceeds budget");
  if (type != tensors::ElementType::float32 &&
      type != tensors::ElementType::bfloat16) {
    throw std::invalid_argument("unsupported sampling storage");
  }
  (void)Access::native_queue(q, d);
  impl_ = std::make_unique<Impl>(
      Impl{d, q, n, type, d.allocate({bytes, tile_width})});
  for (auto name :
       {"sampler_seed", "sampler_clear_stops", "sampler_set_stop",
        "sampler_begin", "sampler_commit", "sampler_keys", "sampler_histogram",
        "sampler_radix_prefix", "sampler_scatter", "sampler_weights",
        "sampler_scan_blocks", "sampler_cutoff", "sampler_select",
        "sampler_finish", "sampler_greedy", "sampler_greedy_finish"}) {
    (void)gpu::metal::pipeline(d, DOLITTLE_METAL_LIBRARY, name);
  }
}
DeviceCategorical::~DeviceCategorical() = default;
gpu::ConstDeviceRegion DeviceCategorical::submit(gpu::ConstDeviceRegion row,
                                                 DeviceDraw draw) {
  return submit_impl(row, draw, false);
}
void DeviceCategorical::start(Settings settings, std::uint64_t seed,
                              std::span<const std::uint32_t> stops) {
  auto& p = *impl_;
  validate_settings(settings);
  if (p.started || settings.top_k > p.vocabulary) {
    throw std::invalid_argument("invalid or repeated device sampler start");
  }
  for (auto id : stops) {
    if (id >= p.vocabulary) {
      throw std::invalid_argument("stop ID outside vocabulary");
    }
  }
  kernels::metal::OperationArgs args;
  args.capacity = seed;
  p.dispatch("sampler_clear_stops", args,
             (p.vocabulary + (tile_width - 1)) / tile_width, tile_width);
  for (auto id : stops) {
    args.first = id;
    p.dispatch("sampler_set_stop", args, 1, 1);
  }
  p.dispatch("sampler_seed", args, 1, 1);
  p.settings = settings;
  p.started = true;
}
gpu::ConstDeviceRegion DeviceCategorical::result_storage() const {
  return {&impl_->arena, impl_->result_offset(), result_bytes};
}
gpu::ConstDeviceRegion DeviceCategorical::terminal_state() const {
  return {&impl_->arena,
          impl_->result_offset() + result_bytes +
              mt_words * sizeof(std::uint64_t) + sizeof(std::uint32_t),
          4};
}
gpu::ConstDeviceRegion DeviceCategorical::current_token() const {
  if (!impl_->started || !impl_->has_result) {
    throw std::logic_error("device sampler has no preceding selection");
  }
  return {&impl_->arena, impl_->result_offset(), 4};
}
gpu::ConstDeviceRegion DeviceCategorical::submit_next(
    gpu::ConstDeviceRegion row) {
  if (!impl_->started) throw std::logic_error("device sampler not started");
  auto result = submit_impl(row, {impl_->settings, 0}, true);
  impl_->has_result = true;
  return result;
}
gpu::ConstDeviceRegion DeviceCategorical::submit_impl(
    gpu::ConstDeviceRegion row, DeviceDraw draw, bool owned) {
  auto& p = *impl_;
  validate_settings(draw.settings);
  if (row.bytes != p.vocabulary * tensors::element_bytes(p.type) ||
      row.offset % tensors::element_bytes(p.type) ||
      draw.settings.top_k > p.vocabulary) {
    throw std::invalid_argument(
        "invalid sampler input or mixed live/diagnostic calls");
  }
  const auto binding = Access::binding(row, p.device);
  kernels::metal::OperationArgs args;
  args.address.at(0) = binding.address;
  args.address.at(temperature_slot) =
      std::bit_cast<std::uint64_t>(draw.settings.temperature);
  args.address.at(top_p_slot) =
      std::bit_cast<std::uint64_t>(draw.settings.top_p);
  args.first = draw.bits;
  args.flags = (owned ? 1U : 0U) |
               (draw.settings.mode == SelectionMode::greedy ? 2U : 0U);
  args.dimensions.at(1) =
      draw.settings.top_k ? draw.settings.top_k : p.vocabulary;
  const auto blocks = (p.vocabulary + (tile_width - 1)) / tile_width;
  const auto kept = (args.dimensions.at(1) + (tile_width - 1)) / tile_width;
  const auto launch = [&](const char* name, std::size_t groups = 1,
                          std::size_t threads = 1, std::size_t shared = 0) {
    p.dispatch(name, args, groups, threads, shared, &binding);
  };
  launch("sampler_begin");
  if (draw.settings.mode == SelectionMode::greedy) {
    launch("sampler_greedy", blocks, tile_width, greedy_shared_bytes);
    launch("sampler_greedy_finish");
  } else {
    launch("sampler_keys", blocks, tile_width);
    for (unsigned shift = 0; shift < simd_lanes; shift += radix_bits) {
      args.dimensions.at(2) = shift;
      launch("sampler_histogram", blocks, tile_width, histogram_bytes);
      launch("sampler_radix_prefix", 1, tile_width, prefix_bytes);
      launch("sampler_scatter", blocks, tile_width, histogram_bytes);
    }
    launch("sampler_weights", kept, tile_width, weight_tile_bytes);
    launch("sampler_scan_blocks");
    launch("sampler_cutoff", kept, tile_width, minima_bytes);
    launch("sampler_select", kept, tile_width, minima_bytes);
    launch("sampler_finish");
  }
  if (owned) launch("sampler_commit");
  return result_storage();
}
}  // namespace dolittle::inference::sampling
