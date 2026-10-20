/*
 * Axis R1/X5: prepared device categorical token selection.
 * Contract: device/queue outlive this stable owner; input is one immutable
 * BF16/FP32 vocabulary row on the same device. submit enqueues selection on the
 * owning queue without allocation/wait. Result is two uint32 words: token and
 * status (zero success). Read only after queue completion. Invalid settings or
 * extents fail before submission; NaN/+inf/all-masked logits report status.
 * Algorithm: encode descending finite FP32 keys (signed zeros equal), stable
 * radix-sort IDs, retain top-k, compute FP64 exp weights, scan fixed 256-entry
 * tiles in order, choose the inclusive top-p prefix, normalize prefixes, and
 * select its first CDF > u. FP64 tile totals combine in index order; run-to-run
 * deterministic, with rounding distinct from the CPU serial fold. GDSA.2
 * (guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md) was read
 * for this design and implementation on 2026-10-20. u=(draw>>11)*2^-53. Greedy
 * picks highest logit / lowest ID without a draw. Supplied-draw submit is
 * diagnostic; live state is device-owned. Transcendental implementations may
 * differ from host libm; tensor precision and selection law are unchanged. No
 * quantization. Cost: O(V) reserved keys/IDs/weights and block histograms;
 * bounded radix passes. Only 8 status/token bytes cross to the host. No host
 * logit readback. Overlay: transformer-execution.md, "Device selection and
 * allocation-free generation". Core I.7 (#ri-post): explicit
 * completion/selection contracts. GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md): keep selection resident. MEM.9
 * (guidelines/memory/MEM.9-embedded-allocate-at-init.md): reserve once. Both
 * corpora queried/read for header design and implementation, 2026-10-20.
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
#pragma once
#include <memory>

#include "gpu/device.h"
#include "inference/sampling/sampler.h"
#include "tensors/representations/representation.h"
namespace dolittle::inference::sampling {
struct DeviceDraw {
  Settings settings;
  std::uint64_t bits;
};
class DeviceCategorical {
 public:
  DeviceCategorical(gpu::Device&, gpu::Queue&, std::size_t vocabulary,
                    tensors::ElementType, std::size_t budget);
  ~DeviceCategorical();
  DeviceCategorical(const DeviceCategorical&) = delete;
  DeviceCategorical& operator=(const DeviceCategorical&) = delete;
  DeviceCategorical(DeviceCategorical&&) = delete;
  DeviceCategorical& operator=(DeviceCategorical&&) = delete;
  [[nodiscard]] static std::size_t required_bytes(std::size_t vocabulary);
  [[nodiscard]] gpu::ConstDeviceRegion submit(gpu::ConstDeviceRegion,
                                              DeviceDraw);

  // Seed once before submission; preserve mt19937_64's sequence on device.
  // Stop/error latches suppress subsequent draws. No host RNG per token.
  void start(Settings, std::uint64_t seed,
             std::span<const std::uint32_t> stop_ids);
  [[nodiscard]] gpu::ConstDeviceRegion submit_next(gpu::ConstDeviceRegion);
  [[nodiscard]] gpu::ConstDeviceRegion current_token() const;
  [[nodiscard]] gpu::ConstDeviceRegion result_storage() const;
  [[nodiscard]] gpu::ConstDeviceRegion terminal_state() const;

 private:
  [[nodiscard]] gpu::ConstDeviceRegion submit_impl(gpu::ConstDeviceRegion,
                                                   DeviceDraw, bool owned);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::inference::sampling
