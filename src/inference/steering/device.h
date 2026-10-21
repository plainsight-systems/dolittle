/*
 * Axis X5: device reconstruction difference and residual update.
 * Contract: borrowed device/queue outlive these nonmoving owners. Direction
 * construction retains two complete same-width FP32/BF16 reconstruction leases.
 * submit writes its independently owned FP32 output and status; finish drains
 * and checks scalar status, returning an immutable lease that survives
 * teardown. No numerical vector is read on the CPU. Intervention borrows that
 * lease, owns initialized status/count words, and submits within the sequence
 * graph. The executor drains before destruction. Invalid bounds fail before
 * enqueue; nonfinite/zero direction and nonfinite mutation report device
 * status. Algorithm: steering.md, Algorithm and numerical boundaries. Eight GPU
 * lanes sum squares of FP32(r1[i])-FP32(r0[i]) for i=lane+8*k with separate
 * multiply and add rounding. Lane zero combines 0..7 in order and computes FP32
 * sqrt. Divide differences by that norm into FP32 direction. For the selected
 * row, compute L2(h) with the same reduction, beta=alpha*norm, then separately
 * round beta*u[i] and h[i]+product before one storage cast. Count one
 * application. Optimization: O(D) work and storage prepared before submission;
 * position, reduction, finite checks and mutation stay GPU-owned. No hot
 * allocation. Header design: Core I.1 (#ri-explicit), R.1 (#rr-raii);
 * performance GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md), GPU.9
 * (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md).
 * Both corpora queried/read, 2026-10-20: explicit lifetime and device
 * residency.
 */
#pragma once

#include "inference/intervention.h"

namespace dolittle::inference {
class DeviceDirection {
 public:
  DeviceDirection(gpu::Device&, gpu::Queue&, DeviceCapture original,
                  DeviceCapture edited, tensors::ElementType,
                  std::size_t width);
  ~DeviceDirection();
  DeviceDirection(const DeviceDirection&) = delete;
  DeviceDirection& operator=(const DeviceDirection&) = delete;
  DeviceDirection(DeviceDirection&&) = delete;
  DeviceDirection& operator=(DeviceDirection&&) = delete;
  [[nodiscard]] static std::size_t required_bytes(std::size_t width);
  void submit();
  [[nodiscard]] DeviceCapture finish();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class DeviceIntervention {
 public:
  DeviceIntervention(gpu::Device&, gpu::Queue&, ResidualIntervention,
                     tensors::ElementType);
  ~DeviceIntervention();
  DeviceIntervention(const DeviceIntervention&) = delete;
  DeviceIntervention& operator=(const DeviceIntervention&) = delete;
  DeviceIntervention(DeviceIntervention&&) = delete;
  DeviceIntervention& operator=(DeviceIntervention&&) = delete;
  static constexpr std::size_t required_bytes = 2 * sizeof(std::uint32_t);
  void initialize();
  void submit(gpu::DeviceRegion residual, gpu::ConstDeviceRegion position,
              std::size_t rows);
  [[nodiscard]] gpu::ConstDeviceRegion status() const;
  [[nodiscard]] gpu::ConstDeviceRegion count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::inference
