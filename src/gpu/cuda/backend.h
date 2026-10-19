/*
 * Axis X4, CUDA-private bridge: checked access for prepared X5 kernels.
 * Contract: native handles never cross the public device interface. Every
 * access verifies the same live Device instance and rejects a poisoned runtime
 * owner. Borrowed buffers/queues/device remain stable through completion. check
 * reports CUDA failure; drain_noexcept is conservative teardown only and
 * poisons on error. Algorithm: transformer overlay "Preparing the first
 * execution recipe" and "Storage lifetimes": validate resource identity,
 * activate its ordinal, expose stream/address/capabilities, submit, check
 * immediate status, explicitly wait later. Cost: constant identity checks; no
 * allocation or synchronization on access. Optimization: descriptors bind
 * existing storage, never create per-launch arenas. Corpus consultation,
 * design/implementation, 2026-10-18: cpp-guidelines I.1 (#ri-explicit):
 * explicit device/queue relationships. cpp-guidelines R.1 (#rr-raii): private
 * factories construct unique resource owners. cpp-perf-guidelines GPU.7: expose
 * ordered streams without an implicit wait. Source:
 * guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 */
#pragma once
#include <cuda_runtime_api.h>

#include "gpu/device.h"

namespace dolittle::gpu::detail {
struct BackendAccess {
  static DeviceBuffer allocate(Device&, AllocationRequest);
  static HostBuffer allocate_staging(Device&, AllocationRequest);
  static Queue create_queue(Device&);
  static cudaStream_t stream(Queue&, Device&);
  static void* pointer(DeviceBuffer&, Device&);
  static const void* pointer(const DeviceBuffer&, Device&);
  static const cudaDeviceProp& properties(Device&);
  static int ordinal(Device&);
  static void check(Device&, cudaError_t);
  static void check_launch(Device&);
  static void drain_noexcept(Device&) noexcept;
};
}  // namespace dolittle::gpu::detail
