/*
 * Implemented foundation, 2026-10-21; header design approved before
 * implementation. Axis X4: Metal-private access for prepared operations and
 * sequences. Contract: every resource must belong to the same live, unpoisoned
 * Device. A binding borrows its buffer and mapped logical region; owners remain
 * stable through completion. GPU addresses are never exposed to application
 * code. Empty optional regions are handled by callers, not bound as valid
 * operands. Bounds/device mismatch -> invalid_argument; unsupported binding
 * extent -> length_error; native failure -> runtime_error; no hidden wait on
 * access.
 *
 * Algorithm: metal-port.md, Change axes and files. Resolve a checked logical
 * DeviceRegion to its native buffer, byte offset, address and extent. Verify
 * one physical view contains the whole operand. Preparation registers all
 * referenced buffers and pipelines with residency owners before command
 * encoding; GPU indirect references need residency even when the CPU does not
 * bind them directly. Submission uses the prepared native commands. Only
 * explicit completion or teardown waits for their users to finish.
 *
 * Cost: fixed identity/bounds checks and view lookup at preparation, with no
 * tensor allocation or copying. Optimization: cache resolved bindings in
 * prepared operations; execute the same ICB bodies without resolving each
 * tensor or changing argument tables per token. Each backend defines this
 * private bridge separately; no binary links both definitions.
 *
 * Both local corpora queried and rules read for header design:
 * cpp-guidelines I.1 (#ri-explicit), R.1 (#rr-raii): borrowed views identify
 * their resource owner. cpp-perf-guidelines GPU.6
 * (guidelines/gpu/GPU.6-batch-tiny-gpu-work.md): stable command bindings;
 * GPU.9 (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md):
 * buffer views do not create independent ownership or permit early reuse.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include "gpu/device.h"

namespace MTL4 {
class CommandQueue;
}
namespace MTL {
class Buffer;
class Device;
class ComputePipelineState;
struct Size;
}  // namespace MTL
namespace dolittle::gpu::metal {
class CommandBody;
struct BufferBinding {
  MTL::Buffer* buffer;
  std::size_t offset;
  std::size_t bytes;
  std::uint64_t address;
};
}  // namespace dolittle::gpu::metal
namespace dolittle::gpu::detail {
struct BackendAccess {
  static DeviceBuffer allocate(Device&, AllocationRequest);
  // Explicit view cap supports bounded native views over one physical backing.
  static DeviceBuffer allocate_viewed(Device&, AllocationRequest,
                                      std::size_t view_bytes);
  static HostBuffer allocate_staging(Device&, AllocationRequest);
  static Queue create_queue(Device&);
  static MTL::Device& native_device(Device&);
  static metal::BufferBinding binding(ConstDeviceRegion, Device&);
  static metal::BufferBinding binding(DeviceRegion, Device&);
  static metal::BufferBinding binding(HostRegion, Device&);
  static metal::BufferBinding binding(ConstHostRegion, Device&);
  static MTL4::CommandQueue& native_queue(Queue&, Device&);
  static void record(Queue&, Device&, metal::CommandBody*);
  static void dispatch(Queue&, Device&, MTL::ComputePipelineState&,
                       std::span<const metal::BufferBinding>, MTL::Size,
                       MTL::Size, std::span<const std::byte> parameters = {},
                       std::size_t threadgroup_bytes = 0);
  static std::shared_ptr<MTL::ComputePipelineState> pipeline(
      Device&, const char* library, const char* name, bool indirect);
  static void poison(Device&) noexcept;
  static void drain_noexcept(Device&) noexcept;
};
}  // namespace dolittle::gpu::detail
