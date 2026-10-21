/*
 * CUDA and Metal implement these resource, transfer and completion contracts.
 * Axis X4: byte storage and ordering only, not transformer mathematics.
 * Contract: Device is stable/noncopyable and outlives every resource and queue
 * it creates. Owning buffer/queue handles are move-only; moved-from objects
 * permit only destruction/reassignment. Regions borrow an unmoved owner and
 * checked [offset, offset+size) bytes. No public vendor handles or host pointer
 * interpretation of device bytes. Allocations are uninitialized.
 * HostBuffer is transfer-suitable staging (pinned on CUDA). Access its bytes
 * only when no pending transfer uses the accessed range. Upload/download use
 * equal-sized regions; copy rejects overlapping source/destination ranges.
 * Submissions use one ordered queue; mark() identifies all prior submissions.
 * wait(point) blocks only for that prefix and makes completed readbacks
 * visible. Points are valid only for their originating live queue;
 * forged/future points fail. Cross-queue execution is outside the first-slice
 * contract.
 *
 * Buffers and staging must remain unmoved while submitted work references them.
 * Destructors drain outstanding uses before releasing memory; they never throw.
 * Explicit wait reports deferred failures as runtime_error and poisons the
 * device on unrecoverable failure; all plans/resources must then be recreated.
 * Cleanup waits are exceptional/teardown cost, never the normal loop protocol.
 * Invalid ranges/device identity/alignment -> invalid_argument; size overflow
 * -> length_error; device/allocation failure -> runtime_error; host bad_alloc
 * propagates. Host methods are serial, not concurrently callable on one device.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Preparing the first
 * execution recipe" and "Storage lifetimes": allocate admitted arenas/staging,
 * enqueue uploads, computations and copies in order, mark/read back/wait, then
 * release only completed storage. Backend kernels access internals privately.
 * Cost: bytes/alignment are explicit; drivers may reserve more than requested.
 * Optimization: forward plans reuse arenas and bounded staging, with no
 * allocation or CPU wait per observation. No overlap claim for one queue.
 *
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): release resources on normal and failure paths.
 * cpp-guidelines I.7 (#ri-post): host readiness follows explicit completion.
 * cpp-perf-guidelines GPU.7: queue-prefix dependencies replace global waits.
 * Source: guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 * cpp-perf-guidelines GPU.9: reuse arenas after proven last use.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines COPY.5: ownership transfers, never implicit buffer
 * copies. Source:
 * guidelines/copy-move/COPY.5-resource-owning-types-are-move-only.md.
 * Queue-completion design/implementation consultation, 2026-10-20:
 * Core I.7 (#ri-post) and CP.3 (#rconc-data): collection establishes host
 * readiness; in-flight slots have exclusive ownership until completion.
 * Performance GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md) keeps
 * sampled IDs and RNG state on-device. GPU.7
 * (guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md) separates submission
 * from readback through explicit prefix completion. Live device-owned decode
 * uses its own graph publication protocol instead of per-token events. Both
 * local corpora were queried and the full rules read in both work phases.
 * reserve_completion_points prepares event slots on an idle queue; it never
 * shrinks storage. Synchronous prefill reserves a completion slot before
 * submission; autonomous decode does not mark or wait per-token events. Waiting
 * an older point retires only its prefix, leaving later work and its readback
 * slot in flight.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace dolittle::gpu {
namespace detail {
struct BackendAccess;
}

struct AllocationRequest {
  std::size_t bytes;
  std::size_t alignment;
};
struct CompletionPoint {
  std::uint64_t queue_identity;
  std::uint64_t sequence;
};

class DeviceBuffer {
 public:
  ~DeviceBuffer();
  DeviceBuffer(DeviceBuffer&&) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&&) noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] std::size_t size_bytes() const noexcept;

 private:
  struct Impl;
  explicit DeviceBuffer(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
  friend struct detail::BackendAccess;
  friend class Queue;
};

class HostBuffer {
 public:
  ~HostBuffer();
  HostBuffer(HostBuffer&&) noexcept;
  HostBuffer& operator=(HostBuffer&&) noexcept;
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
  [[nodiscard]] std::span<std::byte> bytes() noexcept;
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

 private:
  struct Impl;
  explicit HostBuffer(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
  friend struct detail::BackendAccess;
  friend class Queue;
};

struct DeviceRegion {
  DeviceBuffer* owner;
  std::size_t offset;
  std::size_t bytes;
};
struct ConstDeviceRegion {
  const DeviceBuffer* owner;
  std::size_t offset;
  std::size_t bytes;
};
struct HostRegion {
  HostBuffer* owner;
  std::size_t offset;
  std::size_t bytes;
};
struct ConstHostRegion {
  const HostBuffer* owner;
  std::size_t offset;
  std::size_t bytes;
};

class Queue {
 public:
  ~Queue();
  Queue(Queue&&) noexcept;
  Queue& operator=(Queue&&) noexcept;
  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;

  void upload(DeviceRegion destination, ConstHostRegion source);
  void download(HostRegion destination, ConstDeviceRegion source);
  void copy(DeviceRegion destination, ConstDeviceRegion source);
  void reserve_completion_points(std::size_t count);
  [[nodiscard]] CompletionPoint mark();
  void wait(CompletionPoint point);
  void drain();

 private:
  struct Impl;
  explicit Queue(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
  friend struct detail::BackendAccess;
};

struct DeviceInfo {
  std::string backend;
  std::string name;
  std::string architecture;
  std::string driver_version;
  std::size_t total_memory_bytes{};
};

class Device {
 public:
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  virtual ~Device();
  [[nodiscard]] virtual const DeviceInfo& info() const noexcept = 0;
  [[nodiscard]] virtual DeviceBuffer allocate(AllocationRequest request) = 0;
  [[nodiscard]] virtual HostBuffer allocate_staging(
      AllocationRequest request) = 0;
  [[nodiscard]] virtual Queue create_queue() = 0;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

 protected:
  Device();
};

}  // namespace dolittle::gpu
