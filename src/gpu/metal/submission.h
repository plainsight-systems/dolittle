/*
 * Implemented foundation, 2026-10-21; header design approved before
 * implementation. Axis X4: a bounded, completely prepared Metal submission
 * timeline. Contract: Device and borrowed allocations outlive this nonmoving
 * owner. Construct on an idle Queue with W>0 and a nonzero command-storage
 * limit; encode/seal windows once each, in index order, before start. The owner
 * retains command buffers, allocators, argument tables and residency references
 * through drain. No allocator reset or argument-table mutation while any
 * encoded command can read it. start is one-shot. Collection may lag until the
 * entire timeline finishes; it never schedules another window. Invalid
 * lifecycle/index -> invalid_argument; Metal failures -> runtime_error and
 * poison the timeline. Destructor drains without throwing.
 *
 * Algorithm: metal-port.md, Metal scheduling.
 * 1. Allocate W command windows and native command storage during preparation.
 *    Encode producer/consumer barriers within each window. All windows access
 *    stable device sequence state; GPU control determines their active bodies.
 * 2. Seal each window. End it with publication into its own shared output
 *    allocation, which later GPU windows never touch. Keep live sequence state
 *    private; do not let the CPU poll a buffer the GPU is still modifying.
 * 3. At start, commit each already-prepared window and signal a monotonically
 *    increasing shared-event value after it. Order the next window after that
 *    signal. Submit the complete timeline without reading output or waiting
 *    for a token. This is CPU setup, not token-dependent scheduling.
 * 4. wait_prefix(w) waits for completion of windows [0,w), then their immutable
 *    shared output is CPU-readable. The GPU has no wait on that collection.
 * 5. drain retires all commands; only then may allocations/allocators be freed.
 *
 * Cost: O(W) host encoding/commit work and retained command storage, all
 * admitted before start; shared-event boundaries impose GPU ordering cost.
 * Optimization: the first output window ends after first selection for TTFS;
 * later windows batch iterations and reuse ICB bodies. Choose their size from
 * measured submission/drain/publication costs, not a CPU token-poll loop.
 * No per-token host callbacks or resource allocation. Cancellation is a
 * separate sequence control command, not an acknowledgement of a window.
 *
 * Both local corpora queried and rules read for header design:
 * cpp-guidelines R.1 (#rr-raii), I.7 (#ri-post): completion precedes
 * read/reuse. cpp-perf-guidelines GPU.6
 * (guidelines/gpu/GPU.6-batch-tiny-gpu-work.md), GPU.7
 * (guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md), GPU.8
 * (guidelines/gpu/GPU.8-make-barriers-describe-real-hazards.md): reusable
 * commands, independent publication slots and explicit producer/consumer
 * ordering. Apple MTL4CommandQueue signalEvent/waitForEvent and Serenity's
 * allocator lifetime supply the API pattern; on-device proof belongs to
 * implementation.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "gpu/device.h"

namespace MTL4 {
class CommandBuffer;
}  // namespace MTL4
namespace MTL {
class Allocation;
}  // namespace MTL
namespace dolittle::gpu::metal {
struct TimelineLimits {
  std::size_t windows;
  std::size_t max_command_bytes;
};
class Submission {
 public:
  Submission(Device&, Queue&, TimelineLimits);
  ~Submission();
  Submission(const Submission&) = delete;
  Submission& operator=(const Submission&) = delete;
  Submission(Submission&&) = delete;
  Submission& operator=(Submission&&) = delete;
  void retain(std::span<const ConstDeviceRegion>);
  void retain(std::span<const ConstHostRegion>);
  // Native ICB/argument storage also needs residency and an owning reference.
  void retain_native(MTL::Allocation&);
  [[nodiscard]] MTL4::CommandBuffer& begin_window(std::size_t index);
  void end_window(std::size_t index);
  [[nodiscard]] std::size_t command_bytes() const;
  [[nodiscard]] std::size_t host_bytes() const;
  void start();
  [[nodiscard]] std::size_t completed_prefix() const;
  void wait_prefix(std::size_t count);
  void drain();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::gpu::metal
