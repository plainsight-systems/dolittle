/*
 * Metal and CUDA execution implement the same interface. Metal uses a
 * bounded prepared timeline (kernels/metal/sequence_plan.h), reusable ICB
 * bodies and device control (kernels/metal/decode.h). Its host metadata and
 * command storage scale with the admitted sequence, unlike CUDA's conditional
 * graph. The public sequence semantics, dynamic hooks and collector-independent
 * progress remain identical. Metal output is published through completed,
 * separately owned windows; CUDA's mapped release/acquire mechanism below is
 * CUDA-specific. Static device_bytes/host_bytes must not masquerade as total
 * Metal admission: the backend exposes prepared command/storage
 * costs through prepared_resources() after complete() and before start().
 * That result excludes borrowed model/KV/capture owners; command_bytes is a
 * diagnostic subset already included in owned, never added to it again.
 * Costs include private state, publication buffers, native command allocation
 * and host metadata. Exceeding admission rejects the prepared request before
 * start, releasing its unused resources. Earlier estimates bound preparation
 * itself; querying the actual cost is not permission to allocate unboundedly.
 * Cancellation uses a separate device
 * control command, never host iteration over sampled tokens.
 * Overlay: metal-port.md, Metal scheduling. Both corpora queried/read for this
 * design: cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post); performance GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md), GPU.6
 * (guidelines/gpu/GPU.6-batch-tiny-gpu-work.md), GPU.7
 * (guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md): preserve the device
 * execution boundary while exposing each backend's real costs and ordering.
 *
 * Forward-only and intervention control, X1/X4: forward_only executes prefill
 * and the final output stage once, then completes without random draws or
 * decode. Mutation status/count are borrowed device words; stage transitions
 * latch failures and terminal publication checks exactly one application.
 * Position stays device-owned. CPU collection cannot schedule continuation.
 * O(1) extra control metadata, no per-token allocation or transfer.
 * Overlay: steering.md, Device ownership and model phases.
 * Header design: Core I.1 (#ri-explicit), R.1 (#rr-raii); performance GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md), GPU.9
 * (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md).
 * Both corpora queried and rules read for this design, 2026-10-20.

 * Axis X1/X4: autonomous prepared sequence execution and output publication.
 * Contract: Device, Queue and borrowed regions outlive this nonmoving owner.
 * P>0 prompt IDs and G generated IDs fit admitted capacity; B is a positive
 * maximum prefill chunk. Prepare all graph bodies before one start. Neither
 * prefill nor decoding waits for CPU collection. Events contain metadata only;
 * capture destinations are independent leased device allocations.
 *
 * Algorithm: execution-boundary.md sections 3 and 4. Initialize position s=0;
 * repeat full B-row prompt chunks on device, then an optional static remainder.
 * Each chunk copies its IDs from the uploaded device prompt, executes all
 * blocks, and commits its consumed range after hook copies. If continuation is
 * allowed, the final prompt residual produces the first selection. Decode
 * consumes that selected ID, commits its captures, then conditionally selects
 * another. GPU state owns position, budget, EOS and cancellation. A selected
 * token is always consumed before graceful termination. Invalid injection skips
 * all bodies.
 *
 * Publish immutable 32-byte records into P+2G+2 mapped host slots; never
 * publish raw activations there. GPU release / CPU acquire establishes
 * readiness, with no acknowledgement or slot reuse. CPU collection may lag to
 * completion. Capture and replacement kernels use the device position for D2D
 * addressing. Cost: O(P+G) small host metadata, O(1) device state, O(model
 * recipe) graph. Optimization: full-chunk/remainder/decode bodies have static
 * shapes and stable bindings, so graph size does not grow with sequence length.
 * No live allocation. Failure: CUDA errors throw; teardown ends partial
 * capture, stops and drains before freeing resources. Cancellation is an
 * external command, not scheduling. Header-design consultation 2026-10-20: Core
 * I.1 (#ri-explicit), R.1 (#rr-raii) give explicit ownership and completion;
 * Performance GPU.1/GPU.6/GPU.7/GPU.9 require resident intermediates, captured
 * work and separate observation storage. Sources
 * guidelines/gpu/GPU.1-keep-data-on-device.md and CUDA Graphs 4.2. Both local
 * corpora queried and full rules read before header design.
 */
#pragma once
#include <memory>
#include <stop_token>

#include "gpu/device.h"
#include "gpu/memory.h"

namespace dolittle::inference {
struct PreparedSequenceResources {
  gpu::MemoryFootprint owned;
  std::size_t command_bytes{};
};
class DeviceSequence {
 public:
  enum class Stage : std::uint8_t { prefill, tail, first_selection, decode };
  enum class Kind : std::uint8_t { consumed, selected, terminal, injection };
  enum class Reason : std::uint8_t { limit, eos, cancelled, failure };
  struct Event {
    Kind kind{};
    std::uint32_t status{};
    std::uint64_t position{}, count{};
    std::uint32_t token{};
    float norm{};
  };
  struct Shape {
    std::size_t prompt{}, generated{}, chunk{};
    bool forward_only = false;
  };
  DeviceSequence(gpu::Device&, gpu::Queue&, Shape);
  ~DeviceSequence();
  DeviceSequence(const DeviceSequence&) = delete;
  DeviceSequence& operator=(const DeviceSequence&) = delete;
  DeviceSequence(DeviceSequence&&) = delete;
  DeviceSequence& operator=(DeviceSequence&&) = delete;
  [[nodiscard]] static std::size_t device_bytes();
  [[nodiscard]] static std::size_t host_bytes(Shape);
  [[nodiscard]] gpu::ConstDeviceRegion position() const;
  void begin(gpu::ConstDeviceRegion selection, gpu::ConstDeviceRegion terminal,
             gpu::DeviceRegion token_input, gpu::ConstDeviceRegion prompt,
             gpu::ConstDeviceRegion injection_status = {},
             gpu::ConstDeviceRegion injection_norm = {},
             gpu::ConstDeviceRegion intervention_status = {},
             gpu::ConstDeviceRegion intervention_count = {});
  void begin_stage(Stage);
  void end_stage();
  // Ends the decode forward body and enters its conditional logit/sample tail.
  void begin_selection();
  void capture(gpu::ConstDeviceRegion source, gpu::DeviceRegion destination,
               std::size_t row_bytes, std::size_t rows);
  void replace(gpu::ConstDeviceRegion row, gpu::DeviceRegion embeddings,
               std::size_t prompt_position, std::size_t rows);
  void complete();
  // Prepared application-owned footprint. Metal includes native command
  // allocators; CUDA cannot query per-executable driver graph bytes
  // (command_bytes=0). Driver/pipeline opaque allocations require separate
  // admission headroom.
  [[nodiscard]] PreparedSequenceResources prepared_resources() const;
  void start();
  void request_stop() noexcept;
  void drain();
  [[nodiscard]] Event collect(std::size_t index, const std::stop_token&);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
inline constexpr std::size_t sequence_event_bytes = 32;
static_assert(sizeof(DeviceSequence::Event) == sequence_event_bytes);
}  // namespace dolittle::inference
