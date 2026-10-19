/*
 * Prepared full forward execution; numerical verification is recorded separately.
 * Axis: X1, execution planning. This first interface is a complete causal
 * forward pass; persistent KV caching and autoregressive sampling are later.
 *
 * Contract: preparation borrows a verified BoundCheckpoint and Device, which
 * must both outlive the plan. Checkpoint data remains immutable; callers must
 * not concurrently use or reconfigure the device during plan operations.
 * It validates the model's
 * forward/observation capabilities, shape, dtype, selected logits and memory
 * budget before tensor allocation/upload. It copies/resolves metadata rather than
 * retaining the caller's vectors. Unknown sites/modes or incompatible inputs
 * fail explicitly. Invalid input -> invalid_argument; size/budget overflow ->
 * length_error; device failure -> runtime_error; host allocation -> bad_alloc.
 *
 * execute consumes exactly the prepared number of token IDs, all within the
 * bound vocabulary. Sequence positions are [0,N), without padding, truncation,
 * sliding eviction or persistent cache. The ID span is borrowed only for the
 * call. It does not sample tokens or load an NLA. A plan is reusable serially
 * with different token contents of the same length; it is not reentrant.
 *
 * execute returns only after requested host logits and captures are complete.
 * ForwardResult owns all returned bytes and position metadata independently
 * of the plan, checkpoint and device. Views remain valid until result move or
 * destruction. Results carry verified checkpoint identity and caller-supplied
 * execution ID (unique within the consuming experiment/harness), an owned
 * copy of the consumed token IDs, and the executed numerical mode. Failed calls
 * yield no successful result and drain dependent work before reusing/freeing
 * scratch. Device failure makes the plan unusable until it is recreated.
 * Moved-from plans/results support destruction/reassignment only.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Full model flow",
 * "One Qwen3 transformer block", "First forward pass", and "Storage lifetimes":
 * 1. Prepare the selected model recipe, output rows, observation requests,
 *    kernel choices and live-buffer budget. No hard-coded observation block.
 * 2. Embed exact IDs, evaluate each block with full causal attention and
 *    current-pass K/V, and arrange requested copies at advertised boundaries.
 * 3. Apply the selected output path and project only requested logit rows.
 * 4. Complete batched readbacks, attach provenance, and return owned results.
 * No persistent KV cache survives this operation; scratch can be reused.
 * The bound model supplies ForwardRecipe; preparation validates its single
 * producers, parameter bindings, dependencies and observation values. Operation
 * preparation resolves supported kernels and operand/workspace alignment.
 * Plan owns uploaded weights, one ordered queue, scratch and bounded staging;
 * storage_plan.h supplies pure checked scratch placement after schedule last
 * uses are known; weight/output arenas and workspace are admitted separately.
 * the checkpoint owns host payloads. Captures extend source last-use intervals.
 * Raw capture copies are scheduled before reusing the source, with host-ready
 * results established by one final completion point, not per-block waits.
 *
 * Design and cost: preparation includes weights, scratch, output staging and
 * capture/position metadata in admission; report estimated device peak and
 * host output bytes (logical payload and metadata, including token provenance;
 * excludes allocator capacity/overhead and plan-owned transfer staging). Device
 * peak is admitted arena bytes, excluding driver/context reservations and
 * allocator alignment overhead. Actual allocations may still
 * fail. Logit position lists follow PositionSelection's strict ordering and
 * bounds rules, including an empty list selecting no rows. Raw logits retain
 * their declared storage dtype. Reusing the plan avoids rebuilding schedules;
 * result-owned host output allocation remains an explicit per-call cost.
 * Optimization: intermediates stay on device; no per-observation host wait.
 * A final readback wait is intentional at this numerical-harness boundary.
 * No overlap or throughput improvement is claimed before measurement.
 *
 * Corpus guidance queried and full rules read for implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): host-ready results and failure cleanup are explicit.
 * cpp-guidelines R.1 (#rr-raii): owned scratch/results follow plan/result lifetime.
 * cpp-perf-guidelines GPU.1: budget batched output transfers, not each layer's
 * intermediates. Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.7: order source reuse/copies by completion, not a
 * device-wide wait per hook. Source: guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 * cpp-perf-guidelines COPY.5: execution plans/results transfer unique ownership.
 * Source: guidelines/copy-move/COPY.5-resource-owning-types-are-move-only.md.
 * cpp-perf-guidelines GPU.9: scratch reuse includes observation last uses.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#pragma once

#include "inference/observations.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace dolittle::models { class BoundCheckpoint; }
namespace dolittle::gpu { class Device; }

namespace dolittle::inference {

struct ExecutionId {
    std::uint64_t value;
};

struct ForwardOptions {
    tensors::ComputationMode numerical_mode;
    PositionSelection logit_positions;
    std::vector<ObservationRequest> observations;
    ObservationLimits observation_limits;
    std::size_t max_device_bytes;
};

struct LogitRows {
    std::span<const models::TokenPosition> positions;
    std::size_t vocabulary_size;
    tensors::ElementType storage_type;
    std::span<const std::byte> bytes;
};

class FullForwardPlan;

class ForwardResult {
public:
    ~ForwardResult();
    ForwardResult(ForwardResult&&) noexcept;
    ForwardResult& operator=(ForwardResult&&) noexcept;
    ForwardResult(const ForwardResult&) = delete;
    ForwardResult& operator=(const ForwardResult&) = delete;

    [[nodiscard]] ExecutionId execution_id() const noexcept;
    [[nodiscard]] std::string_view checkpoint_identity() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> token_ids() const noexcept;
    [[nodiscard]] tensors::ComputationMode numerical_mode() const noexcept;
    [[nodiscard]] LogitRows logits() const noexcept;
    [[nodiscard]] std::span<const CapturedRows> captures() const noexcept;

private:
    struct Impl;
    explicit ForwardResult(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
    friend class FullForwardPlan;
};

class FullForwardPlan {
public:
    ~FullForwardPlan();
    FullForwardPlan(FullForwardPlan&&) noexcept;
    FullForwardPlan& operator=(FullForwardPlan&&) noexcept;
    FullForwardPlan(const FullForwardPlan&) = delete;
    FullForwardPlan& operator=(const FullForwardPlan&) = delete;

    [[nodiscard]] models::SequenceLength sequence_length() const noexcept;
    [[nodiscard]] std::size_t estimated_peak_device_bytes() const noexcept;
    [[nodiscard]] std::size_t host_output_bytes() const noexcept;
    [[nodiscard]] ForwardResult execute(
        std::span<const std::uint32_t> token_ids, ExecutionId execution_id);

private:
    struct Impl;
    explicit FullForwardPlan(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;

    friend FullForwardPlan prepare_full_forward(
        gpu::Device&, const models::BoundCheckpoint&, models::SequenceLength,
        const ForwardOptions&);
};

[[nodiscard]] FullForwardPlan prepare_full_forward(
    gpu::Device& device,
    const models::BoundCheckpoint& checkpoint,
    models::SequenceLength sequence_length,
    const ForwardOptions& options);

}  // namespace dolittle::inference
