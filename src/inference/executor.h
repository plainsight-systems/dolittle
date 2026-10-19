/*
 * Prepared full forward execution; numerical verification is recorded separately.
 * Axis: X1, execution planning. This first interface is a complete causal
 * forward pass plus single-turn cached prefill/decode. Sampling is separate.
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
 *
 * Incremental execution: cached target execution.
 * Input: a bound model, numerical mode, capacity C, maximum prompt chunk Bmax,
 * selected observation sites and device/host byte budgets. Output: a serial,
 * single-sequence plan owning weights, cache, scratch, staging and token history.
 * The caller establishes the tokenizer/protocol pairing with this checkpoint;
 * the executor validates token IDs and mathematical capabilities, not chat text.
 *
 * Cached algorithm:
 * 1. Prepare the model's incremental recipe and all resource requirements for
 *    up to Bmax new positions and a cache prefix up to C. The same schedule
 *    accepts B=1 for decode. Reuse one weight arena across both.
 * 2. For each consume(ids), let S be committed positions and B be ids.size().
 *    Require 0 < B <= Bmax, S+B <= C and valid IDs before submitting anything.
 *    Use absolute RoPE positions S+i; existing cached keys are already rotated.
 * 3. Execute every block for these B positions. Copy its new K/V into that
 *    block's cache at [S,S+B); attention reads keys through S+i for query i.
 *    Copy requested residual rows before scratch is reused. A capture's absolute
 *    position is S+i even though its temporary row index is i.
 * 4. If requested, project only the final new row to vocabulary logits. Batch
 *    this row and the chunk's captures into the same readback/completion point.
 * 5. After completion, return owned chunk bytes and commit S+B atomically across
 *    blocks. Generation appends them to its retained host result. Earlier chunks
 *    then survive a later poisoned-device failure without another GPU readback.
 * 6. On failure, drain work and invalidate this plan/cache. Do not return the
 *    failed chunk or advance S. finish seals the plan; no restart or resume.
 * Malformed calls rejected before submission leave the plan usable. Failure
 * after submission makes it unusable. Result bytes/views outlive the plan.
 *
 * Implementation design: prepare bounded shapes once, pass current S/B as
 * checked launch parameters, and keep scratch layouts at admitted capacity.
 * Do not build a model recipe, upload weights or allocate device memory per
 * emitted token. FullForwardPlan remains the independent no-cache path.
 * Cost: cache is sum(2*C*Hkv*d*element_bytes), scratch scales with Bmax,
 * device capture staging with Bmax*sum(site_width*element_bytes). Host retained
 * captures scale with C; host staging and per-call result bytes are separate
 * peak costs. Host admission counts payload buffers, positions and sampling
 * scratch; excludes archive mappings, tokenizer tables, allocator overhead and
 * schedule/owner metadata. This is not a bound on process resident memory. Each sampling step has one readback wait. Capture bytes join that
 * existing boundary instead of adding a host wait for each block.
 * Optimization: resident weights/cache, reused chunk/decode schedules, batched
 * capture/logit transfers. Include prompt tails and zero-logit finalization in
 * preparation; padding to Bmax must never become a valid attention position.
 * Overlay: transformer-execution.md, "Prefill and decoding" and "Storage lifetimes".
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post), R.1 (#rr-raii): transactional prefix commit and
 * owned host results preserve completed chunks after a later device failure.
 * cpp-perf-guidelines GPU.9: admit persistent/cache/scratch arenas separately.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines GPU.1: batch captures at the required sampling boundary.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include "inference/observations.h"
#include "inference/cache/cache.h"
#include <optional>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace dolittle::models {
class BoundCheckpoint;
}
namespace dolittle::gpu {
class Device;
}

namespace dolittle::inference {

struct ExecutionId {
    std::uint64_t value;
};

struct ForwardOptions {
    tensors::ComputationMode numerical_mode{};
    PositionSelection logit_positions;
    std::vector<ObservationRequest> observations;
    ObservationLimits observation_limits{};
    std::size_t max_device_bytes{};
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
    ForwardResult(ForwardResult &&) noexcept;
    ForwardResult &operator=(ForwardResult &&) noexcept;
    ForwardResult(const ForwardResult &) = delete;
    ForwardResult &operator=(const ForwardResult &) = delete;

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
    FullForwardPlan(FullForwardPlan &&) noexcept;
    FullForwardPlan &operator=(FullForwardPlan &&) noexcept;
    FullForwardPlan(const FullForwardPlan &) = delete;
    FullForwardPlan &operator=(const FullForwardPlan &) = delete;

    [[nodiscard]] models::SequenceLength sequence_length() const noexcept;
    [[nodiscard]] std::size_t estimated_peak_device_bytes() const noexcept;
    [[nodiscard]] std::size_t host_output_bytes() const noexcept;
    [[nodiscard]] ForwardResult execute(std::span<const std::uint32_t> token_ids, ExecutionId execution_id);

  private:
    struct Impl;
    explicit FullForwardPlan(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;

    friend FullForwardPlan prepare_full_forward(gpu::Device &, const models::BoundCheckpoint &,
                                                models::SequenceLength, const ForwardOptions &);
};

[[nodiscard]] FullForwardPlan prepare_full_forward(gpu::Device &device,
                                                   const models::BoundCheckpoint &checkpoint,
                                                   models::SequenceLength sequence_length,
                                                   const ForwardOptions &options);

// Single-turn incremental execution with persistent K/V.
struct CachedForwardOptions {
    tensors::ComputationMode numerical_mode;
    cache::SequenceId sequence;
    std::size_t capacity;
    std::size_t max_prefill_chunk;
    std::vector<models::ObservationPoint> capture_sites;
    ObservationLimits observation_limits;
    std::size_t max_device_bytes;
    std::size_t max_host_bytes;
};
struct OwnedLogitRow {
    models::TokenPosition consumed_position;
    std::size_t vocabulary_size;
    tensors::ElementType storage_type;
    std::vector<std::byte> bytes;
};
struct ConsumedChunk {
    models::TokenPosition first_position{};
    std::vector<std::uint32_t> tokens;
    std::optional<OwnedLogitRow> next_logits;
    std::vector<OwnedCapture> captures;
};
class CachedForwardPlan {
  public:
    ~CachedForwardPlan();
    CachedForwardPlan(CachedForwardPlan &&) noexcept;
    CachedForwardPlan &operator=(CachedForwardPlan &&) noexcept;
    CachedForwardPlan(const CachedForwardPlan &) = delete;
    CachedForwardPlan &operator=(const CachedForwardPlan &) = delete;
    [[nodiscard]] std::string_view checkpoint_identity() const noexcept;
    [[nodiscard]] cache::SequenceId sequence_id() const noexcept;
    [[nodiscard]] tensors::ComputationMode numerical_mode() const noexcept;
    [[nodiscard]] cache::Prefix prefix() const noexcept;
    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::span<const models::ObservationSite> capture_sites() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t max_prefill_chunk() const noexcept;
    [[nodiscard]] std::size_t estimated_peak_device_bytes() const noexcept;
    [[nodiscard]] ConsumedChunk consume(std::span<const std::uint32_t> token_ids, bool need_next_logits);
    void finish();

  private:
    struct Impl;
    explicit CachedForwardPlan(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
    friend CachedForwardPlan prepare_cached_forward(gpu::Device &, const models::BoundCheckpoint &,
                                                    const CachedForwardOptions &);
};
[[nodiscard]] CachedForwardPlan prepare_cached_forward(gpu::Device &, const models::BoundCheckpoint &,
                                                       const CachedForwardOptions &);

} // namespace dolittle::inference
