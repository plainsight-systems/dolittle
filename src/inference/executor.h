/*
 * Steering execution, X1: prepare_intervention validates the model-declared
 * block output, absolute prefix position, FP32 direction lease and strength.
 * At the producer, device position selects one row h[D]; GPU computes its norm
 * and stores BF16(h + alpha*norm(h)*direction) before consumers/captures.
 * reconstruct requires a vector-output model and runs autonomous prefill plus
 * the final vector head, without a sampler. The returned device allocation owns
 * D elements independently of scratch/weights. Plans drain before borrowers
 die.
 * Admit vector/status storage before graph capture; no hot allocations/copies
 * to host. Prefix replay uses fresh KV; target weights can serve both branches.
 * Overlay: steering.md, Device ownership and model phases.
 * Header design: Core I.1 (#ri-explicit), R.1 (#rr-raii); performance GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md), GPU.9
 * (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md).
 * Both corpora queried and rules read for this design, 2026-10-20.

 * Axis X1: prepare model recipes, bind storage and execute full or cached
 passes.
 * Contract: verified BoundCheckpoint and Device outlive borrowed plans. A
 * ResidentModel owns a checkpoint, uploaded weights and ordered queue; at most
 * one cached plan leases it. Destroy the plan before its resident model and
 * retain Device until all exported capture leases die. Failed plans are
 poisoned
 * and drained; they cannot resume. Allocation, shape/provenance/budget checks
 * happen before live launch. Model-provided observation sites determine hooks.
 *
 * Algorithm: transformer-execution.md full model/block flow and
 * execution-boundary.md sections 3-6.
 * 1. Validate the recipe and each operand. Assign immutable weights, contiguous
 *    per-layer K/V, persistent outputs and transient scratch. Extend source
 *    lifetimes through capture copies. Bind independent attention-scale slots;
 *    initialize 1/sqrt(head_width) on device before their first reader.
 * 2. For live generation, admit P prompt and G continuation positions, upload
 *    all prompt IDs and allocate independent retained device capture storage.
 *    Build fixed full-chunk, remainder, first-selection and decode graph
 bodies.
 *    Device position selects IDs, RoPE position, KV write/read windows and the
 *    destination row for every requested hook.
 * 3. Each body embeds IDs and runs the model's ordered block recipe: RMSNorm,
 *    Q/K/V projections, Q/K head norms, RoPE, causal grouped-query attention,
 *    output projection/residual, post-attention norm, gated SiLU MLP/residual.
 *    Capture raw advertised values after their producer and before scratch
 *    reuse. The head gathers the last row, normalizes and projects vocabulary
 *    logits; device sampling feeds the next embedding without readback.
 * 4. An AV override leases a completed source row and declares type/scale.
 *    DeviceInjection validates and normalizes it inside the root graph. Status
 *    gates all inference bodies. Copy its output over the marker embedding
 *    after lookup and before block 0, only in the containing prompt chunk.
 * 5. Start once; collection updates host metadata from immutable events. The
 *    GPU owns all progress decisions. finish drains and seals the plan. Capture
 *    leases outlive transient storage and target-to-AV weight replacement.
 *
 * FullForwardPlan is an explicit diagnostic path: execute an exact fixed token
 * trace with optional host injection, capture selected rows and batch readbacks
 * at one completion point. Cached consume/consume_view similarly replay known
 * IDs for oracle comparison; borrowed views expire at the next consume. These
 * APIs reject use after live preparation. Host injection cannot enter a live
 * sequence; device injection cannot silently enter a host diagnostic pass.
 * Tensor storage/rounding is model-selected BF16/FP32 with FP32 accumulation;
 * this executor does not alter arithmetic to match a reference.
 * Resident loading returns only after every weight upload completes. Its
 * bounded staging uses two reusable slots: wait for a slot's prior transfer,
 * fill it from the checkpoint, submit/mark its upload, then rotate. One-byte
 * budgets fall back to one slot. Total staging never exceeds the supplied
 * allowance; owners drain pending uses on failure. Byte copies preserve all
 * checkpoint values. Optimization: CPU filling overlaps another slot's DMA;
 * resources are allocated before submission. Upload design/implementation
 * consultation, 2026-10-20 (both servers queried in both phases):
 * cpp-guidelines R.1 (#rr-raii) and I.7 (#ri-post) require owned
 * lifetimes and completed-return semantics. cpp-perf-guidelines GPU.7 requires
 * resource-specific completion and bounded multi-buffering; source:
 * guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 *
 * Design/cost: one prepared graph O(recipe size), not O(sequence length).
 * Weights O(parameter bytes); KV O(layers*capacity*kv_heads*head_width*2*b);
 * capture payload O((P+G)*sum(row_bytes)); scratch O(chunk*model widths).
 * The immutable journal is O(P+G) small metadata. Separate budgets cover host
 * staging/diagnostic buffers and device state, constants, sampler and captures.
 * Optimization: direct device-positioned KV writes avoid append copies;
 * graph execution and GPU NLA conversion allocate nothing or read tensors back.
 * CPU-only oracle replay is explicit and never chosen as a live fallback.
 *
 * Corpus consultation, header design and implementation, 2026-10-20: both local
 * servers queried separately in each phase and the cited rules read in full.
 * cpp-guidelines I.1 (#ri-explicit), R.1 (#rr-raii): explicit residency,
 * completion, ownership and failure contracts. cpp-perf-guidelines GPU.1,
 * GPU.6, GPU.7, GPU.9: keep tensor intermediates/decisions on device, batch
 * dispatch, decouple observation and separate persistent from transient arenas.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Related GPU.6/7/9 entries describe dispatch, staging and storage.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string_view>
#include <vector>

#include "inference/cache/cache.h"
#include "inference/decoding/device.h"
#include "inference/intervention.h"
#include "inference/observations.h"
#include "inference/sampling/device.h"

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
// Device normalization input; the pair specifies the law without evaluating it.
struct DeviceNormalization {
  DeviceCapture source;
  tensors::ElementType source_type;
  float scale;
};
// Replacement applied after lookup, before block 0.
struct EmbeddingOverride {
  models::TokenPosition position;
  std::size_t width;
  tensors::ElementType storage_type;
  std::vector<std::byte> bytes;
  std::optional<DeviceNormalization> device{};
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
  [[nodiscard]] ForwardResult execute(std::span<const std::uint32_t> token_ids,
                                      ExecutionId execution_id);
  [[nodiscard]] ForwardResult execute(std::span<const std::uint32_t> token_ids,
                                      ExecutionId execution_id,
                                      const EmbeddingOverride&);

 private:
  [[nodiscard]] ForwardResult execute_impl(std::span<const std::uint32_t>,
                                           ExecutionId,
                                           const EmbeddingOverride*);
  struct Impl;
  explicit FullForwardPlan(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;

  friend FullForwardPlan prepare_full_forward(gpu::Device&,
                                              const models::BoundCheckpoint&,
                                              models::SequenceLength,
                                              const ForwardOptions&);
};

[[nodiscard]] FullForwardPlan prepare_full_forward(
    gpu::Device& device, const models::BoundCheckpoint& checkpoint,
    models::SequenceLength sequence_length, const ForwardOptions& options);

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
struct ConsumedView {
  models::TokenPosition first_position{};
  std::span<const std::uint32_t> tokens;
  std::optional<LogitRows> next_logits;
  std::span<const CapturedRows> captures;
  std::optional<std::uint32_t> selected_token;
};
class ResidentModel;
class CachedForwardPlan {
 public:
  ~CachedForwardPlan();
  CachedForwardPlan(CachedForwardPlan&&) noexcept;
  CachedForwardPlan& operator=(CachedForwardPlan&&) noexcept;
  CachedForwardPlan(const CachedForwardPlan&) = delete;
  CachedForwardPlan& operator=(const CachedForwardPlan&) = delete;
  [[nodiscard]] std::string_view checkpoint_identity() const noexcept;
  [[nodiscard]] cache::SequenceId sequence_id() const noexcept;
  [[nodiscard]] tensors::ComputationMode numerical_mode() const noexcept;
  [[nodiscard]] cache::Prefix prefix() const noexcept;
  [[nodiscard]] std::size_t vocabulary_size() const noexcept;
  [[nodiscard]] std::span<const models::ObservationSite> capture_sites()
      const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t max_prefill_chunk() const noexcept;
  [[nodiscard]] std::optional<models::TokenPosition>
  embedding_override_position() const noexcept;
  [[nodiscard]] std::size_t estimated_peak_device_bytes() const noexcept;
  // Admission estimate for staging, readback, retained rows and sampler work.
  [[nodiscard]] std::size_t estimated_peak_host_bytes() const noexcept;
  [[nodiscard]] ConsumedChunk consume(std::span<const std::uint32_t> token_ids,
                                      bool need_next_logits);
  void prepare_sampling();
  void prepare_intervention(ResidualIntervention);
  [[nodiscard]] DeviceCapture reconstruct(std::span<const std::uint32_t>,
                                          const std::stop_token& = {});
  // One prepared request owns prefill, first selection and decode. Returned
  // events are observation only; retained capture owners outlive this plan.
  struct DeviceGenerationRequest {
    sampling::Settings sampling;
    std::uint64_t seed;
    std::span<const std::uint32_t> prompt_ids;
    std::span<const std::uint32_t> stop_ids;
    std::size_t max_new_tokens;
    bool forward_only = false;
  };
  void prepare_device_generation(const DeviceGenerationRequest&);
  void start_sequence();
  void request_stop() noexcept;
  [[nodiscard]] DeviceSequence::Event collect_sequence(const std::stop_token&);
  [[nodiscard]] std::span<const DeviceCapture> device_captures() const noexcept;
  [[nodiscard]] ConsumedView consume_view(
      std::span<const std::uint32_t>, bool,
      std::optional<sampling::DeviceDraw> draw = {});
  void finish();

 private:
  struct Impl;
  explicit CachedForwardPlan(std::unique_ptr<Impl>);
  static CachedForwardPlan prepare_impl(gpu::Device&,
                                        const models::BoundCheckpoint&,
                                        const CachedForwardOptions&,
                                        ResidentModel*,
                                        std::optional<EmbeddingOverride>);
  std::unique_ptr<Impl> impl_;
  friend CachedForwardPlan prepare_cached_forward(
      gpu::Device&, const models::BoundCheckpoint&,
      const CachedForwardOptions&);
  friend CachedForwardPlan prepare_cached_forward(
      ResidentModel&, const CachedForwardOptions&,
      std::optional<EmbeddingOverride>);
};
[[nodiscard]] CachedForwardPlan prepare_cached_forward(
    gpu::Device&, const models::BoundCheckpoint&, const CachedForwardOptions&);

struct ResidentOptions {
  tensors::ComputationMode numerical_mode;
  std::size_t max_device_bytes;
  std::size_t max_host_staging_bytes;
};
// Stable owner: moving it would invalidate sequence leases.
class ResidentModel {
 public:
  ~ResidentModel();
  ResidentModel(const ResidentModel&) = delete;
  ResidentModel& operator=(const ResidentModel&) = delete;
  ResidentModel(ResidentModel&&) = delete;
  ResidentModel& operator=(ResidentModel&&) = delete;
  [[nodiscard]] std::string_view checkpoint_identity() const noexcept;
  [[nodiscard]] std::size_t resident_device_bytes() const noexcept;
  [[nodiscard]] bool usable() const noexcept;

 private:
  struct Impl;
  explicit ResidentModel(std::unique_ptr<Impl>);
  friend class CachedForwardPlan;
  std::unique_ptr<Impl> impl_;
  friend std::unique_ptr<ResidentModel> load_resident_model(
      gpu::Device&, std::unique_ptr<models::BoundCheckpoint>,
      const ResidentOptions&);
  friend CachedForwardPlan prepare_cached_forward(
      ResidentModel&, const CachedForwardOptions&,
      std::optional<EmbeddingOverride>);
};
[[nodiscard]] std::unique_ptr<ResidentModel> load_resident_model(
    gpu::Device&, std::unique_ptr<models::BoundCheckpoint>,
    const ResidentOptions&);
[[nodiscard]] CachedForwardPlan prepare_cached_forward(
    ResidentModel&, const CachedForwardOptions&,
    std::optional<EmbeddingOverride> = std::nullopt);

}  // namespace dolittle::inference
