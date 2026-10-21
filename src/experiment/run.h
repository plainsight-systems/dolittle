/*
 * Steering scheduling, R3: resolve the retained sample; admit two branch
 * results and fresh sequence capacities before releasing AV. Tokenize both AR
 * descriptions, run sequential GPU forwards with shared resident AR weights,
 * compute their normalized difference on device and retain its small owner.
 * Release AR, load target once, replay identical fixed IDs with fresh KV/RNG
 * for baseline and steered continuations. Apply the edit once in the second
 * branch. Restore text-classification state from the original generated prefix.
 * Retain trial results without replacing the original run. Cancellation stops
 * the current phase and preserves completed branch output. Budget all retained
 * device owners alongside active model resources. No CPU vector processing.
 * Overlay: steering.md, Device ownership and model phases.
 * Header consultation: Core I.1 (#ri-explicit), R.1 (#rr-raii), explicit
 * inputs and lifetimes; performance GPU.1, resident intermediates, GPU.9,
 * bounded persistent owners (guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md).
 * Both corpora queried/read, 2026-10-20.

 * Axis R3: sequential target/AV phases with durable experiment records.
 * Contract: Run owns pairing, providers, text helpers, records and optional
 * resident AV weights; it borrows Device, which must outlive capture leases.
 * Providers load only verified sources within their host allowance. Operations
 * are serial. Target executes once; later interpretations use fresh sequence
 * state and may reuse AV weights. Rejected preflight is retryable; unexpected
 * target failure closes the run. close releases model resources, while the
 * immutable record and its capture leases remain readable until destruction.
 *
 * Algorithm: logical-overview.md experiment phases; execution-boundary.md
 H0-H4.
 * 1. Prepare/tokenize target text, validate pairing and budgets, load target
 *    weights, then prepare and execute the complete GPU sequence. The host
 *    observer detokenizes/classifies metadata without scheduling GPU progress.
 * 2. Build the retained record from completed device capture leases and exact
 *    IDs. Release target sequence/cache/weights; retained captures remain
 valid.
 * 3. Resolve a selected consumed row, validate the trained pair convention and
 *    construct a device normalization descriptor. Admit the AV model/sequence
 *    alongside retained captures. Reuse AV weights on subsequent requests.
 * 4. GPU normalization/status gates AV prefill; marker embedding replacement
 *    precedes block0. Run the same GPU generation path, retain reported source
 *    norm and exact AV IDs, decode them and extract explanation tags.
 * 5. Append a completed/cancelled/failed interpretation with honest phase costs
 *    and provenance. A load failure can be retried; device failure prevents
 *    pretending execution succeeded. Cancellation is an external stop request.
 * Host-reference selection remains an explicitly composed diagnostic path with
 * host captures/conversion. It is not a live fallback.
 * Cost: peak device=max(target phase, retained captures+AV phase), including
 KV,
 * scratch, sampler, constants, graph state and prepared journal allocations.
 * Host estimates include metadata, verified loading and diagnostic staging.
 * GPU normalization is included in generation duration; conversion_seconds
 * applies only to CPU reference conversion. Estimates exclude driver overhead.
 * Optimization: one model weight set resident at a time, capture arenas survive
 * the switch, and repeated AV requests reuse weights without reloading target.
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

#include <functional>

#include "experiment/record.h"

namespace dolittle::experiment {
struct LoadedModel {
  tensors::formats::SourceIdentity source;
  std::unique_ptr<models::BoundCheckpoint> checkpoint;
  std::size_t
      host_working_bytes;  // Loader allowance, excludes mapped address space.
};
// The provider must enforce the supplied active-loader host working
// allowance.
using ModelProvider =
    std::function<LoadedModel(std::size_t max_host_working_bytes)>;
struct RunInputs {
  std::unique_ptr<nla::Pair> pair;
  ModelProvider target;
  ModelProvider verbalizer;
  tensors::ComputationMode target_mode{};
  tensors::ComputationMode verbalizer_mode{};
  std::unique_ptr<tokenizers::Tokenizer> target_tokenizer;
  std::unique_ptr<tokenizers::Tokenizer> verbalizer_tokenizer;
  std::unique_ptr<text::Protocol> target_protocol;
  inference::sampling::TokenSelector selector;
  std::string backend_identity;
  std::size_t metadata_host_bytes{};  // Admitted owned tokenizers, pair and
                                      // provider metadata.
  inference::sampling::Execution selection_execution =
      inference::sampling::Execution::device;
  ModelProvider reconstructor{};
  std::unique_ptr<tokenizers::Tokenizer> reconstructor_tokenizer{};
};
struct RunLimits {
  std::size_t max_device_bytes{};
  std::size_t max_host_bytes{};
  std::size_t max_prefill_chunk{};
  std::size_t
      max_target_tokens{};  // Prompt + generated, not a truncation policy.
  std::size_t max_verbalizer_tokens{};
  std::size_t max_decoded_bytes{};
  RecordLimits record;
};
enum class Phase : std::uint8_t { empty, retained, verbalizer_ready, closed };
// Live progress contract: borrowed values expire when the callback returns.
enum class StreamRole : std::uint8_t { target, verbalizer, baseline, steered };
enum class WorkPhase : std::uint8_t {
  loading_target,
  generating_target,
  releasing_target,
  target_retained,
  loading_verbalizer,
  generating_verbalizer,
  verbalizer_ready,
  loading_reconstructor,
  reconstructing,
  generating_baseline,
  generating_steered,
  closed
};
struct RunProgress {
  StreamRole role;
  std::size_t first_position;
  std::size_t prompt_tokens;
  std::size_t consumed_tokens;
  std::reference_wrapper<const tokenizers::DecodedBytes> delta;
  std::span<const text::TokenSegment> segments;
  std::optional<models::ObservationSite> capture_site;
};
struct StreamBounds {
  std::size_t prompt_tokens, generated_tokens, prefill_chunk, token_bytes,
      decoded_bytes;
};
class RunObserver {
 public:
  RunObserver() = default;
  RunObserver(const RunObserver&) = delete;
  RunObserver& operator=(const RunObserver&) = delete;
  RunObserver(RunObserver&&) = delete;
  RunObserver& operator=(RunObserver&&) = delete;
  virtual ~RunObserver() = default;
  [[nodiscard]] virtual bool phase(WorkPhase) noexcept = 0;
  virtual void prepare_comparison(StreamBounds) {}
  [[nodiscard]] virtual bool comparison_started(const SteeringTrial&) noexcept {
    return true;
  }
  [[nodiscard]] virtual bool prepare_stream(StreamBounds) noexcept {
    return true;
  }
  [[nodiscard]] virtual bool progress(const RunProgress&) noexcept = 0;
  // Idempotent failure signal, including failures while decoding a delta.
  virtual void publication_failed() noexcept = 0;
};
class Run {
 public:
  Run(gpu::Device&, RunId, RunInputs, RunLimits);
  ~Run();
  Run(const Run&) = delete;
  Run& operator=(const Run&) = delete;
  Run(Run&&) = delete;
  Run& operator=(Run&&) = delete;
  void generate(const text::Prompt&, inference::GenerationSettings,
                const std::stop_token& = {});
  [[nodiscard]] VariantId interpret(CaptureKey, inference::GenerationSettings,
                                    const std::stop_token& = {});
  void generate(const text::Prompt&, inference::GenerationSettings,
                RunObserver&, const std::stop_token& = {});
  [[nodiscard]] VariantId interpret(CaptureKey, inference::GenerationSettings,
                                    RunObserver&, const std::stop_token& = {});
  [[nodiscard]] TrialId steer(SteeringEdit, RunObserver&,
                              const std::stop_token& = {});
  [[nodiscard]] Phase phase() const noexcept;
  [[nodiscard]] const Record* record()
      const noexcept;  // Null until target result retained.
  [[nodiscard]] const measurement::PhaseMeasurements& measurements()
      const noexcept;
  void close();

 private:
  void generate_impl(const text::Prompt&, inference::GenerationSettings,
                     RunObserver*, const std::stop_token&);
  [[nodiscard]] VariantId interpret_impl(CaptureKey,
                                         inference::GenerationSettings,
                                         RunObserver*, const std::stop_token&);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::experiment
