/*
 * R3 experiment scheduling and model retention. Run owns verified model roles,
 * text helpers and records; Device outlives all retained capture/model owners.
 * Public operations are serial. Every generation uses fresh KV and sequence
 * state. Models remain cached by exact source/binding identity and immutable
 * role-specific numerical mode only while complete-request budgets permit.
 *
 * Algorithm: logical-overview.md experiment phases; execution-boundary.md
 * H0-H4; metal-port.md Memory and model residency.
 * 1. Encode target text, reserve its output/captures and full execution peak,
 *    then admit target weights. GPU execution produces IDs and hook captures.
 * 2. Drain the sequence, retain its record and device captures, and release KV
 *    and scratch. Idle target weights may remain for later replay.
 * 3. On interpretation, reserve AV execution beside retained captures, evict
 *    idle roles oldest-first if needed, and load/reuse exact AV weights. Device
 *    normalization replaces the marker embedding before the first block.
 * 4. On steering, forward the original and edited explanation through AR with
 *    fresh caches. Normalize their difference on GPU; retain that vector while
 *    admitting target. Run matched baseline/steered sequences with fresh KV/RNG
 *    and one residual edit in the second branch. Preserve the original record.
 * 5. Decode completed output metadata and append the interpretation/trial.
 *    Collection never schedules transformer steps. Cancellation stops the
 *    active sequence and preserves completed output. Failed devices cannot
 *    reuse cached weights. Rejected admission leaves idle owners intact;
 *    a later upload failure may follow the admitted idle-model evictions.
 *
 * model_retention.h owns the pure decision. Run applies it only at drained
 * phase boundaries; no borrowed model is evicted. Unified memory additionally
 * charges mapped checkpoint payloads when physical sharing is unproven.
 * retained_model_bytes reports cached weight ownership; release_models frees
 * idle weights while keeping records usable; close also ends further work.
 * Cost: O(recipe) preparation/admission; no per-token allocation or CPU vector
 * processing. Optimization: reuse uploaded weights when the next full request
 * fits, without retaining obsolete sequence storage. Admission uses a
 * conservative tensor-live bound; actual prepared sequence costs are checked
 * before start. No precision, capacity or capture reduction to fit memory.
 * Both local corpora queried/read for design and implementation: Core R.1
 * (#rr-raii), I.7 (#ri-post), ES.103 (#res-overflow); performance GPU.1/GPU.9,
 * guidelines/gpu/GPU.1-keep-data-on-device.md and
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#pragma once

#include <functional>

#include "experiment/record.h"
#include "gpu/memory.h"

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
  // Request-boundary memory control: release completed weight owners while
  // keeping records/capture leases and the ability to interpret/steer.
  // No sequence may be in flight; Run's public calls are serial.
  void release_models();
  [[nodiscard]] gpu::MemoryFootprint retained_model_bytes() const;
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
