/*
 * Steering records, R4: retain bounded immutable comparisons alongside the
 * original target and variants. Resolve source by VariantId and capture key;
 * preserve exact description bytes, AR IDs, prefix extent and matched settings.
 * Each branch has its own terminal result and absolute generated segments;
 * failed/cancelled trials may retain one completed branch. Validate byte/count
 * budgets and provenance before append, leaving records unchanged on rejection.
 * replay_prefix returns max(original prompt length,p+1) without retokenization
 * and rejects a stop ID in the fixed generated prefix. O(prefix) checks; no
 * GPU work or copying the original captures. Overlay: steering.md, Exact
 replay.
 * Header consultation: Core I.1 (#ri-explicit), R.1 (#rr-raii), explicit
 * inputs and lifetimes; performance GPU.1, resident intermediates, GPU.9,
 * bounded persistent owners (guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md).
 * Both corpora queried/read, 2026-10-20.

 * Axis R4: immutable target capture index and bounded interpretation history.
 * Contract: a record owns target IDs, decoded byte mapping, classifications,
 * capture leases and provenance. It admits bounded variants and validates every
 * count/extent/site before publishing. Capture lookup returns only a completed
 * consumed-token row. Device must outlive device leases, including borrowed
 * row copies held beyond the record. Record contains no execution queue.
 *
 * Algorithm: logical-overview.md retain/interpret; execution-boundary.md H1/H4.
 * Validate generation provenance and emitted/consumed counts; graceful partial
 * prefill may leave a prompt suffix but never a pending generated token.
 * Validate each site and payload, index (site, absolute consumed position) to
 * its row offset, sort and reject duplicates. Host reference lookup returns a
 * span; live lookup returns an immutable subregion sharing its device owner.
 * Append a variant only after validating its source key, settings, normalized
 * injection provenance when available, generation outcome and text mapping.
 * Failed GPU normalization may have no valid injection provenance. Raw target
 * data and previous variants remain unchanged by rejected appends.
 * Cost: O(captured positions) index, O(log positions) lookup, bounded variant
 * storage. retained_bytes accounts host payload/capacity; Run separately admits
 * the retained device arenas when loading AV weights. Optimization: no live
 * activation copies or CPU normalization during lookup/model switching.
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

#include <memory>

#include "inference/generation.h"
#include "measurement.h"
#include "nla/protocol.h"

namespace dolittle::experiment {
struct RunId {
  std::uint64_t value;
};
struct VariantId {
  std::uint64_t value;
};
struct TrialId {
  std::uint64_t value;
};
struct CaptureKey {
  models::ObservationPoint site;
  models::TokenPosition position;
};
struct RecordLimits {
  std::size_t max_retained_bytes{};
  std::size_t max_variants{};
  std::size_t max_trials = 4;
};
struct TargetRecord {
  RunId run;
  nla::PairIdentity pair;
  models::ObservationSite capture_site{};
  std::string formatted_prompt;
  inference::GenerationResult generation;
  tokenizers::DecodedBytes
      decoded;  // All target IDs, including prompt/control.
  std::vector<text::TokenSegment> segments;  // Absolute generated positions.
  std::string backend_identity;
};
struct InjectionProvenance {
  models::TokenPosition prompt_position;
  float source_l2_norm;
  float scale;
  tensors::ElementType storage;
  std::string conversion_identity;
};
enum class AttemptOutcome : std::uint8_t { completed, cancelled, failed };
enum class AttemptStage : std::uint8_t {
  conversion,
  loading,
  preparation,
  generation,
  decoding,
  complete
};
struct Interpretation {
  CaptureKey capture;
  inference::GenerationSettings requested_settings;
  std::optional<InjectionProvenance> injection;
  std::optional<inference::GenerationResult> generation;
  tokenizers::DecodedBytes
      decoded;  // Generated AV IDs only, raw control bytes retained.
  nla::ExplanationRange explanation;
  AttemptOutcome outcome{AttemptOutcome::failed};
  AttemptStage stage{AttemptStage::conversion};
  std::string failure_message;
  measurement::InterpretationCost cost;
};
struct Variant {
  VariantId id{};
  Interpretation interpretation;
};
struct SteeringEdit {
  VariantId variant;
  std::string explanation;
  float strength;
  inference::GenerationSettings settings;
};
struct BranchResult {
  std::optional<inference::GenerationResult> generation;
  tokenizers::DecodedBytes decoded;
  std::vector<text::TokenSegment> segments;
};
struct SteeringTrial {
  VariantId variant;
  CaptureKey capture;
  std::string original, edited;
  std::vector<tokenizers::TokenId> original_ar_tokens, edited_ar_tokens;
  float strength{};
  inference::GenerationSettings settings{};
  std::size_t prefix_tokens{}, original_prompt_tokens{};
  std::optional<tensors::formats::SourceIdentity> reconstructor;
  std::string numerical_identity =
      "reconstruction-difference/fp32-eight-lane-v1";
  BranchResult baseline, steered;
  AttemptOutcome outcome = AttemptOutcome::failed;
  std::string message;
};
struct Trial {
  TrialId id{};
  SteeringTrial steering;
};
// Pure replay rule. Keeps the entire original prompt or generated prefix
// through the selected position, rejects stop-token branches and stale
// captures.
[[nodiscard]] std::size_t replay_prefix(const TargetRecord&, CaptureKey);
class Record {
 public:
  Record(TargetRecord, RecordLimits);
  ~Record();
  Record(const Record&) = delete;
  Record& operator=(const Record&) = delete;
  Record(Record&&) = delete;
  Record& operator=(Record&&) = delete;
  [[nodiscard]] const TargetRecord& target() const noexcept;
  [[nodiscard]] std::optional<nla::CaptureRow> capture(CaptureKey) const;
  [[nodiscard]] std::size_t variant_count() const noexcept;
  [[nodiscard]] const Variant& variant(VariantId) const;
  [[nodiscard]] std::size_t retained_bytes() const noexcept;
  [[nodiscard]] VariantId append(Interpretation);
  [[nodiscard]] std::size_t trial_count() const noexcept;
  [[nodiscard]] const Trial& trial(TrialId) const;
  [[nodiscard]] TrialId append(SteeringTrial);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::experiment
