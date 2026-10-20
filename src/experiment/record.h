/*
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
struct CaptureKey {
  models::ObservationPoint site;
  models::TokenPosition position;
};
struct RecordLimits {
  std::size_t max_retained_bytes;
  std::size_t max_variants;
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

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::experiment
