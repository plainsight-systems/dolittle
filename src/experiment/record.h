/*
 * Durable in-memory experiment record contract.
 * Axis R4: owned target transcript/captures and append-only interpretation
 * variants. Contract: one Record owns one completed or interrupted target
 * result, its exact formatted prompt, decoded byte mapping and segmentation,
 * plus bounded variants. RunId identifies the experiment; CaptureKey identifies
 * a consumed target token at an advertised site; VariantId identifies an
 * attempt within that run. None is an AV token position. No device, plan,
 * tokenizer or checkpoint lifetime is needed after construction. This is
 * in-memory retention, not persistence to disk.
 *
 * Algorithm: docs/architecture/logical-overview.md, "Capture alignment",
 * "3 Retain the experiment and change phases", "4 Interpret selected
 * positions":
 * 1. Move the generation's exact tokens, completed raw capture chunks and
 *    provenance into the record. Validate prompt/consumed/emitted lengths and
 *    capture row extents. Each (site,position) is unique, position < consumed,
 *    and the site's storage/width agrees with the pair-validated capture_site
 *    supplied by Run. Preserve original bytes. Record checks consistency; Pair
 *    and Run establish checkpoint/semantic compatibility before creating it.
 * 2. Index rows by site and absolute consumed position, retaining chunk owners;
 *    do not concatenate/copy the full capture payload. Absent rows remain
 * absent. An emitted but unconsumed token after failure has no fabricated
 * vector.
 * 3. Receive the byte mapping and segmentation already prepared by Run's target
 *    tokenizer/protocol. Validate ID correspondence, byte extents and absolute
 *    segment positions. Record does not tokenize or classify. Exact IDs remain
 *    authoritative, including byte splits, unmapped IDs and unclassified
 * segments.
 * 4. For each selected row, append a separately owned AV attempt with settings,
 *    seed, injection provenance, exact output IDs/bytes, stop/failure and tag
 * status. Repeated attempts never overwrite an earlier description or rerun the
 * target. Failed loading/conversion attempts can have no generated result.
 * Completed tags do not erase cancellation, token-limit or execution-failure
 * status. completed means the loop ended by stop ID or token cap, not that tags
 * were complete. A generation result must agree with outcome, requested
 * settings, injection provenance and decoded generated IDs; without one,
 * outcome cannot be completed and there can be no decoded output or explanation
 * range.
 * 5. Validate and admit the entire candidate before assigning its monotonic ID
 *    and publishing it. A rejected append leaves the record unchanged. A failed
 *    attempt that was admitted remains a record, not a successful
 * interpretation.
 *
 * Ownership/failure: views borrow this stable noncopyable/nonmovable owner.
 * Capture views survive variant insertion; variant references are stable until
 * Record destruction (individually owned entries). Callers serialize all
 * access. Invalid provenance/alignment -> invalid_argument; byte/count overflow
 * or budget exhaustion -> length_error; bad_alloc propagates with no partial
 * publication. Limits include owned vector/string capacities and index/entry
 * storage; allocator and process overhead are excluded explicitly, never
 * represented as a RAM cap.
 *
 * Cost: O(N + output bytes + R log R) construction for R indexed capture rows;
 * payload N*D*b for one site. Selected BF16 pair costs 8192*N capture bytes,
 * plus metadata, decoded text, pending candidate and generation staging
 * admitted by Run. Optimization: move payload owners and index their rows; no
 * whole-trace copy on model switch or variant append. Growth is bounded by
 * count and byte budgets. Corpus consultation, header design and
 * implementation, 2026-10-19 (full rules read): cpp-guidelines R.1 (#rr-raii):
 * records, not GPU plans, own retained payloads. cpp-guidelines I.7 (#ri-post):
 * atomic append and unavailable-row semantics. cpp-perf-guidelines GPU.1:
 * retain the existing completed readback; no replay or extra device transfer to
 * rediscover a selected capture. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
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
