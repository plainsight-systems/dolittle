/*
 * Target-to-verbalizer conversion and pairing contract.
 * Axis M7: trained target/AV compatibility, activation conversion and output
 * interpretation. Concrete pairs own conventions; execution owns no NLA names.
 * Contract: a Pair owns immutable, verified provenance and a prepared AV
 * prompt. Its source identities come from verified archives, not model-name
 * guesses. validate_model checks the role's source and mathematical
 * capabilities before upload. prepare_injection accepts exactly one completed
 * raw capture row from the declared site. No transcript text or neighboring
 * rows enter the AV prompt. Results own their bytes; inputs are borrowed for a
 * call. Invalid site, source, shape, dtype, nonfinite values or unsupported
 * semantics -> invalid_argument; checked limits -> length_error. Allocation
 * failure propagates without mutation.
 *
 * Algorithm: logical-overview.md, "4 Interpret selected positions", and
 * transformer-execution.md, "NLA entry and exit paths" (both under
 * docs/architecture/). Let h have D elements and p be its target-token
 * position.
 * 1. Validate h's site, width, storage and provenance against the selected
 * pair. Decode the original storage to FP32 without modifying the retained
 * bytes.
 * 2. The concrete pair computes its trained normalization/scale and casts once
 *    into the AV embedding storage. Return an owned replacement row and the
 *    normalization metadata, with the AV prompt position separate from p.
 * 3. The executor looks up the exact prepared AV IDs, replaces only that row,
 *    and generates with a fresh sequence/cache. It never treats h as token IDs.
 * 4. Decode generated IDs losslessly; locate the pair's explanation delimiters.
 *    Return a byte range only when the delimiter contract is satisfied.
 * Preserve complete raw output otherwise. Delimiters are formatting, not
 * confidence. Generation stop/failure status is independent of delimiter
 * status.
 *
 * Cost: O(D) conversion and temporary storage, O(output bytes) extraction.
 * Optimization: prepare the fixed prompt once per pair and upload only one
 * D-element replacement per interpretation; no host copy of all AV embeddings.
 * Numerical reduction and rounding belong to the concrete pair, not this API.
 * Corpus consultation, header design and implementation, 2026-10-19 (full rules
 * read): cpp-guidelines I.7 (#ri-post): explicit owned replacement and
 * raw-output rules. cpp-guidelines R.1 (#rr-raii): pair and conversion payloads
 * own their resources. cpp-perf-guidelines GPU.1: count the single-row
 * transfer, keep embeddings on GPU. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md. Normalization implementation:
 * accumulate squared FP32 values into eight interleaved lanes (i modulo 8),
 * reduce lanes left-to-right, then take FP32 sqrt. This matches the frozen CPU
 * norm on the tested vectors. Multiplication and addition are separately
 * rounded (-ffp-contract=off); no FP64 intermediate. One D-element FP32 buffer,
 * eight lane sums and one D-element output are owned.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "inference/executor.h"
#include "models/checkpoint.h"
#include "tensors/formats/format.h"
#include "text/protocol.h"

namespace dolittle::nla {
enum class ModelRole : std::uint8_t { target, verbalizer };
struct PairIdentity {
  std::string
      convention;  // Versioned normalization/template/extraction policy.
  tensors::formats::SourceIdentity target;
  tensors::formats::SourceIdentity verbalizer;
  std::string metadata_sha256;
};
struct CaptureRow {
  models::ObservationSite site;
  models::TokenPosition position;  // Target consumed-token position.
  std::span<const std::byte> bytes;
};
struct Injection {
  inference::EmbeddingOverride embedding;
  float source_l2_norm{};
  float scale{};
};
enum class ExplanationFormat : std::uint8_t { complete, missing, malformed };
struct ByteRange {
  std::size_t begin;
  std::size_t end;  // Exclusive, into the original decoded byte string.
};
struct ExplanationRange {
  ExplanationFormat format{ExplanationFormat::missing};
  std::optional<ByteRange> content;
};
// Pure FP32 L2 conversion used by pairs with this trained normalization law.
// Reject empty/malformed/nonfinite rows, unsupported storage and invalid scale.
[[nodiscard]] Injection normalize_activation(
    CaptureRow, models::TokenPosition injection_position, float scale,
    tensors::ElementType output_storage);
[[nodiscard]] ExplanationRange explanation_tags(std::string_view raw_bytes);
class Pair {
 public:
  Pair() = default;
  virtual ~Pair() = default;
  Pair(const Pair&) = delete;
  Pair& operator=(const Pair&) = delete;
  Pair(Pair&&) = delete;
  Pair& operator=(Pair&&) = delete;
  [[nodiscard]] virtual const PairIdentity& identity() const noexcept = 0;
  [[nodiscard]] virtual models::ObservationSite target_site()
      const noexcept = 0;
  [[nodiscard]] virtual const text::PreparedPrompt& verbalizer_prompt()
      const noexcept = 0;
  virtual void validate_model(ModelRole,
                              const tensors::formats::SourceIdentity&,
                              const models::BoundCheckpoint&,
                              tensors::ComputationMode) const = 0;
  [[nodiscard]] virtual Injection prepare_injection(CaptureRow) const = 0;
  [[nodiscard]] virtual ExplanationRange explanation(
      std::string_view raw_bytes) const = 0;
};
}  // namespace dolittle::nla
