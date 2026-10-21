/*
 * Reconstruction, M7: optional pinned AR identity and trained text-to-vector
 * convention. reconstruction_prompt encodes the complete description template
 * through the AR tokenizer, validates its suffix and returns owned exact IDs.
 * Unsupported pairs reject; no target chat formatting or generic prompt guess.
 * Pair validation checks role, schema and mode before uploading weights.
 * O(description bytes/tokens) host work only; AR arithmetic remains GPU work.
 * Overlay: steering.md, Paper method and our model pair.
 * Header consultation: Core I.1 (#ri-explicit), R.1 (#rr-raii), explicit
 * inputs and lifetimes; performance GPU.1, resident intermediates, GPU.9,
 * bounded persistent owners (guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md).
 * Both corpora queried/read, 2026-10-20.

 * Axis M7: pairing, activation-injection descriptors and explanation
 extraction.
 * Contract: a Pair owns model/source identities and its exact trained AV
 prompt.
 * It validates the advertised target site and checkpoint convention. A live
 * CaptureRow carries one completed device lease and no host span; a diagnostic
 * row carries explicit host bytes. Device and source owner outlive injection.
 * An Injection describes the marker, output storage, source type and trained
 * scale. Its source norm is populated after successful GPU normalization.
 *
 * Algorithm: execution-boundary.md H2/H3 and logical-overview.md
 interpretation.
 * The pair validates provenance and selects the trained marker. For a live row
 * it returns a DeviceNormalization descriptor; the executor widens source
 * values on GPU, computes FP32 L2, divides, scales and casts once, then gates
 * inference with device status and replaces the marker embedding before block0.
 * normalize_activation is the explicit host reference helper: eight interleaved
 * FP32 square-sum lanes, left-to-right lane reduction, FP32 sqrt, divide then
 * multiply, one final cast. It rejects malformed/nonfinite/zero-norm inputs.
 * explanation_tags scans decoded bytes for exactly one ordered nonnested tag
 * pair, trims ASCII whitespace and returns a borrowed byte range. All raw text
 * remains available even when tags are missing/malformed.
 * Cost: live preparation O(1) descriptor copying; numerical conversion O(D)
 * on device. Host reference conversion owns O(D) FP32/output temporaries.
 * Optimization: no live host vector materialization or per-token conversion.
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
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "inference/executor.h"
#include "models/checkpoint.h"
#include "tensors/formats/format.h"
#include "text/protocol.h"

namespace dolittle::nla {
inline constexpr std::size_t max_explanation_bytes = 32 * std::size_t{1024};
inline constexpr std::size_t max_reconstruction_tokens = 4096;
enum class ModelRole : std::uint8_t { target, verbalizer, reconstructor };
struct PairIdentity {
  std::string
      convention;  // Versioned normalization/template/extraction policy.
  tensors::formats::SourceIdentity target;
  tensors::formats::SourceIdentity verbalizer;
  std::string metadata_sha256;
  std::optional<tensors::formats::SourceIdentity> reconstructor{};
  std::string reconstruction_metadata_sha256{};
};
struct CaptureRow {
  models::ObservationSite site;
  models::TokenPosition position;    // Target consumed-token position.
  std::span<const std::byte> bytes;  // Host-reference path only.
  std::optional<inference::DeviceCapture> device{};
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
// Reject empty/malformed/nonfinite rows, unsupported storage and invalid
// scale.
[[nodiscard]] Injection normalize_activation(
    const CaptureRow&, models::TokenPosition injection_position, float scale,
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
  [[nodiscard]] virtual std::vector<tokenizers::TokenId> reconstruction_prompt(
      std::string_view explanation, const tokenizers::Tokenizer&) const;
  [[nodiscard]] virtual ExplanationRange explanation(
      std::string_view raw_bytes) const = 0;
};
}  // namespace dolittle::nla
