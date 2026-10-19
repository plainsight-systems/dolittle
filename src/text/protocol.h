/*
 * Target-generation interface and implementation contract.
 * Axis M5: prompt construction, stop IDs and thinking/answer segmentation.
 * Input: optional system text, one user prompt and an explicit thinking
 * setting. Output: exact formatted bytes, encoded IDs, stop IDs and
 * protocol/tokenizer identities. This first experiment accepts a fresh text
 * prompt; conversation history, tools and multimodal content need explicit
 * future protocol support.
 *
 * Algorithm contract:
 * 1. Format the supported messages according to the selected protocol.
 * 2. Encode the complete formatted text through the supplied tokenizer; retain
 *    control IDs and bytes exactly, rather than independently encoding
 * fragments and assuming concatenation preserves token boundaries.
 * 3. Return stop IDs as protocol data. Generation handles stopping and final
 *    capture completion; text segmentation never changes generated IDs.
 * 4. Classify each generated ID as unclassified, reasoning, answer or control.
 *    Preserve its absolute position in the complete prompt-plus-generated
 * trace. A missing or malformed thinking delimiter produces an unclassified
 * span, not an inferred reasoning narrative. Mode is an input convention, not a
 * guarantee of the generated text's semantics. Empty system and absent system
 * differ. Inputs are borrowed only for calls; results own their data. Invalid
 * UTF-8, unsupported input or tokenizer/protocol mismatch -> invalid_argument;
 * checked limits -> length_error; allocation failure propagates.
 *
 * Design and cost: O(text bytes + token count), in addition to encoding.
 * Optimization: classify IDs in one pass and reuse the immutable tokenizer;
 * no repeated full-history decoding per emitted token. Presentation may buffer
 * incomplete UTF-8 bytes while keeping the contributing token positions.
 * Overlay: docs/architecture/logical-overview.md, "1 Establish the run",
 * "2 Generate and capture" and "Capture alignment".
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): separate formatting and
 * stop metadata from progression, and retain exact token provenance.
 * cpp-perf-guidelines CDSA.32: resolve protocol token identities once.
 * Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include <cstdint>
#include <optional>

#include "tokenizers/tokenizer.h"

namespace dolittle::text {
enum class ThinkingMode : std::uint8_t { enabled, disabled };
struct Prompt {
  std::optional<std::string> system;
  std::string user;
  ThinkingMode thinking{};
};
struct PreparedPrompt {
  std::string protocol_identity;
  std::string tokenizer_identity;
  std::string formatted_utf8;
  std::vector<tokenizers::TokenId> ids;
  std::vector<tokenizers::TokenId> stop_ids;
  ThinkingMode thinking{};
};
enum class SegmentKind : std::uint8_t {
  unclassified,
  reasoning,
  answer,
  control
};
struct TokenSegment {
  std::size_t position;
  SegmentKind kind;
};
class Protocol {
 public:
  Protocol() = default;
  Protocol(const Protocol&) = delete;
  Protocol& operator=(const Protocol&) = delete;
  Protocol(Protocol&&) = delete;
  Protocol& operator=(Protocol&&) = delete;
  virtual ~Protocol() = default;
  [[nodiscard]] virtual PreparedPrompt prepare(
      const Prompt&, const tokenizers::Tokenizer&) const = 0;
  [[nodiscard]] virtual std::vector<TokenSegment> segment(
      std::span<const tokenizers::TokenId> generated,
      std::size_t first_position, ThinkingMode mode) const = 0;
};
}  // namespace dolittle::text
