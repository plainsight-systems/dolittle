/*
 * Target-generation interface and implementation contract.
 * Axis M4: token identity, text encoding and lossless token-byte decoding.
 * Inputs: valid UTF-8 text for encode, or token IDs for decode.
 * Outputs: owned IDs, or owned raw bytes with a byte range for every input ID.
 * The implementation owns its immutable vocabulary and pipeline configuration.
 * Identity includes verified artifact hashes and the tokenizer implementation
 * version. No caller spans are retained. No automatic BOS/EOS, padding,
 * truncation, cleanup or special-token removal is implied by this interface.
 *
 * Algorithm contract:
 * 1. Encode applies the selected tokenizer's declared added-token matching,
 *    normalization, splitting and token construction pipeline in that order.
 * 2. Decode resolves each ID to its byte contribution and appends it in input
 *    order, recording its [begin,end) byte range. Added tokens retain their
 *    literal spelling. A model-vocabulary ID absent from the tokenizer maps
 *    to an empty range marked unmapped; it must not disappear from provenance.
 * 3. Return raw bytes even when individual tokens split a UTF-8 character.
 *    Presentation performs UTF-8 replacement separately, retaining token
 * ranges. encode/decode need not recover the original input bytes:
 * normalization may change them. Encode rejects malformed UTF-8; decode does
 * not invent a word for an unmapped ID. Allocation/size failures propagate;
 * inputs are unchanged.
 *
 * Design and cost: decode is linear in token count plus produced bytes.
 * Optimization: build ID-to-byte and token lookup tables once, share them
 * immutably, and do not reparse checkpoint JSON per prompt or generated token.
 * Concrete kinds declare their own encode cost and supported pipeline.
 * Overlay: docs/architecture/logical-overview.md, "1 Establish the run" and
 * "2 Generate and capture"; file-mapping.md, "Model data, tensor storage
 * and tokenization".
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): no hidden token insertion
 * or lossy decoding; ownership and unmapped-ID behavior are explicit.
 * cpp-perf-guidelines CDSA.32: prepare immutable encode/decode lookup tables.
 * Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 * Steady-state design, 2026-10-20:
 * decode_into fills caller-owned byte/token buffers; max_token_bytes bounds
 * reservation before the stream. Borrow immutable vocabulary data, preserve
 * unmapped IDs and exact raw ranges. Retain owned decode as a convenience
 * wrapper.
 * Overlay: transformer-execution.md, "Device selection and allocation-free
 * generation". Core I.7 (#ri-post): borrowing, bounds and failure contracts.
 * Performance GPU.1: device-resident decisions; MEM.9: prepare storage before
 * steady state and check the invariant. Sources:
 * guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/memory/MEM.9-embedded-allocate-at-init.md.
 * Both corpora queried/read for header design and implementation, 2026-10-20.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dolittle::tokenizers {
using TokenId = std::uint32_t;
struct TokenBytes {
  TokenId id;
  std::size_t begin;
  std::size_t end;
  bool mapped;
};
struct DecodedBytes {
  std::string bytes;
  std::vector<TokenBytes> tokens;
};
class Tokenizer {
 public:
  Tokenizer() = default;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&&) = delete;
  Tokenizer& operator=(Tokenizer&&) = delete;
  virtual ~Tokenizer() = default;
  [[nodiscard]] virtual std::string_view identity() const noexcept = 0;
  [[nodiscard]] virtual std::vector<TokenId> encode(
      std::string_view utf8) const = 0;
  [[nodiscard]] virtual DecodedBytes decode(
      std::span<const TokenId> ids) const = 0;
  // Caller reserves both capacities; insufficient capacity fails before writes.
  virtual void decode_into(std::span<const TokenId>, DecodedBytes&) const = 0;
  [[nodiscard]] virtual std::size_t max_token_bytes() const noexcept = 0;
  [[nodiscard]] virtual bool contains(TokenId id) const noexcept = 0;
};
}  // namespace dolittle::tokenizers
