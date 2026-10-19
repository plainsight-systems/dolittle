/*
 * Target-generation interface and implementation contract.
 * Axis M4: byte-level BPE driven by verified checkpoint tokenizer data.
 * Inputs: tokenizer.json/config bytes, verified source identity, and explicit
 * metadata/input/output limits. Output: an immutable Tokenizer owning parsed
 * vocabulary, ranked merges, added-token matcher and Unicode pipeline metadata.
 * Reject duplicate/invalid IDs, inconsistent added tokens, missing byte
 * alphabet, unknown pipeline stages, dropout or unsupported options. No
 * approximate fallback.
 *
 * Algorithm for the pinned Qwen3 pipeline:
 * 1. Match non-normalized added tokens before normalization, retaining their
 *    declared IDs. Match longest content first, then declared ID for a tie.
 *    The selected data has no whitespace stripping or word-boundary flags.
 * 2. Apply NFC to each intervening ordinary-text span. Split it with the exact
 *    ordered Unicode regex from tokenizer.json, preserving every byte. Its
 *    number branch matches one digit at a time, not an arbitrary number run.
 * 3. Map each UTF-8 byte to the declared ByteLevel alphabet. Each mapped byte
 *    begins as one symbol; do not add a prefix space.
 * 4. Find adjacent symbol pairs with a merge rank. Merge the lowest-rank pair
 *    first (leftmost on a tie), update its neighboring candidates, and repeat
 *    until no ranked pair remains. Never merge across pre-token boundaries.
 * 5. Resolve remaining symbols to vocabulary IDs and interleave added-token
 *    IDs in their original order. No BOS/EOS or padding is inserted.
 * Decode inverts the byte alphabet and preserves added-token spellings as
 * specified by tokenizer.h. NFC means byte-for-byte input round trips are not
 * promised. The exact pinned regex and Unicode behavior are compatibility
 * inputs, not an ASCII approximation or std::regex assumption.
 *
 * Implementation design: parse once into byte/token lookup tables and ranked
 * pair keys. Use adjacent-symbol links and a priority queue with stale-entry
 * checks; do not rebuild every concatenation after each merge. For B bytes,
 * merge scheduling is O(B log B) work/O(B) scratch, apart from Unicode
 * matching. Optimization: precompute merged token IDs and reuse immutable
 * tables. Unicode primitives use utf8proc 2.10.0 (NFC) and Oniguruma 6.9.10
 * (regex), pinned by source hash in CMake. BPE, added-token matching and byte
 * decoding are implemented here; no third-party tokenizer or inference runtime
 * is used. Added tokens are a small immutable list scanned for the earliest
 * match, longest first at that position. Matching costs O(added tokens * input
 * bytes). Limits bound JSON, text, piece length and emitted IDs before
 * allocation grows. Invalid metadata/input -> invalid_argument; limits/overflow
 * -> length_error; allocation failure propagates. Input bytes are borrowed only
 * for construction.
 *
 * Static-analysis design/implementation, 2026-10-19: named source records
 * distinguish metadata roles without copying bytes. Core I.1 (#ri-explicit);
 * performance CDSA.32 (source cited below): parse borrowed bytes once into
 * owned lookup tables; no view is retained. Reference: Qwen/Qwen3-8B at
 * b968826d9c46dd6066d109eabc6255188de91218, tokenizer.json and
 * tokenizer_config.json; hashes in scripts/validation/models-8b.json. Overlay:
 * docs/architecture/file-mapping.md, "Model data, tensor storage and
 * tokenization"; logical-overview.md, "1 Establish the run".
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): reject unsupported
 * pipelines; make normalization, matching and insertion behavior part of the
 * contract. cpp-perf-guidelines CDSA.32: compile vocabulary/merges into lookup
 * tables once. Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include <memory>

#include "tokenizers/tokenizer.h"

namespace dolittle::tokenizers {
struct ByteBpeLimits {
  std::size_t max_metadata_bytes;
  std::size_t max_input_bytes;
  std::size_t max_piece_bytes;
  std::size_t max_output_tokens;
};
// Named borrowed inputs are valid only for the duration of the factory call.
struct ByteBpeSources {
  std::string_view tokenizer_json;
  std::string_view tokenizer_config_json;
  std::string_view verified_source_identity;
};
[[nodiscard]] std::unique_ptr<Tokenizer> make_byte_bpe(ByteBpeSources sources,
                                                       ByteBpeLimits limits);
}  // namespace dolittle::tokenizers
