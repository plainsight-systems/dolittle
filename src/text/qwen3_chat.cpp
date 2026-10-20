/*
 * Pinned Qwen3 single-turn text protocol implementation.
 * Axis M5: the selected Qwen3 target's single-turn chat/thinking convention.
 * Inputs: verified tokenizer_config.json and generation_config.json bytes,
 * source identity and a byte budget. Output: an owned immutable Protocol.
 * Validate the supported template branch, added-token IDs and stop settings
 * against the pinned tokenizer. Never execute arbitrary Jinja from a
 * checkpoint. Reject unsupported template/config changes instead of guessing
 * equivalence.
 *
 * Algorithm for prompt formatting:
 * 1. If system text is present, append exactly
 *    "<|im_start|>system\n" + system + "<|im_end|>\n".
 * 2. Append "<|im_start|>user\n" + user + "<|im_end|>\n".
 * 3. Append "<|im_start|>assistant\n".
 * 4. With thinking disabled, append "<think>\n\n</think>\n\n".
 *    With thinking enabled, append nothing more: the pinned template does NOT
 *    force an opening <think> into the prompt.
 * 5. Encode the entire string with no additional BOS/EOS. Set stop IDs from
 *    generation_config.json: 151645 (<|im_end|>) and 151643 (<|endoftext|>).
 * No trimming or escaping of message content is invented. Literal added-token
 * spellings in content follow the verified tokenizer's matching behavior.
 *
 * Segmentation: in enabled mode start unclassified; opening ID 151667 starts
 * reasoning, closing ID 151668 starts answer. In disabled mode start answer
 * because the prompt contains an empty thinking section. Delimiters and stop
 * IDs are control. Nested/repeated malformed delimiters reset to unclassified
 * until a valid opening/closing transition; do not discard any ID. Decode text
 * separately. Both thinking IDs are added tokens marked special=false, so
 * skip_special_tokens is not a valid segmentation algorithm.
 *
 * Design and cost: bounded linear formatting and token classification.
 * Optimization: validate template/control metadata once at construction.
 * Inputs are borrowed only during construction/calls; the protocol owns parsed
 * metadata. Errors and lifetimes follow protocol.h; limits include formatted
 * bytes and prevent size overflow before concatenation.
 * Reference: Qwen/Qwen3-8B at b968826d9c46dd6066d109eabc6255188de91218;
 * artifact hashes are in scripts/validation/models-8b.json.
 * Overlay: docs/architecture/logical-overview.md, "1 Establish the run".
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): exact template bytes and
 * control-token transitions, independent of model arithmetic.
 * cpp-perf-guidelines CDSA.32: resolve fixed template/control data once.
 * Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 *
 * Static-analysis design/implementation, 2026-10-19:
 * Named borrowed source fields distinguish the two JSON documents and identity.
 * Parsed metadata is owned; token-span access is checked without copying it.
 * cpp-guidelines SL.con.3 (#rsl-bounds), I.30 (#ri-encapsulate): make checked
 * host access explicit and confine necessary native API operations.
 * cpp-perf-guidelines COPY.7: borrow spans/references without hidden payload
 * copies. Source: guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 *
 * Incremental classification (implementation corpus queried 2026-10-19): a
 * cursor starts at the first generated absolute position. Require the next
 * contiguous span, check position overflow, reserve its output, then apply the
 * same classify transition to every ID and advance the absolute watermark.
 * Reserve before mutation so allocation failure permits retry. Batch segment
 * uses this same cursor. Work is O(new IDs), state O(1); no text prefix
 * rescans. CP.22 (#rconc-unknown), independent cursor state with no external
 * callbacks; COPY.7, borrow ID spans
 * (guidelines/copy-move/COPY.7-eliminate-hidden-copies.md).
 * Steady-state design, 2026-10-20:
 * append_into validates caller capacity before clearing output or mutating
 * state, classifies each contiguous ID once and commits the new cursor.
 * Prepared streaming capacity prevents allocation in successful steady state.
 * Overlay: transformer-execution.md, "Device selection and allocation-free
 * generation". Core I.7 (#ri-post): borrowing, bounds and failure contracts.
 * Performance GPU.1: device-resident decisions; MEM.9: prepare storage before
 * steady state and check the invariant. Sources:
 * guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/memory/MEM.9-embedded-allocate-at-init.md.
 * Both corpora queried/read for header design and implementation, 2026-10-20.
 */
#include "text/qwen3_chat.h"

#include <openssl/sha.h>

#include <array>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "bounds.h"
#include "tensors/formats/json_metadata.h"

namespace dolittle::text {
namespace {
inline constexpr std::size_t kibibyte = 1024;
inline constexpr std::uint32_t end_message = 151645;
inline constexpr std::uint32_t end_text = 151643;
inline constexpr std::uint32_t think_begin = 151667;
inline constexpr std::uint32_t think_end = 151668;

using Json = nlohmann::json;
Json parse(std::string_view text) {
  return tensors::formats::detail::parse_metadata(
             std::as_bytes(std::span(text.data(), text.size())),
             kibibyte * kibibyte)
      .value;
}
std::string sha256(std::string_view text) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  // Native API byte/address boundary; no unrelated typed object is
  // dereferenced. NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  SHA256(reinterpret_cast<const unsigned char*>(text.data()), text.size(),
         digest.data());
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (auto b : digest) {
    out << std::setw(2) << static_cast<unsigned>(b);
  }
  return out.str();
}
SegmentKind classify(SegmentKind& state, tokenizers::TokenId id) {
  auto kind = state;
  if (id == think_begin) {
    state = state == SegmentKind::unclassified ? SegmentKind::reasoning
                                               : SegmentKind::unclassified;
    kind = SegmentKind::control;
  } else if (id == think_end) {
    state = state == SegmentKind::reasoning ? SegmentKind::answer
                                            : SegmentKind::unclassified;
    kind = SegmentKind::control;
  } else if (id == end_message || id == end_text) {
    kind = SegmentKind::control;
  }
  return kind;
}
struct Qwen3Cursor final : SegmentCursor {
  std::size_t next;
  SegmentKind state;
  Qwen3Cursor(std::size_t start, ThinkingMode mode)
      : next(start),
        state(mode == ThinkingMode::enabled ? SegmentKind::unclassified
                                            : SegmentKind::answer) {
    if (mode != ThinkingMode::enabled && mode != ThinkingMode::disabled) {
      throw std::invalid_argument("unknown thinking mode");
    }
  }
  std::vector<TokenSegment> append(std::span<const tokenizers::TokenId> ids,
                                   std::size_t first) override {
    std::vector<TokenSegment> result;
    result.reserve(ids.size());
    append_into(ids, first, result);
    return result;
  }
  void append_into(std::span<const tokenizers::TokenId> ids, std::size_t first,
                   std::vector<TokenSegment>& result) override {
    if (first != next ||
        ids.size() > std::numeric_limits<std::size_t>::max() - next) {
      throw std::invalid_argument("noncontiguous segment input");
    }
    if (ids.size() > result.capacity()) {
      throw std::length_error("segment destination capacity");
    }
    result.clear();
    auto after = state;
    for (auto id : ids) {
      result.push_back({first + result.size(), classify(after, id)});
    }
    next += ids.size();
    state = after;
  }
};
struct Qwen3Chat final : Protocol {
  std::unique_ptr<SegmentCursor> make_cursor(std::size_t first,
                                             ThinkingMode mode) const override {
    return std::make_unique<Qwen3Cursor>(first, mode);
  }
  std::string source;
  std::size_t budget{};
  PreparedPrompt prepare(
      const Prompt& prompt,
      const tokenizers::Tokenizer& tokenizer) const override {
    if (prompt.thinking != ThinkingMode::enabled &&
        prompt.thinking != ThinkingMode::disabled) {
      throw std::invalid_argument("unknown thinking mode");
    }
    if (!tokenizer.identity().starts_with(source + "/")) {
      throw std::invalid_argument("tokenizer/protocol identity mismatch");
    }
    for (const auto& [text, id] :
         std::array<std::pair<std::string_view, std::uint32_t>, 5>{
             {{"<|endoftext|>", end_text},
              {"<|im_start|>", 151644},
              {"<|im_end|>", end_message},
              {"<think>", think_begin},
              {"</think>", think_end}}}) {
      const auto ids = tokenizer.encode(text);
      if (ids.size() != 1 || ids.at(0) != id) {
        throw std::invalid_argument("Qwen3 control token mismatch");
      }
    }
    PreparedPrompt result;
    result.protocol_identity = source + "/qwen3-single-turn-v1";
    result.tokenizer_identity = tokenizer.identity();
    result.thinking = prompt.thinking;
    const auto append = [&](std::string_view s) {
      if (s.size() > budget - result.formatted_utf8.size()) {
        throw std::length_error("formatted prompt byte budget");
      }
      result.formatted_utf8 += s;
    };
    if (prompt.system) {
      append("<|im_start|>system\n");
      append(*prompt.system);
      append("<|im_end|>\n");
    }
    append("<|im_start|>user\n");
    append(prompt.user);
    append("<|im_end|>\n<|im_start|>assistant\n");
    if (prompt.thinking == ThinkingMode::disabled) {
      append("<think>\n\n</think>\n\n");
    }
    result.ids = tokenizer.encode(result.formatted_utf8);
    result.stop_ids = {end_message, end_text};
    return result;
  }
  std::vector<TokenSegment> segment(std::span<const tokenizers::TokenId> ids,
                                    std::size_t start,
                                    ThinkingMode mode) const override {
    Qwen3Cursor cursor(start, mode);
    return cursor.append(ids, start);
  }
};
}  // namespace
std::unique_ptr<Protocol> make_qwen3_chat(Qwen3ChatSources sources,
                                          std::size_t budget) {
  const auto [config, generation, source] = sources;
  if (source.empty() || !budget) {
    throw std::invalid_argument("invalid protocol identity/budget");
  }
  const auto cfg = parse(config), gen = parse(generation);
  try {
    if (sha256(cfg.at("chat_template").get<std::string>()) !=
            "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d"
            "8" ||
        gen.at("eos_token_id") != Json::array({end_message, end_text})) {
      throw std::invalid_argument("unsupported Qwen3 template or stop IDs");
    }
    for (const auto& [id, text] :
         std::array<std::pair<std::string_view, std::string_view>, 5>{
             {{"151643", "<|endoftext|>"},
              {"151644", "<|im_start|>"},
              {"151645", "<|im_end|>"},
              {"151667", "<think>"},
              {"151668", "</think>"}}}) {
      if (cfg.at("added_tokens_decoder").at(id).at("content") != text) {
        throw std::invalid_argument("Qwen3 added-token mismatch");
      }
    }
    auto protocol = std::make_unique<Qwen3Chat>();
    protocol->source = source;
    protocol->budget = budget;
    return protocol;
  } catch (const Json::exception&) {
    throw std::invalid_argument("invalid Qwen3 protocol JSON schema");
  }
}
}  // namespace dolittle::text
