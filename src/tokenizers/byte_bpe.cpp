/*
 * Byte-level BPE implementation with pinned Unicode support.
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
 * Reference: Qwen/Qwen3-8B at b968826d9c46dd6066d109eabc6255188de91218,
 * tokenizer.json and tokenizer_config.json; hashes in
 * scripts/validation/models-8b.json. Overlay:
 * docs/architecture/file-mapping.md, "Model data, tensor storage and
 * tokenization"; logical-overview.md, "1 Establish the run".
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): reject unsupported
 * pipelines; make normalization, matching and insertion behavior part of the
 * contract. cpp-perf-guidelines CDSA.32: compile vocabulary/merges into lookup
 * tables once. Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 *
 * Static-analysis design/implementation, 2026-10-19:
 * Named source views distinguish metadata roles. C APIs alone inspect unsigned
 * character views; checked match extents and table access prevent unchecked
 * host indexing. Parsed tables own data; factory views are not retained.
 * cpp-guidelines SL.con.3 (#rsl-bounds), I.30 (#ri-encapsulate): make checked
 * host access explicit and confine necessary native API operations.
 * cpp-perf-guidelines COPY.7: borrow spans/references without hidden payload
 * copies. Source: guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 * Steady-state design, 2026-10-20:
 * Precompute the longest decoded vocabulary entry. decode_into clears sizes
 * but retains caller capacity, appends immutable token bytes and ranges;
 * preparation reserves the maximum admitted delta. No vocabulary or
 * temporary per-token copies.
 * Overlay: transformer-execution.md, "Device selection and allocation-free
 * generation". Core I.7 (#ri-post): borrowing, bounds and failure contracts.
 * Performance GPU.1: device-resident decisions; MEM.9: prepare storage before
 * steady state and check the invariant. Sources:
 * guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/memory/MEM.9-embedded-allocate-at-init.md.
 * Both corpora queried/read for header design and implementation, 2026-10-20.
 */
#include "tokenizers/byte_bpe.h"

#include <oniguruma.h>
#include <utf8proc.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "tensors/formats/json_metadata.h"

namespace dolittle::tokenizers {
namespace {
inline constexpr unsigned token_id_bits = 32;
inline constexpr unsigned byte_alphabet_size = 256;
inline constexpr unsigned ascii_graphic_first = 33;
inline constexpr unsigned ascii_graphic_last = 126;
inline constexpr unsigned latin1_graphic_first = 161;
inline constexpr unsigned latin1_before_soft_hyphen = 172;
inline constexpr unsigned latin1_after_soft_hyphen = 174;

using Json = nlohmann::json;
constexpr std::string_view pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
void require(bool condition) {
  if (!condition) {
    throw std::invalid_argument("unsupported or invalid byte-BPE metadata");
  }
}
Json parse(std::string_view text, std::size_t budget) {
  return tensors::formats::detail::parse_metadata(
             std::as_bytes(std::span(text.data(), text.size())), budget)
      .value;
}
std::uint32_t id_value(const Json& j) {
  auto n = tensors::formats::detail::unsigned_integer(j);
  if (n > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("token ID overflow");
  }
  return static_cast<std::uint32_t>(n);
}
std::string utf8(std::int32_t c) {
  std::array<utf8proc_uint8_t, 4> b{};
  auto n = utf8proc_encode_char(c, b.data());
  if (n <= 0) {
    throw std::invalid_argument("invalid Unicode scalar");
  }
  // Native API byte/address boundary; no unrelated typed object is
  // dereferenced. NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return {reinterpret_cast<char*>(b.data()), static_cast<std::size_t>(n)};
}
void valid_utf8(std::string_view s) {
  if (s.size() >
      static_cast<std::size_t>(std::numeric_limits<utf8proc_ssize_t>::max())) {
    throw std::length_error("text too large");
  }
  for (std::size_t i = 0; i < s.size();) {
    utf8proc_int32_t c = 0;
    // UTF-8 C API reads the same character bytes as unsigned octets.
    auto n = utf8proc_iterate(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<const utf8proc_uint8_t*>(s.substr(i).data()),
        static_cast<utf8proc_ssize_t>(s.size() - i), &c);
    if (n <= 0) {
      throw std::invalid_argument("malformed UTF-8");
    }
    i += static_cast<std::size_t>(n);
  }
}
std::uint64_t pair_key(TokenId a, TokenId b) {
  return (std::uint64_t{a} << token_id_bits) | b;
}
struct RegexDelete {
  void operator()(OnigRegex p) const { onig_free(p); }
};
struct RegionDelete {
  void operator()(OnigRegion* p) const { onig_region_free(p, 1); }
};
struct ByteBpe final : Tokenizer {
  struct Merge {
    std::size_t rank;
    TokenId output;
  };
  struct Added {
    std::string text;
    TokenId id;
  };
  std::string identity_;
  ByteBpeLimits limits{};
  std::size_t longest_token{};
  std::size_t max_token_bytes() const noexcept override {
    return longest_token;
  }
  std::unordered_map<std::string, TokenId> vocabulary;
  std::unordered_map<TokenId, std::string> decoded;
  std::unordered_map<std::uint64_t, Merge> merges;
  std::array<TokenId, byte_alphabet_size> alphabet{};
  std::vector<Added> added;
  std::unique_ptr<OnigRegexType, RegexDelete> regex;
  std::string_view identity() const noexcept override { return identity_; }
  bool contains(TokenId id) const noexcept override {
    return decoded.contains(id);
  }
  DecodedBytes decode(std::span<const TokenId> ids) const override {
    const auto bytes = decoded_size(ids);
    DecodedBytes out;
    out.tokens.reserve(ids.size());
    out.bytes.reserve(bytes);
    decode_into(ids, out);
    return out;
  }
  void decode_into(std::span<const TokenId> ids,
                   DecodedBytes& out) const override {
    const auto bytes = decoded_size(ids);
    if (ids.size() > out.tokens.capacity() || bytes > out.bytes.capacity()) {
      throw std::length_error("decode destination capacity");
    }
    out.bytes.clear();
    out.tokens.clear();
    for (auto id : ids) {
      auto start = out.bytes.size();
      auto it = decoded.find(id);
      if (it != decoded.end()) {
        if (it->second.size() > limits.max_input_bytes - out.bytes.size()) {
          throw std::length_error("decode byte budget");
        }
        out.bytes += it->second;
      }
      out.tokens.push_back({id, start, out.bytes.size(), it != decoded.end()});
    }
  }
  std::size_t decoded_size(std::span<const TokenId> ids) const {
    if (ids.size() > limits.max_output_tokens) {
      throw std::length_error("decode token budget");
    }
    std::size_t bytes = 0;
    for (auto id : ids) {
      const auto it = decoded.find(id);
      if (it == decoded.end()) {
        continue;
      }
      if (it->second.size() > limits.max_input_bytes - bytes) {
        throw std::length_error("decode byte budget");
      }
      bytes += it->second.size();
    }
    return bytes;
  }
  void emit(std::vector<TokenId>& out, TokenId id) const {
    if (out.size() == limits.max_output_tokens) {
      throw std::length_error("encode token budget");
    }
    out.push_back(id);
  }
  void piece(std::string_view bytes, std::vector<TokenId>& out) const {
    if (bytes.size() > limits.max_piece_bytes) {
      throw std::length_error("BPE piece budget");
    }
    if (bytes.empty()) {
      return;
    }
    struct Node {
      TokenId id;
      std::size_t prev, next, version = 0;
      bool live = true;
    };
    struct Candidate {
      std::size_t rank, left, right, version;
      TokenId output;
    };
    auto later = [](const Candidate& a, const Candidate& b) {
      return a.rank != b.rank ? a.rank > b.rank : a.left > b.left;
    };
    std::vector<Node> nodes;
    nodes.reserve(bytes.size());
    const auto end = bytes.size();
    for (std::size_t i = 0; i < end; ++i) {
      nodes.push_back({alphabet.at(static_cast<unsigned char>(bytes.at(i))),
                       i ? i - 1 : end, i + 1});
    }
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(later)>
        heap(later);
    const auto offer = [&](std::size_t i) {
      if (i == end || !nodes.at(i).live || nodes.at(i).next == end) {
        return;
      }
      auto r = nodes.at(i).next;
      auto it = merges.find(pair_key(nodes.at(i).id, nodes.at(r).id));
      if (it != merges.end()) {
        heap.push(
            {it->second.rank, i, r, nodes.at(i).version, it->second.output});
      }
    };
    for (std::size_t i = 0; i < end; ++i) {
      offer(i);
    }
    while (!heap.empty()) {
      const auto c = heap.top();
      heap.pop();
      auto& l = nodes.at(c.left);
      auto& r = nodes.at(c.right);
      if (!l.live || !r.live || l.next != c.right || l.version != c.version) {
        continue;
      }
      // The right symbol may itself have merged since this entry was made.
      auto m = merges.find(pair_key(l.id, r.id));
      if (m == merges.end() || m->second.rank != c.rank ||
          m->second.output != c.output) {
        continue;
      }
      l.id = c.output;
      l.next = r.next;
      ++l.version;
      r.live = false;
      if (l.next != end) {
        nodes.at(l.next).prev = c.left;
      }
      offer(l.prev);
      offer(c.left);
    }
    for (std::size_t i = 0; i != end; i = nodes.at(i).next) {
      emit(out, nodes.at(i).id);
    }
  }
  void ordinary(std::string_view raw, std::vector<TokenId>& out) const {
    if (raw.empty()) {
      return;
    }
    utf8proc_uint8_t* result = nullptr;
    // UTF-8 C API reads bytes; no unrelated typed object is accessed.
    auto size = utf8proc_map(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<const utf8proc_uint8_t*>(raw.data()),
        static_cast<utf8proc_ssize_t>(raw.size()), &result,
        // utf8proc_option_t is a bitmask; the API accepts this documented
        // flag combination.
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        static_cast<utf8proc_option_t>(UTF8PROC_COMPOSE | UTF8PROC_STABLE));
    std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> owner(result,
                                                                  &std::free);
    if (size == UTF8PROC_ERROR_NOMEM) {
      throw std::bad_alloc();
    }
    if (size < 0) {
      throw std::invalid_argument("NFC normalization failed");
    }
    if (static_cast<std::size_t>(size) > limits.max_input_bytes) {
      throw std::length_error("normalized text budget");
    }
    // Native API byte/address boundary; no unrelated typed object is
    // dereferenced. NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const std::string_view normalized(reinterpret_cast<const char*>(result),
                                      static_cast<std::size_t>(size));
    std::unique_ptr<OnigRegion, RegionDelete> region(onig_region_new());
    if (!region) {
      throw std::bad_alloc();
    }
    // Native API byte/address boundary; no unrelated typed object is
    // dereferenced. NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto* begin = reinterpret_cast<const OnigUChar*>(normalized.data());
    const std::span bytes(begin, normalized.size());
    std::size_t offset = 0;
    while (offset < normalized.size()) {
      int rc = onig_search(
          regex.get(), begin, bytes.subspan(bytes.size()).data(),
          bytes.subspan(offset).data(), bytes.subspan(bytes.size()).data(),
          region.get(), ONIG_OPTION_NONE);
      if (rc < 0 || region->num_regs < 1 || !region->beg || !region->end) {
        throw std::invalid_argument("tokenizer regex returned no match bounds");
      }
      if (*region->beg < 0 ||
          static_cast<std::size_t>(*region->beg) != offset ||
          *region->end <= *region->beg) {
        throw std::invalid_argument("tokenizer regex failed to cover text");
      }
      auto end = static_cast<std::size_t>(*region->end);
      piece(normalized.substr(offset, end - offset), out);
      offset = end;
    }
  }
  std::vector<TokenId> encode(std::string_view input) const override {
    if (input.size() > limits.max_input_bytes) {
      throw std::length_error("encode byte budget");
    }
    valid_utf8(input);
    std::vector<TokenId> out;
    std::size_t offset = 0;
    while (offset < input.size()) {
      std::size_t next = input.size();
      const Added* selected = nullptr;
      for (const auto& a : added) {
        auto pos = input.find(a.text, offset);
        if (pos < next ||
            (pos == next && pos != std::string_view::npos &&
             (!selected || a.text.size() > selected->text.size() ||
              (a.text.size() == selected->text.size() &&
               a.id < selected->id)))) {
          next = pos;
          selected = &a;
        }
      }
      ordinary(input.substr(offset, next - offset), out);
      if (!selected) {
        break;
      }
      emit(out, selected->id);
      offset = next + selected->text.size();
    }
    return out;
  }
};
}  // namespace
std::unique_ptr<Tokenizer> make_byte_bpe(ByteBpeSources sources,
                                         ByteBpeLimits limits) {
  const auto [json, config, source] = sources;
  require(!source.empty() && limits.max_metadata_bytes &&
          limits.max_input_bytes && limits.max_piece_bytes &&
          limits.max_output_tokens);
  require(limits.max_input_bytes <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()));
  const auto data = parse(json, limits.max_metadata_bytes);
  const auto cfg = parse(config, limits.max_metadata_bytes);
  try {
    require(data.at("normalizer") == Json{{"type", "NFC"}} &&
            data.at("padding").is_null() && data.at("truncation").is_null());
    Json byte = {{"type", "ByteLevel"},
                 {"add_prefix_space", false},
                 {"trim_offsets", false},
                 {"use_regex", false}};
    Json split = {{"type", "Split"},
                  {"pattern", {{"Regex", pattern}}},
                  {"behavior", "Isolated"},
                  {"invert", false}};
    require(data.at("pre_tokenizer") ==
                Json{{"type", "Sequence"},
                     {"pretokenizers", Json::array({split, byte})}} &&
            data.at("decoder") == byte && data.at("post_processor") == byte);
    const auto& model = data.at("model");
    require(model.at("type") == "BPE" && model.at("dropout").is_null() &&
            model.at("unk_token").is_null() &&
            model.at("continuing_subword_prefix") == "" &&
            model.at("end_of_word_suffix") == "" &&
            model.at("fuse_unk") == false &&
            model.at("byte_fallback") == false &&
            model.at("ignore_merges") == false);
    require(cfg.at("add_bos_token") == false &&
            cfg.at("add_prefix_space") == false &&
            cfg.at("clean_up_tokenization_spaces") == false &&
            cfg.at("split_special_tokens") == false &&
            cfg.at("unk_token").is_null());
    auto bpe = std::make_unique<ByteBpe>();
    bpe->limits = limits;
    bpe->identity_ =
        std::string(source) + "/byte-bpe-v1/utf8proc-2.10.0/oniguruma-6.9.10";
    std::unordered_map<std::int32_t, unsigned char> inverse;
    std::array<std::int32_t, byte_alphabet_size> codes{};
    std::int32_t extra = byte_alphabet_size;
    for (unsigned i = 0; i < byte_alphabet_size; ++i) {
      codes.at(i) = (i >= ascii_graphic_first && i <= ascii_graphic_last) ||
                            (i >= latin1_graphic_first &&
                             i <= latin1_before_soft_hyphen) ||
                            (i >= latin1_after_soft_hyphen)
                        ? static_cast<std::int32_t>(i)
                        : extra++;
      inverse.emplace(codes.at(i), static_cast<unsigned char>(i));
    }
    for (const auto& [token, value] : model.at("vocab").items()) {
      auto id = id_value(value);
      require(!token.empty() && !bpe->decoded.contains(id));
      bpe->vocabulary.emplace(token, id);
      std::string bytes;
      for (std::size_t offset = 0; offset < token.size();) {
        utf8proc_int32_t c = 0;
        const auto remaining = std::string_view(token).substr(offset);
        // UTF-8 C API reads the same character bytes as unsigned octets.
        const auto* input =
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<const utf8proc_uint8_t*>(remaining.data());
        auto n = utf8proc_iterate(
            input, static_cast<utf8proc_ssize_t>(remaining.size()), &c);
        require(n > 0 && inverse.contains(c));
        bytes += static_cast<char>(inverse.at(c));
        offset += static_cast<std::size_t>(n);
      }
      bpe->decoded.emplace(id, std::move(bytes));
    }
    for (unsigned i = 0; i < byte_alphabet_size; ++i) {
      auto it = bpe->vocabulary.find(utf8(codes.at(i)));
      require(it != bpe->vocabulary.end());
      bpe->alphabet.at(i) = it->second;
    }
    std::size_t rank = 0;
    for (const auto& merge : model.at("merges")) {
      require(merge.is_array() && merge.size() == 2 &&
              merge.at(0).is_string() && merge.at(1).is_string());
      auto a = merge.at(0).get<std::string>(),
           b = merge.at(1).get<std::string>();
      require(bpe->vocabulary.contains(a) && bpe->vocabulary.contains(b) &&
              bpe->vocabulary.contains(a + b));
      require(
          bpe->merges
              .emplace(pair_key(bpe->vocabulary.at(a), bpe->vocabulary.at(b)),
                       ByteBpe::Merge{rank++, bpe->vocabulary.at(a + b)})
              .second);
    }
    std::unordered_set<std::string> added_text;
    for (const auto& a : data.at("added_tokens")) {
      auto id = id_value(a.at("id"));
      auto text = a.at("content").get<std::string>();
      require(!text.empty() && added_text.insert(text).second &&
              !bpe->decoded.contains(id));
      valid_utf8(text);
      require(a.at("normalized") == false && a.at("lstrip") == false &&
              a.at("rstrip") == false && a.at("single_word") == false);
      const auto& ca = cfg.at("added_tokens_decoder").at(std::to_string(id));
      auto expected = a;
      expected.erase("id");
      require(ca == expected);
      bpe->added.push_back({text, id});
      bpe->decoded.emplace(id, std::move(text));
    }
    require(cfg.at("added_tokens_decoder").size() == bpe->added.size());
    static std::once_flag initialized;
    std::call_once(initialized, [] {
      OnigEncoding enc = ONIG_ENCODING_UTF8;
      if (onig_initialize(&enc, 1) != ONIG_NORMAL) {
        throw std::runtime_error("Oniguruma initialization failed");
      }
    });
    OnigRegex regex = nullptr;
    OnigErrorInfo error{};
    // Native API byte/address boundary; no unrelated typed object is
    // dereferenced. NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto* start = reinterpret_cast<const OnigUChar*>(pattern.data());
    const std::span pattern_bytes(start, pattern.size());
    auto status = onig_new(
        &regex, start, pattern_bytes.subspan(pattern_bytes.size()).data(),
        ONIG_OPTION_NONE, ONIG_ENCODING_UTF8, ONIG_SYNTAX_RUBY, &error);
    if (status == ONIGERR_MEMORY) {
      throw std::bad_alloc();
    }
    if (status != ONIG_NORMAL) {
      throw std::runtime_error("tokenizer regex compilation failed");
    }
    bpe->regex.reset(regex);
    for (const auto& entry : bpe->decoded) {
      bpe->longest_token = std::max(bpe->longest_token, entry.second.size());
    }
    return bpe;
  } catch (const Json::exception&) {
    throw std::invalid_argument("invalid tokenizer JSON schema");
  }
}
}  // namespace dolittle::tokenizers
