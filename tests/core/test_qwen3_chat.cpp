/*
 * Axis D3: isolated host contract tests; all inputs are in memory.
 * Algorithm: construct bounded inputs, invoke the unit directly, assert its
 * documented output/rejection and state invariants. No files, GPU or processes.
 * Cost: small deterministic fixtures; no throughput claims.
 * Overlay: transformer-execution.md, "Verification boundaries".
 * cpp-guidelines I.7 (#ri-post): test public postconditions, not private calls.
 * cpp-perf-guidelines GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md):
 * host contract checks are separate from device timing or correctness claims.
 * Both corpora consulted for implementation, 2026-10-18.
 * Prepared-buffer/device-selection contract consultation, 2026-10-20:
 * Core I.7 (#ri-post) and MEM.9
 * (guidelines/memory/MEM.9-embedded-allocate-at-init.md), both corpora queried
 * and full rules read for design/implementation. Exercise public results and
 * capacity/failure contracts; independent reference inputs remain unchanged.
 */
#include <array>

#include "tests/support/check.h"
#include "tests/support/text_fixture.h"
#include "text/qwen3_chat.h"
namespace text = dolittle::text;
namespace tokenizers = dolittle::tokenizers;
using dolittle::tests::byte_bpe_json;
using dolittle::tests::check;
using dolittle::tests::expect_throws;
using dolittle::tests::qwen3_config_json;
using dolittle::tests::qwen3_generation_json;
// A value-only collaborator isolates formatting from the BPE implementation.
// It recognizes the protocol controls and otherwise echoes input bytes as IDs.
struct ByteEchoTokenizer final : tokenizers::Tokenizer {
  std::string_view identity() const noexcept override {
    return "test/byte-echo";
  }
  std::vector<std::uint32_t> encode(std::string_view text) const override {
    const std::array<std::pair<std::string_view, std::uint32_t>, 5> controls{
        {{"<|endoftext|>", 151643},
         {"<|im_start|>", 151644},
         {"<|im_end|>", 151645},
         {"<think>", 151667},
         {"</think>", 151668}}};
    for (const auto& [spelling, id] : controls) {
      if (text == spelling) {
        return {id};
      }
    }
    std::vector<std::uint32_t> result;
    for (unsigned char byte : text) {
      result.push_back(byte);
    }
    return result;
  }
  std::size_t max_token_bytes() const noexcept override { return 1; }
  void decode_into(std::span<const std::uint32_t>,
                   tokenizers::DecodedBytes&) const override {
    throw std::logic_error("byte-echo collaborator is encode-only");
  }
  bool contains(std::uint32_t id) const noexcept override { return id < 256; }
  tokenizers::DecodedBytes decode(
      std::span<const std::uint32_t>) const override {
    throw std::logic_error("byte-echo collaborator is encode-only");
  }
};
int main() try {
  ByteEchoTokenizer tokenizer;
  auto protocol =
      text::make_qwen3_chat({.tokenizer_config_json = qwen3_config_json,
                             .generation_config_json = qwen3_generation_json,
                             .verified_source_identity = "test"},
                            4096);
  auto p = protocol->prepare({std::nullopt, "abc", text::ThinkingMode::enabled},
                             tokenizer);
  check(p.formatted_utf8 ==
            "<|im_start|>user\nabc<|im_end|>\n<|im_start|>assistant\n",
        "enabled prompt does not invent thinking opener");
  check(p.ids == tokenizer.encode(p.formatted_utf8),
        "encode complete formatted text");
  auto disabled = protocol->prepare(
      {std::string{}, "abc", text::ThinkingMode::disabled}, tokenizer);
  check(
      disabled.formatted_utf8.starts_with("<|im_start|>system\n<|im_end|>\n") &&
          disabled.formatted_utf8.ends_with("<think>\n\n</think>\n\n"),
      "empty system and disabled suffix");
  using K = text::SegmentKind;
  const std::array<std::uint32_t, 8> ids{97, 151667, 98,  151668,
                                         99, 151668, 100, 151645};
  const std::array kinds{K::unclassified, K::control, K::reasoning,
                         K::control,      K::answer,  K::control,
                         K::unclassified, K::control};
  {
    auto cursor = protocol->make_cursor(19, text::ThinkingMode::enabled);
    std::vector<text::TokenSegment> output;
    dolittle::tests::expect_throws<std::length_error>(
        [&] { cursor->append_into(ids, 19, output); });
    output.reserve(ids.size());
    cursor->append_into(ids, 19, output);
    check(
        output.size() == ids.size() && output.front().position == 19 &&
            output.back().kind == kinds.back(),
        "capacity failure preserves cursor position and classification state");
  }
  auto segments = protocol->segment(ids, 19, text::ThinkingMode::enabled);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    check(
        segments.at(i).position == 19 + i && segments.at(i).kind == kinds.at(i),
        "delimiter state and absolute provenance");
  }
  // Chunking preserves the public classification and absolute-position
  // contract.
  for (std::size_t split = 0; split <= ids.size(); ++split) {
    auto cursor = protocol->make_cursor(19, text::ThinkingMode::enabled);
    auto prefix = cursor->append(std::span(ids).first(split), 19);
    expect_throws<std::invalid_argument>(
        [&] { (void)cursor->append({}, 20 + split); });
    auto suffix = cursor->append(std::span(ids).subspan(split), 19 + split);
    prefix.insert(prefix.end(), suffix.begin(), suffix.end());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      check(prefix.at(i).position == segments.at(i).position &&
                prefix.at(i).kind == segments.at(i).kind,
            "chunk-invariant segmentation");
    }
  }
  auto other =
      text::make_qwen3_chat({.tokenizer_config_json = qwen3_config_json,
                             .generation_config_json = qwen3_generation_json,
                             .verified_source_identity = "wrong"},
                            4096);
  expect_throws<std::invalid_argument>([&] {
    (void)other->prepare({{}, "abc", text::ThinkingMode::enabled}, tokenizer);
  });
  expect_throws<std::invalid_argument>([] {
    (void)text::make_qwen3_chat(
        {.tokenizer_config_json = "{}",
         .generation_config_json = qwen3_generation_json,
         .verified_source_identity = "test"},
        4096);
  });
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
