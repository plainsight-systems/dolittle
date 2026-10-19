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
 */
#include <array>
#include <nlohmann/json.hpp>

#include "tests/support/check.h"
#include "tests/support/text_fixture.h"
#include "tokenizers/byte_bpe.h"
namespace tokenizers = dolittle::tokenizers;
using dolittle::tests::byte_bpe_json;
using dolittle::tests::check;
using dolittle::tests::expect_throws;
using dolittle::tests::qwen3_config_json;
using dolittle::tests::qwen3_generation_json;
int main() try {
  const tokenizers::ByteBpeLimits limits{65536, 4096, 4096, 4096};
  auto tokenizer =
      tokenizers::make_byte_bpe({.tokenizer_json = byte_bpe_json,
                                 .tokenizer_config_json = qwen3_config_json,
                                 .verified_source_identity = "verified-test"},
                                limits);
  check(tokenizer->encode("abc") == std::vector<std::uint32_t>{257},
        "ranked adjacent merges");
  check(
      tokenizer->encode("aaaaa") == std::vector<std::uint32_t>({259, 259, 97}),
      "leftmost overlapping merge");
  check(tokenizer->encode("").empty(), "empty encoding");
  const auto nfc = tokenizer->decode(tokenizer->encode("e\xcc\x81"));
  check(nfc.bytes == "\xc3\xa9", "NFC normalization");
  const std::string text = "x<think>abc</think>!";
  auto ids = tokenizer->encode(text);
  auto decoded = tokenizer->decode(ids);
  check(decoded.bytes == text && decoded.tokens.size() == ids.size(),
        "added tokens retain literal bytes");
  auto bytes =
      tokenizer->decode(std::array<std::uint32_t, 3>{195, 169, 999999});
  check(bytes.bytes == "\xc3\xa9" && bytes.tokens.at(0).end == 1 &&
            bytes.tokens.at(1).begin == 1 && !bytes.tokens.at(2).mapped &&
            bytes.tokens.at(2).begin == bytes.tokens.at(2).end,
        "raw split UTF8 and unmapped provenance");
  const std::string zero("a\0b", 3);
  check(tokenizer->decode(tokenizer->encode(zero)).bytes == zero,
        "embedded NUL is preserved");
  expect_throws<std::invalid_argument>(
      [&] { (void)tokenizer->encode("\xff"); });
  auto bad = nlohmann::json::parse(byte_bpe_json);
  bad.at("model").at("byte_fallback") = true;
  expect_throws<std::invalid_argument>([&] {
    (void)tokenizers::make_byte_bpe({.tokenizer_json = bad.dump(),
                                     .tokenizer_config_json = qwen3_config_json,
                                     .verified_source_identity = "test"},
                                    limits);
  });
  auto small = limits;
  small.max_piece_bytes = 2;
  auto bounded =
      tokenizers::make_byte_bpe({.tokenizer_json = byte_bpe_json,
                                 .tokenizer_config_json = qwen3_config_json,
                                 .verified_source_identity = "test"},
                                small);
  expect_throws<std::length_error>([&] { (void)bounded->encode("abc"); });
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
