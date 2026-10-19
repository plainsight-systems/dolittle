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
#include "text/qwen3_chat.h"
#include "tests/support/check.h"
#include "tests/support/text_fixture.h"
#include <array>
using namespace dolittle;
using namespace dolittle::tests;
// A value-only collaborator isolates formatting from the BPE implementation.
// It recognizes the protocol controls and otherwise echoes input bytes as IDs.
struct ByteEchoTokenizer final : tokenizers::Tokenizer {
    std::string_view identity() const noexcept override {
        return "test/byte-echo";
    }
    std::vector<std::uint32_t> encode(std::string_view text) const override {
        const std::array<std::pair<std::string_view, std::uint32_t>, 5> controls{{{"<|endoftext|>", 151643},
                                                                                  {"<|im_start|>", 151644},
                                                                                  {"<|im_end|>", 151645},
                                                                                  {"<think>", 151667},
                                                                                  {"</think>", 151668}}};
        for (const auto &[spelling, id] : controls)
            if (text == spelling)
                return {id};
        std::vector<std::uint32_t> result;
        for (unsigned char byte : text)
            result.push_back(byte);
        return result;
    }
    bool contains(std::uint32_t id) const noexcept override {
        return id < 256;
    }
    tokenizers::DecodedBytes decode(std::span<const std::uint32_t>) const override {
        throw std::logic_error("byte-echo collaborator is encode-only");
    }
};
int main() {
    ByteEchoTokenizer tokenizer;
    auto protocol = text::make_qwen3_chat(qwen3_config_json, qwen3_generation_json, "test", 4096);
    auto p = protocol->prepare({std::nullopt, "abc", text::ThinkingMode::enabled}, tokenizer);
    check(p.formatted_utf8 == "<|im_start|>user\nabc<|im_end|>\n<|im_start|>assistant\n",
          "enabled prompt does not invent thinking opener");
    check(p.ids == tokenizer.encode(p.formatted_utf8), "encode complete formatted text");
    auto disabled = protocol->prepare({std::string{}, "abc", text::ThinkingMode::disabled}, tokenizer);
    check(disabled.formatted_utf8.starts_with("<|im_start|>system\n<|im_end|>\n") &&
              disabled.formatted_utf8.ends_with("<think>\n\n</think>\n\n"),
          "empty system and disabled suffix");
    using K = text::SegmentKind;
    const std::array<std::uint32_t, 8> ids{97, 151667, 98, 151668, 99, 151668, 100, 151645};
    const std::array kinds{K::unclassified, K::control, K::reasoning,    K::control,
                           K::answer,       K::control, K::unclassified, K::control};
    auto segments = protocol->segment(ids, 19, text::ThinkingMode::enabled);
    for (std::size_t i = 0; i < ids.size(); ++i)
        check(segments[i].position == 19 + i && segments[i].kind == kinds[i],
              "delimiter state and absolute provenance");
    auto other = text::make_qwen3_chat(qwen3_config_json, qwen3_generation_json, "wrong", 4096);
    expect_throws<std::invalid_argument>(
        [&] { (void)other->prepare({{}, "abc", text::ThinkingMode::enabled}, tokenizer); });
    expect_throws<std::invalid_argument>(
        [] { (void)text::make_qwen3_chat("{}", qwen3_generation_json, "test", 4096); });
}
