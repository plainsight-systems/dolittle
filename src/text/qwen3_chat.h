/*
 * Target-generation interface and implementation contract.
 * Axis M5: the selected Qwen3 target's single-turn chat/thinking convention.
 * Inputs: verified tokenizer_config.json and generation_config.json bytes,
 * source identity and a byte budget. Output: an owned immutable Protocol.
 * Validate the supported template branch, added-token IDs and stop settings
 * against the pinned tokenizer. Never execute arbitrary Jinja from a checkpoint.
 * Reject unsupported template/config changes instead of guessing equivalence.
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
 * Static-analysis design/implementation, 2026-10-19: named source records
 * distinguish metadata roles without copying bytes. Core I.1 (#ri-explicit);
 * performance CDSA.32 (source cited below): parse borrowed bytes once into owned
 * lookup tables; no view is retained.
 * Reference: Qwen/Qwen3-8B at b968826d9c46dd6066d109eabc6255188de91218;
 * artifact hashes are in scripts/validation/models-8b.json.
 * Overlay: docs/architecture/logical-overview.md, "1 Establish the run".
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): exact template bytes and
 * control-token transitions, independent of model arithmetic.
 * cpp-perf-guidelines CDSA.32: resolve fixed template/control data once.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include "text/protocol.h"
#include <memory>

namespace dolittle::text {
// Named borrowed inputs are valid only for the duration of the factory call.
struct Qwen3ChatSources {
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
    std::string_view verified_source_identity;
};
[[nodiscard]] std::unique_ptr<Protocol> make_qwen3_chat(Qwen3ChatSources sources,
                                                        std::size_t max_formatted_bytes);
} // namespace dolittle::text
