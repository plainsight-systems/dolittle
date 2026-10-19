/*
 * Axis R2: pure generation progression, independent of devices and sampling.
 * Inputs: prompt length P>0, emitted count G, consumed count S, token budget
 * and stop flags. Output: the next action and a reason only when finishing.
 * 1. Reject impossible counts, overflow, emission before prompt completion,
 *    more than one pending generated token, or a stop ID without emission.
 * 2. If S<P, consume the remaining prompt. If S<P+G, consume the pending ID.
 *    Neither stop flags nor token exhaustion may skip these captures.
 * 3. Once every emitted ID is consumed, finish on stop ID, then token limit,
 *    then user stop, in that precedence. Otherwise request one sampling draw.
 * Invalid states throw invalid_argument; no input or external state is changed.
 * Cost: O(1), no allocation, GPU access or random draw. Optimization: this
 * explicit transition rule is reused by the loop and tested without a runtime.
 * Overlay: logical-overview.md, "2 Generate and capture" and "Capture alignment".
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): consumption must precede terminal completion.
 * cpp-perf-guidelines GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md):
 * a host decision happens only at the completed chunk/readback boundary.
 */
#include "inference/generation.h"
#include <limits>
#include <stdexcept>
namespace dolittle::inference {
Decision next_action(Progress p) {
    if (!p.prompt_tokens || p.generated_tokens > p.max_new_tokens ||
        p.prompt_tokens > std::numeric_limits<std::size_t>::max() - p.generated_tokens ||
        p.consumed_tokens > p.prompt_tokens + p.generated_tokens ||
        (!p.generated_tokens && p.last_emitted_is_stop) ||
        (p.generated_tokens && (p.consumed_tokens < p.prompt_tokens ||
                                p.prompt_tokens + p.generated_tokens - p.consumed_tokens > 1))) {
        throw std::invalid_argument("invalid sequence progression");
    }
    if (p.consumed_tokens < p.prompt_tokens) {
        return {Action::consume_prompt, std::nullopt};
    }
    if (p.consumed_tokens < p.prompt_tokens + p.generated_tokens) {
        return {Action::consume_pending, std::nullopt};
    }
    if (p.last_emitted_is_stop) {
        return {Action::finish, StopReason::stop_token};
    }
    if (p.generated_tokens == p.max_new_tokens) {
        return {Action::finish, StopReason::token_limit};
    }
    if (p.stop_requested) {
        return {Action::finish, StopReason::user_stop};
    }
    return {Action::sample, std::nullopt};
}
} // namespace dolittle::inference
