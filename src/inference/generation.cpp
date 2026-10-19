/*
 * Complete target-generation execution loop.
 * Axis R2: one sequence's token progression, stopping and capture finalization.
 * Inputs: a prepared prompt, compatible cached execution plan, sampler settings,
 * seed, maximum new tokens and a cooperative stop token. Output: owned exact
 * prompt/generated IDs, consumed length, raw captures, provenance and stop reason.
 * The caller establishes tokenizer/protocol pairing with the verified model.
 * Generate validates nonempty prompt IDs, fresh prefix, capacity and sampler
 * inputs before upload. The selector is borrowed for the call, selected at
 * composition, and obeys sampler.h; it is invoked once per emitted token.
 * No tokenizer, template, sampling law, cache layout or block 24 is hardcoded here.
 *
 * Algorithm (P prompt tokens, G maximum new tokens):
 * 1. Cached-plan preparation admits capacity C, weights, cache, scratch and
 *    host outputs. Here require P+G <= C before uploading token IDs, including
 *    room to consume the final emitted token. Reject rather than truncate.
 * 2. Consume prompt chunks through cached execution, preserving absolute
 *    positions. Only the last prompt row needs logits for the first draw.
 * 3. If stopping is already requested or G=0, finish after consuming the prompt
 *    without drawing. Otherwise select one token from the latest logits and
 *    append its ID to the trace as emitted but not yet consumed.
 * 4. Consume that token at the next absolute position. Its capture belongs to
 *    this pass, not the pass whose logits selected it. Commit consumed length
 *    only after all block writes and required copies succeed.
 * 5. If its ID is a stop ID, the token budget is exhausted, or cooperative stop
 *    is requested, finish without another draw. Otherwise use its logits for
 *    step 3. Stop precedence is stop ID, then token limit, then user stop.
 * 6. Seal the completed plan and return an owned result. Stop IDs remain
 *    in the exact trace and receive captures even when hidden in display text.
 * A graceful stop is checked between completed calls. It never abandons an
 * emitted-but-unconsumed token; finalization performs no further sampling.
 * Execution failure drains work, invalidates the session and returns a failed
 * result containing only completed captures plus explicit consumed length.
 * An unconsumed emitted token may remain in that failed trace without a capture.
 * Preflight errors throw before a run starts. Resource exhaustion may throw if
 * no result can be materialized; it must not publish fabricated completion.
 * next_action is the pure progression rule used by the loop: consume unfinished
 * prompt first, then any pending emitted token, then finish or sample. It rejects
 * impossible counts (including more than one pending generated token). Stopping
 * never takes precedence over consuming a pending token. It touches no GPU,
 * tokenizer or RNG, allowing state invariants to be tested as an isolated unit.
 *
 * Design and cost: one resident weight set, one admitted contiguous cache and
 * bounded prefill/decode scratch. Each draw downloads one vocabulary row and
 * waits once; captures join the same readback to preserve a durable host prefix.
 * Reserve trace/result metadata before the first emission. For C admitted
 * positions, each selected BF16 D=4096 site costs 8 KiB*C retained host bytes.
 * Device capture scratch scales with the admitted prompt chunk size, not C;
 * transfer staging and per-call result copies add explicit peak bytes.
 * Optimization: chunk prompt work, reuse the same bounded schedule for decode,
 * and avoid per-block waits or per-token GPU allocation. Finalization may omit
 * vocabulary projection; it must still execute every requested capture site.
 * Owned results survive plan/cache/weights teardown. The plan is serial, single
 * sequence, non-reentrant and not resumable after finish/stop/failure.
 * Overlay: docs/architecture/logical-overview.md, "2 Generate and capture"
 * and "Capture alignment"; transformer-execution.md, "Prefill and decoding".
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post), R.1 (#rr-raii): record exact
 * emitted/consumed state, completion, failure and resource ownership.
 * cpp-perf-guidelines GPU.1: retain KV on device; batch capture bytes with the
 * required logit readback/completion per draw.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.9: admit/reserve persistent, transient and output arenas.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 *
 * The loop also runs AV prompts. Before any consume, an optional replacement
 * position must be inside the prompt; executor applies it once in the containing
 * prefill chunk. AV IDs, stop/failure and final consumed-token semantics are the
 * same as target generation, with role/alignment owned by the experiment record.
 * Corpus consultation, implementation (both servers queried, full rules read),
 * 2026-10-19: cpp-guidelines R.1 (#rr-raii) ties resources to their owning phase;
 * I.7 (#ri-post) makes completion, provenance and failure effects explicit.
 * cpp-perf-guidelines GPU.1 keeps intermediate embeddings on device and budgets
 * required transfers; source guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.9 separates persistent weights from sequence storage;
 * source guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "inference/generation.h"
#include "tensors/tensor.h"
#include "tensors/representations/bf16.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace dolittle::inference {
GenerationResult generate(CachedForwardPlan &plan, const text::PreparedPrompt &prompt,
                          GenerationSettings settings, const sampling::TokenSelector &select, ExecutionId id,
                          const std::stop_token &stop) {
    sampling::validate_settings(settings.sampling);
    if (!select || !plan.prefix().usable || plan.prefix().committed_positions || prompt.ids.empty() ||
        prompt.tokenizer_identity.empty() || prompt.protocol_identity.empty() || prompt.stop_ids.empty() ||
        prompt.ids.size() > plan.capacity() ||
        settings.max_new_tokens > plan.capacity() - prompt.ids.size() ||
        settings.sampling.top_k > plan.vocabulary_size() || id.value != plan.sequence_id().value) {
        throw std::invalid_argument("incompatible generation request");
    }
    for (auto token : prompt.ids) {
        if (token >= plan.vocabulary_size()) {
            throw std::invalid_argument("prompt ID outside vocabulary");
        }
    }
    for (auto token : prompt.stop_ids) {
        if (token >= plan.vocabulary_size()) {
            throw std::invalid_argument("stop ID outside vocabulary");
        }
    }
    if (prompt.thinking != text::ThinkingMode::enabled && prompt.thinking != text::ThinkingMode::disabled) {
        throw std::invalid_argument("invalid generation thinking mode");
    }
    if (const auto position = plan.embedding_override_position(); position && position->value >= prompt.ids.size()) {
        throw std::invalid_argument("embedding replacement must belong to prompt");
    }
    GenerationResult result{};
    result.execution_id = id;
    result.checkpoint_identity = plan.checkpoint_identity();
    result.tokenizer_identity = prompt.tokenizer_identity;
    result.protocol_identity = prompt.protocol_identity;
    result.numerical_mode = plan.numerical_mode();
    result.thinking = prompt.thinking;
    result.settings = settings;
    result.stop_ids = prompt.stop_ids;
    result.prompt_tokens = prompt.ids.size();
    result.consumed_tokens = 0;
    const auto capacity = prompt.ids.size() + settings.max_new_tokens;
    result.tokens.reserve(capacity);
    result.tokens.assign(prompt.ids.begin(), prompt.ids.end());
    result.captures.reserve(plan.capture_sites().size());
    for (auto site : plan.capture_sites()) {
        const auto row = site.vector_width * tensors::element_bytes(site.storage_type);
        if (row && capacity > std::numeric_limits<std::size_t>::max() / row) {
            throw std::length_error("generation capture size overflow");
        }
        OwnedCapture capture{site, {}, {}};
        capture.positions.reserve(capacity);
        capture.bytes.reserve(capacity * row);
        result.captures.push_back(std::move(capture));
    }
    sampling::RandomStream rng(settings.seed);
    std::vector<float> logits(plan.vocabulary_size());
    bool have_logits = false, last_stop = false;
    const auto accept = [&](ConsumedChunk chunk) {
        for (std::size_t i = 0; i < chunk.captures.size(); ++i) {
            auto &into = result.captures.at(i);
            const auto &from = chunk.captures.at(i);
            into.positions.insert(into.positions.end(), from.positions.begin(), from.positions.end());
            into.bytes.insert(into.bytes.end(), from.bytes.begin(), from.bytes.end());
        }
        result.consumed_tokens += chunk.tokens.size();
        have_logits = chunk.next_logits.has_value();
        if (chunk.next_logits) {
            const auto &row = *chunk.next_logits;
            if (row.storage_type == tensors::ElementType::float32) {
                std::memcpy(logits.data(), row.bytes.data(), row.bytes.size());
            } else {
                for (std::size_t i = 0; i < logits.size(); ++i) {
                    logits.at(i) = tensors::to_float(tensors::read_le(
                        std::span<const std::byte, 2>(std::span(row.bytes).subspan(2 * i).data(), 2)));
                }
            }
        }
    };
    try {
        for (;;) {
            Progress progress{result.prompt_tokens,
                              result.tokens.size() - result.prompt_tokens,
                              result.consumed_tokens,
                              settings.max_new_tokens,
                              last_stop,
                              stop.stop_requested()};
            const auto decision = next_action(progress);
            if (decision.action == Action::finish) {
                if (!decision.reason) {
                    throw std::logic_error("finished progression has no stop reason");
                }
                result.reason = *decision.reason;
                plan.finish();
                return result;
            }
            if (decision.action == Action::consume_prompt) {
                const auto n =
                    std::min(plan.max_prefill_chunk(), result.prompt_tokens - result.consumed_tokens);
                const bool final = result.consumed_tokens + n == result.prompt_tokens;
                accept(plan.consume(
                    std::span<const std::uint32_t>(result.tokens).subspan(result.consumed_tokens, n),
                    final && settings.max_new_tokens && !stop.stop_requested()));
            } else if (decision.action == Action::consume_pending) {
                const bool another = !last_stop && progress.generated_tokens < settings.max_new_tokens &&
                                     !stop.stop_requested();
                accept(plan.consume(
                    std::span<const std::uint32_t>(result.tokens).subspan(result.consumed_tokens, 1),
                    another));
            } else {
                if (!have_logits) {
                    throw std::logic_error("generation requested sampling without logits");
                }
                const auto token = select(logits, settings.sampling, rng);
                if (token >= plan.vocabulary_size()) {
                    throw std::invalid_argument("sampler returned out-of-vocabulary ID");
                }
                result.tokens.push_back(token);
                last_stop =
                    std::find(result.stop_ids.begin(), result.stop_ids.end(), token) != result.stop_ids.end();
            }
        }
    } catch (const std::exception &e) {
        result.reason = StopReason::execution_failure;
        result.failure_message = e.what();
        if (plan.prefix().usable) {
            plan.finish();
        }
        return result;
    }
}
} // namespace dolittle::inference
