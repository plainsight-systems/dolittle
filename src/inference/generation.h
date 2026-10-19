/*
 * Complete target-generation contract.
 * Axis R2: one sequence's token progression, stopping and capture finalization.
 * Inputs: a prepared prompt, compatible cached execution plan, sampler
 * settings, seed, maximum new tokens and a cooperative stop token. Output:
 * owned exact prompt/generated IDs, consumed length, raw captures, provenance
 * and stop reason. The caller establishes tokenizer/protocol pairing with the
 * verified model. Generate validates nonempty prompt IDs, fresh prefix,
 * capacity and sampler inputs before upload. The selector is borrowed for the
 * call, selected at composition, and obeys sampler.h; it is invoked once per
 * emitted token. No tokenizer, template, sampling law, cache layout or block 24
 * is hardcoded here.
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
 * An unconsumed emitted token may remain in that failed trace without a
 * capture. Preflight errors throw before a run starts. Resource exhaustion may
 * throw if no result can be materialized; it must not publish fabricated
 * completion. next_action is the pure progression rule used by the loop:
 * consume unfinished prompt first, then any pending emitted token, then finish
 * or sample. It rejects impossible counts (including more than one pending
 * generated token). Stopping never takes precedence over consuming a pending
 * token. It touches no GPU, tokenizer or RNG, allowing state invariants to be
 * tested as an isolated unit.
 *
 * Design and cost: one resident weight set, one admitted contiguous cache and
 * bounded prefill/decode scratch. Each draw downloads one vocabulary row and
 * waits once; captures join the same readback to preserve a durable host
 * prefix. Reserve trace/result metadata before the first emission. For C
 * admitted positions, each selected BF16 D=4096 site costs 8 KiB*C retained
 * host bytes. Device capture scratch scales with the admitted prompt chunk
 * size, not C; transfer staging and per-call result copies add explicit peak
 * bytes. Optimization: chunk prompt work, reuse the same bounded schedule for
 * decode, and avoid per-block waits or per-token GPU allocation. Finalization
 * may omit vocabulary projection; it must still execute every requested capture
 * site. Owned results survive plan/cache/weights teardown. The plan is serial,
 * single sequence, non-reentrant and not resumable after finish/stop/failure.
 * Overlay: docs/architecture/logical-overview.md, "2 Generate and capture"
 * and "Capture alignment"; transformer-execution.md, "Prefill and decoding".
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post), R.1 (#rr-raii): record
 * exact emitted/consumed state, completion, failure and resource ownership.
 * cpp-perf-guidelines GPU.1: retain KV on device; batch capture bytes with the
 * required logit readback/completion per draw.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.9: admit/reserve persistent, transient and output
 * arenas. Source:
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 *
 * AV reuse (implemented, 2026-10-19): this same loop consumes the
 * Pair's PreparedPrompt and a resident-AV sequence plan.
 * PreparedPrompt.thinking records the AV template setting, not the target
 * trace's classification. The Pair validates its own tokenizer/protocol
 * identities; no target text is copied into the AV prompt. Before the first
 * consume, inspect the plan's optional embedding_override_position and require
 * it < prompt.ids.size(). The executor handles the replacement in whichever
 * prompt chunk contains it; decode must never apply a replacement. Preserve
 * exact AV prompt/generated IDs, settings, seed, consumed count and
 * stop/failure as for target generation. The experiment attaches the role and
 * original capture identity, then extracts output tags. No second generation
 * algorithm, new sampling law or early tag-based stop. Optimization: reuse
 * progression and bounded decode work; only one injected row adds a transfer,
 * with no per-token change after prefill. Overlay: logical-overview.md, "4
 * Interpret selected positions". Corpus consultation, header design and
 * implementation (full rules read), 2026-10-19: cpp-guidelines I.7 (#ri-post):
 * replacement is restricted to the prompt. cpp-perf-guidelines GPU.1: no
 * repeated injection transfer during decode; source
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include <cstdint>
#include <stop_token>

#include "inference/executor.h"
#include "inference/sampling/sampler.h"
#include "text/protocol.h"

namespace dolittle::inference {
enum class StopReason : std::uint8_t {
  stop_token,
  token_limit,
  user_stop,
  execution_failure
};
enum class Action : std::uint8_t {
  consume_prompt,
  consume_pending,
  sample,
  finish
};
struct Progress {
  std::size_t prompt_tokens;
  std::size_t generated_tokens;  // Emitted tokens, excluding the prompt.
  std::size_t consumed_tokens;   // Includes completed prompt positions.
  std::size_t max_new_tokens;
  bool last_emitted_is_stop;
  bool stop_requested;
};
struct Decision {
  Action action{};
  std::optional<StopReason> reason;  // Present only for finish.
};
[[nodiscard]] Decision next_action(Progress);
struct GenerationSettings {
  sampling::Settings sampling;
  std::uint64_t seed;
  std::size_t max_new_tokens;
};
struct GenerationResult {
  ExecutionId execution_id{};
  std::string checkpoint_identity;
  std::string tokenizer_identity;
  std::string protocol_identity;
  tensors::ComputationMode numerical_mode{};
  text::ThinkingMode thinking{};
  GenerationSettings settings{};
  std::vector<std::uint32_t> stop_ids;
  std::vector<std::uint32_t> tokens;
  std::size_t prompt_tokens{};
  std::size_t consumed_tokens{};
  std::vector<OwnedCapture> captures;
  StopReason reason{};
  std::string failure_message;
};
[[nodiscard]] GenerationResult generate(CachedForwardPlan&,
                                        const text::PreparedPrompt&,
                                        GenerationSettings,
                                        const sampling::TokenSelector&,
                                        ExecutionId,
                                        const std::stop_token& stop = {});
}  // namespace dolittle::inference
