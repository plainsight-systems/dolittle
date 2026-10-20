/*
 * Axis R2: complete target/AV sequence generation and observed result assembly.
 * Contract: borrow a fresh compatible CachedForwardPlan and validated prompt;
 * admit P prompt IDs plus G generated IDs before starting. Live execution uses
 * device sampling; the explicitly selected host_reference path uses the
 caller's
 * TokenSelector. Results own exact IDs, consumed positions, provenance and
 * capture leases. Device must outlive those leases. No model/block constant is
 * hardcoded here. One run is serial and cannot resume after finalization.
 *
 * Algorithm: execution-boundary.md sections 3-5; logical-overview.md
 generation.
 * 1. Validate IDs/settings/identity/capacity, reserve result metadata and
 prepare
 *    the device graph, journal and capture arenas. Upload the complete prompt.
 * 2. Start once. The GPU consumes full prompt chunks and its remainder, then
 *    selects the first ID from the final prompt row. Each decode consumes the
 *    previously selected ID before deciding whether to select another.
 * 3. Collect immutable metadata: selection appends an ID; consumed-range events
 *    advance the capture watermark; injection events report a valid source
 norm.
 *    The collector never supplies input, acknowledges a slot or relaunches
 work.
 * 4. Terminal metadata supplies EOS, budget, cancellation or validation
 failure.
 *    Drain the graph and seal the result. Every selected token, including EOS,
 *    receives its own forward pass/capture before graceful termination.
 * Cancellation may end prefill with an unconsumed prompt suffix. An observer
 * may lag until GPU completion; callback failure requests cancellation but
 * collection still preserves all selected/consumed metadata. Borrow callback
 * spans only for the call. Checked builds reject steady-state allocations.
 * CUDA/transport failure drains and discards device capture leases rather than
 * advertising invalid tensors. Device-side validation failure retains only the
 * already completed prefix. Preparation failures throw before launch.
 *
 * The host_reference path separately consumes chunks, downloads logits and
 * captures, calls the supplied selector and consumes its final selected ID.
 * This diagnostic mode does not satisfy the live execution boundary.
 * Numerical laws belong to kernels/sampler and remain unchanged here.
 * Cost: O(P+G) host IDs/positions and metadata journal; O((P+G)*sum(row_bytes))
 * persistent device captures. No ordinary activation/logit readback, injection
 * upload, host RNG or per-token GPU submission. Optimization: all successful
 * publication writes use prepared capacity; GPU continuation has no consumer
 * dependency. Capture owners survive plan/cache/model teardown.
 *
 * Corpus consultation, header design and implementation, 2026-10-20: both local
 * servers queried separately in each phase and the cited rules read in full.
 * cpp-guidelines I.1 (#ri-explicit), R.1 (#rr-raii): explicit residency,
 * completion, ownership and failure contracts. cpp-perf-guidelines GPU.1,
 * GPU.6, GPU.7, GPU.9: keep tensor intermediates/decisions on device, batch
 * dispatch, decouple observation and separate persistent from transient arenas.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Related GPU.6/7/9 entries describe dispatch, staging and storage.
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
  std::optional<float> injection_norm{};
};
// Optional observer contract; callback values are borrowed for the call.
struct GenerationUpdate {
  std::span<const tokenizers::TokenId> appended;
  std::size_t first_position;
  std::size_t prompt_tokens;
  std::size_t consumed_tokens;
};
class GenerationObserver {
 public:
  GenerationObserver() = default;
  GenerationObserver(const GenerationObserver&) = delete;
  GenerationObserver& operator=(const GenerationObserver&) = delete;
  GenerationObserver(GenerationObserver&&) = delete;
  GenerationObserver& operator=(GenerationObserver&&) = delete;
  virtual ~GenerationObserver() = default;
  [[nodiscard]] virtual bool update(const GenerationUpdate&) noexcept = 0;
};
[[nodiscard]] GenerationResult generate(
    CachedForwardPlan&, const text::PreparedPrompt&, GenerationSettings,
    const sampling::TokenSelector&, ExecutionId, GenerationObserver&,
    const std::stop_token& stop = {},
    sampling::Execution execution = sampling::Execution::device);
[[nodiscard]] GenerationResult generate(
    CachedForwardPlan&, const text::PreparedPrompt&, GenerationSettings,
    const sampling::TokenSelector&, ExecutionId,
    const std::stop_token& stop = {},
    sampling::Execution execution = sampling::Execution::device);
}  // namespace dolittle::inference
