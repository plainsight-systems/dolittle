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
#include "inference/generation.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "bounds.h"
#include "diagnostics/allocation.h"
#include "tensors/representations/bf16.h"
#include "tensors/tensor.h"

namespace dolittle::inference {
namespace {
GenerationResult generate_impl(CachedForwardPlan& plan,
                               const text::PreparedPrompt& prompt,
                               GenerationSettings settings,
                               const sampling::TokenSelector& select,
                               ExecutionId id, const std::stop_token& stop,
                               GenerationObserver* observer,
                               sampling::Execution execution) {
  sampling::validate_settings(settings.sampling);
  if ((execution != sampling::Execution::host_reference &&
       execution != sampling::Execution::device) ||
      (execution == sampling::Execution::host_reference && !select) ||
      !plan.prefix().usable || plan.prefix().committed_positions ||
      prompt.ids.empty() || prompt.tokenizer_identity.empty() ||
      prompt.protocol_identity.empty() || prompt.stop_ids.empty() ||
      prompt.ids.size() > plan.capacity() ||
      settings.max_new_tokens > plan.capacity() - prompt.ids.size() ||
      settings.sampling.top_k > plan.vocabulary_size() ||
      id.value != plan.sequence_id().value) {
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
  if (prompt.thinking != text::ThinkingMode::enabled &&
      prompt.thinking != text::ThinkingMode::disabled) {
    throw std::invalid_argument("invalid generation thinking mode");
  }
  if (const auto position = plan.embedding_override_position();
      position && position->value >= prompt.ids.size()) {
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
  const bool on_device = execution == sampling::Execution::device;
  const auto capacity = prompt.ids.size() + settings.max_new_tokens;
  result.tokens.reserve(capacity);
  result.tokens.assign(prompt.ids.begin(), prompt.ids.end());
  result.captures.reserve(plan.capture_sites().size());
  for (auto site : plan.capture_sites()) {
    const auto row =
        site.vector_width * tensors::element_bytes(site.storage_type);
    if (row && capacity > std::numeric_limits<std::size_t>::max() / row) {
      throw std::length_error("generation capture size overflow");
    }
    OwnedCapture capture{site, {}, {}};
    capture.positions.reserve(capacity);
    if (!on_device) {
      capture.bytes.reserve(capacity * row);
    }
    result.captures.push_back(std::move(capture));
  }
  constexpr std::size_t failure_message_capacity = 128;
  result.failure_message.reserve(failure_message_capacity);
  bool publication_failed = false;
  const auto notify = [&](std::size_t first) {
    if (observer && !publication_failed) {
      publication_failed =
          !observer->update({std::span(result.tokens).subspan(first), first,
                             result.prompt_tokens, result.consumed_tokens});
    }
  };
  if (on_device) {
    plan.prepare_device_generation({.sampling = settings.sampling,
                                    .seed = settings.seed,
                                    .prompt_ids = prompt.ids,
                                    .stop_ids = prompt.stop_ids,
                                    .max_new_tokens = settings.max_new_tokens});
    const auto captures = plan.device_captures();
    for (std::size_t i = 0; i < result.captures.size(); ++i) {
      result.captures.at(i).device = dolittle::at(captures, i);
    }
    try {
      if (stop.stop_requested()) {
        plan.request_stop();
      }
      plan.start_sequence();
      {
        const diagnostics::AllocationScope scope(true, &publication_failed);
        notify(0);
      }
      for (;;) {
        const diagnostics::AllocationScope scope(true, &publication_failed);
        const auto event = plan.collect_sequence(stop);
        switch (event.kind) {
          case DeviceSequence::Kind::selected:
            if (event.position != result.tokens.size() ||
                event.token >= plan.vocabulary_size()) {
              throw std::runtime_error("invalid selected token publication");
            }
            result.tokens.push_back(event.token);
            notify(result.tokens.size() - 1);
            break;
          case DeviceSequence::Kind::consumed:
            if (event.position != result.consumed_tokens ||
                event.count > result.tokens.size() - result.consumed_tokens) {
              throw std::runtime_error(
                  "invalid capture completion publication");
            }
            for (auto& capture : result.captures) {
              for (std::size_t i = 0; i < event.count; ++i) {
                capture.positions.push_back({event.position + i});
              }
            }
            result.consumed_tokens += event.count;
            notify(result.tokens.size());
            break;
          case DeviceSequence::Kind::injection:
            if (!event.status) {
              result.injection_norm = event.norm;
            }
            break;
          case DeviceSequence::Kind::terminal:
            switch (static_cast<DeviceSequence::Reason>(event.status)) {
              case DeviceSequence::Reason::limit:
                result.reason = StopReason::token_limit;
                break;
              case DeviceSequence::Reason::eos:
                result.reason = StopReason::stop_token;
                break;
              case DeviceSequence::Reason::cancelled:
                result.reason = StopReason::user_stop;
                break;
              case DeviceSequence::Reason::failure:
                result.reason = StopReason::execution_failure;
                break;
            }
            plan.finish();
            if (result.reason == StopReason::execution_failure) {
              result.failure_message =
                  "device injection or sampling validation failed";
            }
            return result;
        }
        if (publication_failed) {
          // Request cancellation without driving GPU scheduling; still drain
          // the immutable journal so every selected token retains its
          // consumed state.
          plan.request_stop();
        }
      }
    } catch (const std::exception& e) {
      result.reason = StopReason::execution_failure;
      result.failure_message = e.what();
      // A device/transport failure cannot promise validity of leased
      // tensors.
      for (auto& capture : result.captures) {
        capture.positions.clear();
        capture.device.reset();
      }
      if (plan.prefix().usable) {
        plan.finish();
      }
      return result;
    }
  }
  // Explicit host reference execution begins here; the live branch returned.
  sampling::RandomStream rng(settings.seed);
  std::vector<float> logits(plan.vocabulary_size());
  notify(0);
  bool have_logits = false, last_stop = false;
  const auto accept = [&](ConsumedView chunk) {
    for (std::size_t i = 0; i < chunk.captures.size(); ++i) {
      auto& into = result.captures.at(i);
      const auto& from = dolittle::at(chunk.captures, i);
      into.positions.insert(into.positions.end(), from.positions.begin(),
                            from.positions.end());
      into.bytes.insert(into.bytes.end(), from.bytes.begin(), from.bytes.end());
    }
    result.consumed_tokens += chunk.tokens.size();
    notify(result.tokens.size());
    have_logits = chunk.next_logits.has_value();
    if (chunk.next_logits) {
      const auto& row = *chunk.next_logits;
      if (row.storage_type == tensors::ElementType::float32) {
        std::memcpy(logits.data(), row.bytes.data(), row.bytes.size());
      } else {
        for (std::size_t i = 0; i < logits.size(); ++i) {
          logits.at(i) =
              tensors::to_float(tensors::read_le(std::span<const std::byte, 2>(
                  std::span(row.bytes).subspan(2 * i).data(), 2)));
        }
      }
    }
  };
  const auto consume = [&](std::span<const std::uint32_t> ids,
                           bool need_logits) {
    const auto view = plan.consume_view(ids, need_logits);
    accept(view);
  };

  try {
    for (;;) {
      Progress progress{result.prompt_tokens,
                        result.tokens.size() - result.prompt_tokens,
                        result.consumed_tokens,
                        settings.max_new_tokens,
                        last_stop,
                        (stop.stop_requested() || publication_failed)};
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
        const auto n = std::min(plan.max_prefill_chunk(),
                                result.prompt_tokens - result.consumed_tokens);
        const bool final = result.consumed_tokens + n == result.prompt_tokens;
        consume(std::span<const std::uint32_t>(result.tokens)
                    .subspan(result.consumed_tokens, n),
                final && settings.max_new_tokens &&
                    !(stop.stop_requested() || publication_failed));
      } else if (decision.action == Action::consume_pending) {
        const bool another =
            !last_stop && progress.generated_tokens < settings.max_new_tokens &&
            !(stop.stop_requested() || publication_failed);
        consume(std::span<const std::uint32_t>(result.tokens)
                    .subspan(result.consumed_tokens, 1),
                another);
      } else {
        if (!have_logits) {
          throw std::logic_error(
              "generation requested sampling without logits");
        }
        const auto token = select(logits, settings.sampling, rng);
        if (token >= plan.vocabulary_size()) {
          throw std::invalid_argument("sampler returned out-of-vocabulary ID");
        }
        result.tokens.push_back(token);
        last_stop = std::find(result.stop_ids.begin(), result.stop_ids.end(),
                              token) != result.stop_ids.end();
        notify(result.tokens.size() - 1);
      }
    }
  } catch (const std::exception& e) {
    result.reason = StopReason::execution_failure;
    result.failure_message = e.what();
    if (plan.prefix().usable) {
      plan.finish();
    }
    return result;
  }
}
}  // namespace
GenerationResult generate(CachedForwardPlan& plan,
                          const text::PreparedPrompt& prompt,
                          GenerationSettings settings,
                          const sampling::TokenSelector& select, ExecutionId id,
                          const std::stop_token& stop,
                          sampling::Execution execution) {
  return generate_impl(plan, prompt, settings, select, id, stop, nullptr,
                       execution);
}
GenerationResult generate(CachedForwardPlan& plan,
                          const text::PreparedPrompt& prompt,
                          GenerationSettings settings,
                          const sampling::TokenSelector& select, ExecutionId id,
                          GenerationObserver& observer,
                          const std::stop_token& stop,
                          sampling::Execution execution) {
  return generate_impl(plan, prompt, settings, select, id, stop, &observer,
                       execution);
}
}  // namespace dolittle::inference
