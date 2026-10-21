/*
 * X1/X4 bounded timeline implementation; contract: sequence_plan.h. Derive all
 * prompt/head/decode/finalize entries from admitted P/G/B; split publication at
 * the initial head and the configured window limit. Reserve maximum event
 * capacity, including two control events per window. No generated value or GPU
 * API participates. O(ceil(P/B)+G) setup/metadata; no runtime allocations.
 * Overlay: metal-port.md, Metal scheduling. Both implementation corpora queried
 * and read 2026-10-21: Core I.1/I.7 and ES.103 (#res-overflow) bound the
 * result; performance GPU.6 (guidelines/gpu/GPU.6-batch-tiny-gpu-work.md)
 * amortizes publication, GPU.7
 * (guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md) gives each completed
 * window independent ownership. This is not a CPU decode loop.
 */
#include "kernels/metal/sequence_plan.h"

#include <limits>
#include <stdexcept>

namespace dolittle::kernels::metal {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::length_error("sequence schedule size overflows");
  }
  return a + b;
}
}  // namespace
SequencePlan plan_sequence(inference::DeviceSequence::Shape shape,
                           PublicationPolicy policy) {
  if (!shape.prompt || !shape.chunk ||
      shape.chunk > std::numeric_limits<std::uint32_t>::max() ||
      !policy.iterations_per_window || !policy.max_steps ||
      (shape.forward_only && shape.generated)) {
    throw std::invalid_argument("invalid Metal sequence shape/policy");
  }
  (void)add(shape.prompt, shape.generated);
  const bool head = shape.generated != 0 || shape.forward_only;
  const auto prompt_steps =
      add(shape.prompt / shape.chunk, shape.prompt % shape.chunk != 0 ? 1 : 0);
  const auto count =
      add(add(prompt_steps, head ? 1 : 0), add(shape.generated, 1));
  if (count > policy.max_steps) throw std::length_error("sequence step budget");
  SequencePlan result{{}, {}, 0};
  result.steps.reserve(count);
  result.steps.insert(result.steps.end(), shape.prompt / shape.chunk,
                      SequenceStep::prefill);
  if (shape.prompt % shape.chunk) result.steps.push_back(SequenceStep::tail);
  if (head) result.steps.push_back(SequenceStep::initial_head);
  result.steps.insert(result.steps.end(), shape.generated,
                      SequenceStep::decode);
  result.steps.push_back(SequenceStep::finalize);
  std::size_t first = 0, events = 2;
  for (std::size_t i = 0; i < result.steps.size(); ++i) {
    const auto step = result.steps.at(i);
    const auto event_count =
        step == SequenceStep::decode
            ? 2
            : (step == SequenceStep::finalize ||
                       (step == SequenceStep::initial_head &&
                        shape.forward_only)
                   ? 0
                   : 1);
    events = add(events, static_cast<std::size_t>(event_count));
    const auto steps = i - first + 1;
    if (steps == policy.iterations_per_window ||
        step == SequenceStep::initial_head || i + 1 == result.steps.size()) {
      result.windows.push_back({first, steps, events});
      result.event_capacity = add(result.event_capacity, events);
      first = i + 1;
      events = 2;
    }
  }
  return result;
}
}  // namespace dolittle::kernels::metal
