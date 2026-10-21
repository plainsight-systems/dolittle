/*
 * Implemented foundation, 2026-10-21; header design approved before
 * implementation. Axis X1/X4: pure preparation of Metal's bounded sequence
 * timeline. Contract: P=prompt>0, G=generated>=0, B=chunk>0. forward_only
 * requires G=0. Policy bounds iterations per publication window and total
 * encoded steps. Invalid shapes/policy -> invalid_argument; checked size
 * overflow or an exceeded admission bound -> length_error; bad_alloc
 * propagates. The result owns host metadata only. No Metal call or generated
 * token enters this unit.
 *
 * Algorithm: metal-port.md, Metal scheduling; execution-boundary.md, sections
 * 3-4. List floor(P/B) full prompt iterations and one P%B tail if nonzero.
 * Append the initial head when G>0 or forward_only. Append G decode iterations
 * for ordinary generation. Each decode consumes one pending sampled token,
 * then runs its selection tail only if device state permits another token.
 * Append finalization even when no head or decode was required.
 *
 * Each iteration references a reusable body, not copies of its layer commands.
 * Partition the ordered list into windows of at most policy.iterations_per_
 * window; force a boundary after the initial head so first selection does not
 * wait for subsequent decode. Finalization belongs to the last window. GPU
 * control can make scheduled bodies empty; the CPU schedule never decides EOS,
 * sampling, cancellation or the next absolute position.
 *
 * Reserve output slots per window for its maximum events: one consumed event
 * per prompt iteration, one selected per initial head (none for forward_only),
 * two per decode iteration, plus injection/terminal allowance. A shared window
 * starts with a count and contains only completed immutable Event records.
 * Give each window separate shared backing: the CPU never accesses a resource
 * that a later GPU window writes. Count alignment, padding, private journal,
 * snapshots, allocator bytes and native command objects in admission, not just
 * Event payload. The final window must publish terminal status even after
 * empty bodies; earlier terminal publication is idempotent.
 *
 * Cost: O(ceil(P/B)+G) schedule metadata and O(P+G) bounded event storage;
 * native ICB bodies are O(recipe). Optimization: reuse the same forward bodies
 * and stable tensor bindings while amortizing publication over several steps.
 * This bound differs from CUDA's O(recipe) conditional graph. Values remain
 * device-owned regardless of this host-prepared schedule's size.
 *
 * Both local corpora queried and rules read for header design:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post), ES.103 (#res-overflow):
 * explicit schedule bounds and checked event/iteration arithmetic.
 * cpp-perf-guidelines GPU.6 (guidelines/gpu/GPU.6-batch-tiny-gpu-work.md):
 * reusable bodies; GPU.7 (guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md):
 * independent completed publication resources; GDSA.4
 * (guidelines/gpu-dsa/GDSA.4-no-cross-workgroup-waits-without-forward-progress.md):
 * dependencies between dispatches, no grid-wide spin waiting.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "inference/decoding/device.h"

namespace dolittle::kernels::metal {
enum class SequenceStep : std::uint8_t {
  prefill,
  tail,
  initial_head,
  decode,
  finalize
};
struct PublicationPolicy {
  std::size_t iterations_per_window;
  std::size_t max_steps;
};
struct PublicationWindow {
  std::size_t first_step;
  std::size_t step_count;
  std::size_t event_capacity;
};
struct SequencePlan {
  std::vector<SequenceStep> steps;
  std::vector<PublicationWindow> windows;
  std::size_t event_capacity;
};
[[nodiscard]] SequencePlan plan_sequence(inference::DeviceSequence::Shape,
                                         PublicationPolicy);
}  // namespace dolittle::kernels::metal
