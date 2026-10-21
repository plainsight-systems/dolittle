/*
 * D3 pure bounded sequence planning. Check complete non-overlapping publication
 * coverage, first-head visibility, forward-only behavior and rejected overflow.
 * No device/runtime workflow. Core I.7 (#ri-post), ES.103 (#res-overflow),
 * performance GPU.6/GPU.7 (guidelines/gpu/GPU.6-batch-tiny-gpu-work.md and
 * guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md), read for implementation,
 * require bounded setup and independent output windows.
 */
#include <algorithm>
#include <limits>

#include "kernels/metal/sequence_plan.h"
#include "tests/support/check.h"
namespace tests = dolittle::tests;
int main() {
  try {
    using dolittle::kernels::metal::plan_sequence;
    using dolittle::kernels::metal::SequenceStep;
    for (const std::size_t prompt : {1, 8, 9, 16}) {
      const auto plan = plan_sequence({prompt, 7, 8}, {4, 100});
      tests::check(std::ranges::count(plan.steps, SequenceStep::decode) == 7,
                   "decode slots cover admitted budget");
      tests::check(plan.steps.back() == SequenceStep::finalize,
                   "terminal finalization");
      std::size_t next = 0, capacity = 0;
      for (const auto window : plan.windows) {
        tests::check(window.first_step == next && window.step_count > 0 &&
                         window.step_count <= 4,
                     "windows cover timeline without gaps");
        next += window.step_count;
        capacity += window.event_capacity;
        for (std::size_t i = window.first_step; i < next; ++i) {
          if (plan.steps.at(i) == SequenceStep::initial_head) {
            tests::check(i + 1 == next, "first output publishes immediately");
          }
        }
      }
      tests::check(next == plan.steps.size() && capacity == plan.event_capacity,
                   "complete publication capacity");
    }
    const auto ar = plan_sequence({9, 0, 8, true}, {4, 100});
    tests::check(
        std::ranges::count(ar.steps, SequenceStep::initial_head) == 1 &&
            std::ranges::count(ar.steps, SequenceStep::decode) == 0,
        "forward-only executes head without decode");
    tests::expect_throws<std::invalid_argument>(
        [] { (void)plan_sequence({0, 1, 8}, {4, 100}); });
    tests::expect_throws<std::invalid_argument>(
        [] { (void)plan_sequence({1, 1, 8, true}, {4, 100}); });
    tests::expect_throws<std::length_error>(
        [] { (void)plan_sequence({10, 100, 8}, {4, 100}); });
    tests::expect_throws<std::length_error>([] {
      (void)plan_sequence({std::numeric_limits<std::size_t>::max(), 1, 8},
                          {4, 100});
    });
  } catch (const std::exception& error) {
    return tests::report_failure(error.what());
  }
}
