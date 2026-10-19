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
#include <limits>

#include "inference/generation.h"
#include "tests/support/check.h"
using dolittle::inference::Action;
using dolittle::inference::next_action;
using dolittle::inference::Progress;
using dolittle::inference::StopReason;
using dolittle::tests::check;
using dolittle::tests::expect_throws;
int main() try {
  check(next_action({3, 0, 0, 2, false, true}).action == Action::consume_prompt,
        "stop still consumes prompt");
  check(next_action({3, 0, 3, 2, false, false}).action == Action::sample,
        "sample after prompt");
  for (bool stop : {false, true}) {
    check(
        next_action({3, 1, 3, 1, true, stop}).action == Action::consume_pending,
        "final stop token needs consumption");
    check(
        next_action({3, 1, 4, 1, true, stop}).reason == StopReason::stop_token,
        "stop ID precedence");
    check(next_action({3, 1, 4, 1, false, stop}).reason ==
              StopReason::token_limit,
          "budget precedence");
  }
  check(next_action({3, 1, 4, 2, false, true}).reason == StopReason::user_stop,
        "graceful stop after consumption");
  check(
      next_action({3, 0, 3, 0, false, false}).reason == StopReason::token_limit,
      "zero generation budget");
  for (auto bad :
       {Progress{0, 0, 0, 0, false, false}, Progress{3, 2, 3, 2, false, false},
        Progress{3, 1, 2, 2, false, false}, Progress{3, 0, 3, 2, true, false},
        Progress{3, 0, 4, 2, false, false},
        Progress{3, 3, 5, 2, false, false}}) {
    expect_throws<std::invalid_argument>([&] { (void)next_action(bad); });
  }
  expect_throws<std::invalid_argument>([] {
    (void)next_action(
        {std::numeric_limits<std::size_t>::max(), 1, 0, 1, false, false});
  });
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
