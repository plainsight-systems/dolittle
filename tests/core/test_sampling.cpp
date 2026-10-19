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
#include <array>
#include <limits>

#include "inference/sampling/categorical.h"
#include "tests/support/check.h"
using dolittle::inference::sampling::Categorical;
using dolittle::inference::sampling::RandomStream;
using dolittle::inference::sampling::SelectionMode;
using dolittle::inference::sampling::Settings;
using dolittle::tests::check;
using dolittle::tests::expect_throws;
int main() try {
  Categorical sampler(4);
  // Fixed seeds make the sampling contract repeatable; this is not entropy
  // generation. NOLINTNEXTLINE(bugprone-random-generator-seed)
  RandomStream rng(42), untouched = rng;
  const float masked = -std::numeric_limits<float>::infinity();
  const std::array logits{masked, 3.f, 3.f, 1.f};
  const Settings greedy{SelectionMode::greedy, 1, 0, 1};
  check(sampler.select(logits, greedy, rng) == 1 && rng == untouched,
        "greedy tie and RNG contract");
  Settings categorical{SelectionMode::categorical, 1, 1, 1};
  auto once = rng;
  (void)once();
  check(sampler.select(logits, categorical, rng) == 1 && rng == once,
        "top-k retains lowest tied ID and draws once");
  categorical.top_k = 0;
  categorical.top_p = .25;
  check(sampler.select(logits, categorical, rng) == 1,
        "nucleus keeps at least one candidate");
  const std::array uniform{0.f, 0.f, 0.f, 0.f};
  categorical.top_p = .5;
  // Fixed seeds make the sampling contract repeatable; this is not entropy
  // generation. NOLINTNEXTLINE(bugprone-random-generator-seed)
  RandomStream reference(51);
  rng = reference;
  const double u = static_cast<double>(reference() >> 11) * 0x1.0p-53;
  check(sampler.select(uniform, categorical, rng) == (u < .5 ? 0u : 1u),
        "inclusive nucleus boundary and CDF mapping");
  check(rng == reference, "one draw per categorical selection");
  untouched = rng;
  for (const auto bad :
       std::array{Settings{SelectionMode::categorical, 0, 0, 1},
                  Settings{SelectionMode::categorical, 1, 5, 1},
                  Settings{SelectionMode::categorical, 1, 0, 0},
                  Settings{SelectionMode::greedy, .5, 0, 1}}) {
    expect_throws<std::invalid_argument>(
        [&] { (void)sampler.select(logits, bad, rng); });
  }
  auto bad = logits;
  bad.at(2) = std::numeric_limits<float>::quiet_NaN();
  expect_throws<std::invalid_argument>(
      [&] { (void)sampler.select(bad, greedy, rng); });
  bad.fill(masked);
  expect_throws<std::invalid_argument>(
      [&] { (void)sampler.select(bad, greedy, rng); });
  check(rng == untouched, "invalid input must not advance RNG");
  // Fixed seeds make the sampling contract repeatable; this is not entropy
  // generation. NOLINTNEXTLINE(bugprone-random-generator-seed)
  RandomStream a(9), b(9), unrelated(8);
  categorical.top_p = 1;
  for (int i = 0; i < 50; ++i) {
    (void)sampler.select(uniform, categorical, unrelated);
    check(sampler.select(logits, categorical, a) ==
              sampler.select(logits, categorical, b),
          "independent seeded streams");
  }
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
