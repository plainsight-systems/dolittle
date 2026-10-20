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
 * Sampling extension, header design and implementation, 2026-10-19:
 * cpp-guidelines NL.2 (#rl-comments-intent): state distribution invariants.
 * cpp-perf-guidelines CDSA.20: verify numeric order and signed-zero ties
 * against an independent comparison-based CDF. Source:
 * guidelines/cpu-dsa/CDSA.20-sort-integer-keys-not-records.md.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "inference/sampling/categorical.h"
#include "tests/support/check.h"
using dolittle::inference::sampling::Categorical;
using dolittle::inference::sampling::RandomStream;
using dolittle::inference::sampling::SelectionMode;
using dolittle::inference::sampling::Settings;
using dolittle::tests::check;
using dolittle::tests::expect_throws;
namespace {
// Independent contract model: order numeric logits, then integrate the declared
// FP64 categorical distribution. No encoded keys or sorting implementation
// assumptions enter the expected result.
std::uint32_t reference_select(const std::vector<float>& logits,
                               Settings settings, RandomStream& random) {
  std::vector<std::uint32_t> ids(logits.size());
  std::iota(ids.begin(), ids.end(), 0U);
  std::erase_if(ids, [&](auto id) { return !std::isfinite(logits.at(id)); });
  std::sort(ids.begin(), ids.end(), [&](auto a, auto b) {
    return logits.at(a) != logits.at(b) ? logits.at(a) > logits.at(b) : a < b;
  });
  if (settings.top_k && ids.size() > settings.top_k) ids.resize(settings.top_k);
  std::vector<double> weights;
  double total = 0;
  for (auto id : ids) {
    weights.push_back(
        std::exp((static_cast<double>(logits.at(id)) - logits.at(ids.front())) /
                 settings.temperature));
    total += weights.back();
  }
  double retained = weights.front();
  std::size_t count = 1;
  while (count < weights.size() && retained / total < settings.top_p) {
    retained += weights.at(count++);
  }
  const double u = static_cast<double>(random() >> 11) * 0x1.0p-53;
  double cumulative = 0;
  for (std::size_t i = 0; i < count; ++i) {
    cumulative += weights.at(i) / retained;
    if (u < cumulative) return ids.at(i);
  }
  return ids.at(count - 1);
}
void check_distribution_contract() {
  // Repeatable inputs exercise ties, signed zero, subnormals, masks and varying
  // finite magnitudes. Each call must preserve the declared CDF and RNG state.
  // NOLINTNEXTLINE(bugprone-random-generator-seed)
  RandomStream input(7391);
  for (const std::size_t size : {17U, 1023U, 4096U}) {
    Categorical sampler(size);
    std::vector<float> values(size);
    for (auto& value : values) {
      value =
          static_cast<float>(static_cast<int>(input() % 4001) - 2000) / 127.0F;
    }
    const std::array edges{0.0F,
                           -0.0F,
                           std::numeric_limits<float>::denorm_min(),
                           -std::numeric_limits<float>::denorm_min(),
                           -std::numeric_limits<float>::infinity(),
                           1.0F,
                           1.0F};
    std::copy(edges.begin(), edges.end(), values.begin());
    for (const auto settings :
         {Settings{SelectionMode::categorical, 1, 0, 1},
          Settings{SelectionMode::categorical, .6, 0, .95},
          Settings{SelectionMode::categorical, 2, 11, .7}}) {
      RandomStream actual(input()), expected = actual;
      for (int draw = 0; draw < 64; ++draw) {
        check(sampler.select(values, settings, actual) ==
                  reference_select(values, settings, expected),
              "sampling preserves ordered CDF for finite logits and masks");
        check(actual == expected, "sampling preserves one-draw RNG state");
      }
    }
    for (std::size_t i = 0; i < size; ++i) values.at(i) = i % 2 ? -0.0F : 0.0F;
    RandomStream actual(input()), expected = actual;
    const Settings tied{SelectionMode::categorical, 1, 0, .5};
    for (int draw = 0; draw < 64; ++draw) {
      check(sampler.select(values, tied, actual) ==
                reference_select(values, tied, expected),
            "signed-zero ties retain ascending token ID");
    }
  }
}
}  // namespace
int main() try {
  check_distribution_contract();
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
