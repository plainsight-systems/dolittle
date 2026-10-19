/*
 * Implementation of the target-generation contract.
 * Axis R1: temperature/top-k/top-p categorical selection and greedy mode.
 * Input: immutable FP32 logits[V], settings and a mutable independent RNG.
 * Output: one uint32 token ID; logits are never mutated. V is positive and
 * fits token IDs. Temperature is positive finite; top_p is in (0,1]; top_k
 * is zero or <= V. Greedy requires temperature=1, top_k=0, top_p=1 so ignored
 * settings cannot silently misrepresent a recorded run.
 *
 * Algorithm:
 * 1. Validate inputs and collect non-masked (finite) candidates. For greedy,
 *    return the highest logit, choosing the smallest ID on an exact tie.
 * 2. For categorical mode, order candidates by descending logit, then ascending
 *    ID. Retain the first top_k candidates when top_k is nonzero.
 * 3. Compute exp((logit - maximum_logit)/temperature) in FP64 and normalize
 *    by the sum, keeping the largest weight equal to one before normalization.
 * 4. Retain the smallest ordered prefix whose cumulative probability is >=
 *    top_p, always keeping at least one candidate. Normalize that prefix again.
 * 5. Draw exactly one uint64 from mt19937_64. Set u=(draw >> 11)*2^-53, in
 *    [0,1). Choose the first cumulative interval strictly greater than u;
 *    assign any final rounding remainder to the last retained candidate.
 * Ordering, boundary inclusion and RNG conversion are versioned semantics;
 * changing them can change the sampled trace even with identical logits.
 *
 * Implementation design: FP64 host weights/cumulative sums and stable ID tie
 * ordering. One reusable candidate buffer, O(V) scratch and O(V log V) sorting
 * in the baseline. No std::discrete_distribution dependency on vendor mapping.
 * Optimization: reuse scratch; measure partial selection before replacing the
 * full sort, preserving exactly the retained candidates and order.
 * Invalid inputs or allocation failure leave the RNG unchanged. Commit the RNG
 * advance only after all validation/preparation succeeds. Objects are serial,
 * not reentrant; moved-from objects permit only destruction/reassignment.
 * Overlay: docs/architecture/file-mapping.md, "Sampling without a class
 * hierarchy". Corpus consultation, implementation, 2026-10-18: cpp-guidelines
 * I.1 (#ri-explicit), I.7 (#ri-post): tie/filter/draw definitions and failure
 * effects are public, not unspecified library behavior. cpp-perf-guidelines
 * MEM.1: reserve vocabulary-bounded scratch once; candidate records share one
 * draw lifetime and need no individual allocations. Source:
 * guidelines/memory/MEM.1-arena-allocator-for-frame-scope.md.
 */
#include "inference/sampling/categorical.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bounds.h"

namespace dolittle::inference::sampling {
void validate_settings(Settings s) {
  if ((s.mode != SelectionMode::greedy &&
       s.mode != SelectionMode::categorical) ||
      !std::isfinite(s.temperature) || s.temperature <= 0 ||
      !std::isfinite(s.top_p) || s.top_p <= 0 || s.top_p > 1 ||
      (s.mode == SelectionMode::greedy &&
       (s.temperature != 1 || s.top_k != 0 || s.top_p != 1))) {
    throw std::invalid_argument("invalid sampling settings");
  }
}
struct Categorical::Impl {
  struct Candidate {
    std::uint32_t id;
    double value;
  };
  std::size_t vocabulary;
  std::vector<Candidate> candidates;
  explicit Impl(std::size_t n) : vocabulary(n) { candidates.reserve(n); }
};
Categorical::Categorical(std::size_t n) {
  if (!n || n - 1 > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("invalid sampling vocabulary");
  }
  impl_ = std::make_unique<Impl>(n);
}
Categorical::~Categorical() = default;
Categorical::Categorical(Categorical&&) noexcept = default;
Categorical& Categorical::operator=(Categorical&&) noexcept = default;
std::uint32_t Categorical::select(std::span<const float> logits, Settings s,
                                  RandomStream& rng) {
  validate_settings(s);
  if (logits.size() != impl_->vocabulary || s.top_k > logits.size()) {
    throw std::invalid_argument("sampling shape or top-k mismatch");
  }
  auto& c = impl_->candidates;
  c.clear();
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (std::isnan(dolittle::at(logits, i)) ||
        dolittle::at(logits, i) == std::numeric_limits<float>::infinity()) {
      throw std::invalid_argument("nonfinite sampling logit");
    }
    if (std::isfinite(dolittle::at(logits, i))) {
      c.push_back({static_cast<std::uint32_t>(i), dolittle::at(logits, i)});
    }
  }
  if (c.empty()) {
    throw std::invalid_argument("all sampling candidates masked");
  }
  const auto order = [](const auto& a, const auto& b) {
    return a.value != b.value ? a.value > b.value : a.id < b.id;
  };
  if (s.mode == SelectionMode::greedy) {
    return std::min_element(c.begin(), c.end(), order)->id;
  }
  std::sort(c.begin(), c.end(), order);
  if (s.top_k && c.size() > s.top_k) {
    c.resize(s.top_k);
  }
  const double maximum = c.front().value;
  double total = 0;
  for (auto& candidate : c) {
    candidate.value = std::exp((candidate.value - maximum) / s.temperature);
    total += candidate.value;
  }
  double retained = 0;
  std::size_t count = 0;
  while (true) {
    retained += c.at(count++).value;
    if (!(count < c.size() && retained / total < s.top_p)) {
      break;
    }
  }
  const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;
  double cumulative = 0;
  for (std::size_t i = 0; i < count; ++i) {
    cumulative += c.at(i).value / retained;
    if (u < cumulative) {
      return c.at(i).id;
    }
  }
  return c.at(count - 1).id;
}
}  // namespace dolittle::inference::sampling
