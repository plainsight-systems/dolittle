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
 * ordering. Optimization: encode each finite FP32 logit's complete bit pattern
 * as a descending unsigned key; both signed zeros share one key. Four stable
 * 8-bit radix passes preserve the initial ascending ID order on equal logits.
 * Build all histograms in one scan and skip digits shared by every candidate.
 * Small sets (<512 candidates) retain comparison sorting to avoid histogram
 * setup. Two vocabulary-reserved candidate buffers provide O(V) scratch; large
 * sorts take O(V) work without altering any probability or RNG arithmetic.
 * No std::discrete_distribution dependency on vendor mapping.
 * Corpus consultation, header design and implementation, 2026-10-19:
 * cpp-guidelines NL.2
 * (#rl-comments-intent): document exact ordering, including signed zero.
 * cpp-perf-guidelines CDSA.20: use exact integer float keys, stable passes and
 * reusable scratch; these small records can be sorted directly. Source:
 * guidelines/cpu-dsa/CDSA.20-sort-integer-keys-not-records.md.
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
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bounds.h"

namespace dolittle::inference::sampling {
namespace {
constexpr std::size_t radix_min_candidates = 512;
constexpr unsigned digit_bits = 8;
constexpr unsigned digit_count = 4;
constexpr std::size_t bucket_count = 1U << digit_bits;
static_assert(std::numeric_limits<float>::is_iec559 &&
              sizeof(float) == sizeof(std::uint32_t));
// Keep all 32 bits. Signed zeros compare equal in the public contract.
std::uint32_t descending_key(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value == 0 ? 0.0F : value);
  const auto mask = (0U - (bits >> 31)) | 0x80000000U;
  return ~(bits ^ mask);
}
std::size_t digit(std::uint32_t key, unsigned index) {
  return (key >> (index * digit_bits)) & (bucket_count - 1);
}
}  // namespace

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
    std::uint32_t key;
    double value;
  };
  std::size_t vocabulary;
  std::vector<Candidate> candidates;
  std::vector<Candidate> scratch;
  explicit Impl(std::size_t n) : vocabulary(n) {
    candidates.reserve(n);
    scratch.reserve(n);
  }
  void sort_candidates() {
    if (candidates.size() < radix_min_candidates) {
      std::sort(candidates.begin(), candidates.end(),
                [](const Candidate& a, const Candidate& b) {
                  return a.key != b.key ? a.key < b.key : a.id < b.id;
                });
      return;
    }
    std::array<std::array<std::size_t, bucket_count>, digit_count> counts{};
    for (const auto& candidate : candidates) {
      for (unsigned d = 0; d < digit_count; ++d) {
        ++counts.at(d).at(digit(candidate.key, d));
      }
    }
    scratch.resize(candidates.size());
    for (unsigned d = 0; d < digit_count; ++d) {
      auto& offsets = counts.at(d);
      if (offsets.at(digit(candidates.front().key, d)) == candidates.size()) {
        continue;
      }
      std::size_t offset = 0;
      for (auto& bucket : offsets) {
        offset += std::exchange(bucket, offset);
      }
      for (const auto& candidate : candidates) {
        scratch.at(offsets.at(digit(candidate.key, d))++) = candidate;
      }
      candidates.swap(scratch);
    }
  }
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
      const auto value = dolittle::at(logits, i);
      c.push_back(
          {static_cast<std::uint32_t>(i), descending_key(value), value});
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
  impl_->sort_candidates();
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
