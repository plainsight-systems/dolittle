/*
 * Target-generation interface and implementation contract.
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
 * hierarchy". Corpus consultation, header design and implementation,
 * 2026-10-18: cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post):
 * tie/filter/draw definitions and failure effects are public, not unspecified
 * library behavior. cpp-perf-guidelines MEM.1: reserve vocabulary-bounded
 * scratch once; candidate records share one draw lifetime and need no
 * individual allocations. Source:
 * guidelines/memory/MEM.1-arena-allocator-for-frame-scope.md.
 */
#pragma once

#include <memory>
#include <span>

#include "inference/sampling/sampler.h"

namespace dolittle::inference::sampling {
class Categorical {
 public:
  explicit Categorical(std::size_t vocabulary_size);
  ~Categorical();
  Categorical(Categorical&&) noexcept;
  Categorical& operator=(Categorical&&) noexcept;
  Categorical(const Categorical&) = delete;
  Categorical& operator=(const Categorical&) = delete;
  [[nodiscard]] std::uint32_t select(std::span<const float> logits,
                                     Settings settings, RandomStream& random);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::inference::sampling
