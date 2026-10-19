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
 * ordering. One reusable candidate buffer, O(V) scratch and O(V log V) sorting
 * in the baseline. No std::discrete_distribution dependency on vendor mapping.
 * Optimization: reuse scratch; measure partial selection before replacing the
 * full sort, preserving exactly the retained candidates and order.
 * Invalid inputs or allocation failure leave the RNG unchanged. Commit the RNG
 * advance only after all validation/preparation succeeds. Objects are serial,
 * not reentrant; moved-from objects permit only destruction/reassignment.
 * Overlay: docs/architecture/file-mapping.md, "Sampling without a class hierarchy".
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): tie/filter/draw definitions
 * and failure effects are public, not unspecified library behavior.
 * cpp-perf-guidelines MEM.1: reserve vocabulary-bounded scratch once; candidate
 * records share one draw lifetime and need no individual allocations.
 * Source: guidelines/memory/MEM.1-arena-allocator-for-frame-scope.md.
 */
#pragma once

#include "inference/sampling/sampler.h"
#include <memory>
#include <span>

namespace dolittle::inference::sampling {
class Categorical {
  public:
    explicit Categorical(std::size_t vocabulary_size);
    ~Categorical();
    Categorical(Categorical &&) noexcept;
    Categorical &operator=(Categorical &&) noexcept;
    Categorical(const Categorical &) = delete;
    Categorical &operator=(const Categorical &) = delete;
    [[nodiscard]] std::uint32_t select(std::span<const float> logits, Settings settings,
                                       RandomStream &random);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace dolittle::inference::sampling
