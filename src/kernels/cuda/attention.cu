/*
 * Implementation-file header design only; no CUDA bodies yet.
 * Axis X5, CUDA full-sequence causal grouped-query attention.
 * Algorithm: transformer overlay, "Attention and positional semantics":
 * 1. Assign a query row/head (or bounded tile) to a block; map its KV head by
 *    floor(h/(Hq/Hkv)). Keep Q and a per-query accumulator on chip.
 * 2. Stream K/V tiles, visiting only j<=p. Compute FP32 scaled dot products.
 * 3. Carry live flag, maximum m, denominator l and vector accumulator a. For a
 *    new tile maximum m_new, rescale old l/a by exp(m-m_new), add exp(s-m_new)
 *    and its weighted values, then update m. Skip fully masked tiles rather than
 *    subtracting infinities. Every valid unpadded query has at least one key.
 * 4. Store a/l once, rounding only at output storage; no materialized probabilities.
 * Design and cost: O(N^2*Hq*d) work, O(d) carried state per query and bounded
 * on-chip K/V tile storage. There is no N*N global score buffer or expanded KV
 * head copy. Resource limits may require smaller tiles or explicit rejection.
 * Optimization: stage reused contiguous K/V feature tiles; tile capacity and
 * occupancy are selected within device limits. No global partial-sum atomics or
 * cross-block waits. Softmax/dot reduction order can differ from the reference
 * SDPA backend; compare with justified bounds, not bitwise identity.
 * Native and forced-PTX runs exercise the same essential numerical scenario and
 * are counted honestly under the integration-test policy.
 * Contract: implement the matching operation in kernels/operations.h for dense
 * BF16/FP32 storage and FP32 accumulation. Preparation rejects unsupported shapes,
 * overflowing launch extents or insufficient device resources before submission.
 * Operands/workspace are borrowed from the prepared plan through queue completion;
 * no allocation, host readback or host wait in a kernel launch. Submission errors
 * follow operations.h; deferred device faults surface at the explicit queue wait.
 * Every address and tail lane is bounded; no cross-block spin waits. Kernel
 * selection and arithmetic policy are recorded by the prepared implementation.
 * Bodies, launch geometry and GPU validation are pending; these are design choices,
 * not measured performance or evidence of numerical equivalence.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): equations, rounding and completion are explicit.
 * cpp-perf-guidelines GDSA.16: stream bounded tiles with small online state.
 * Source: guidelines/gpu-dsa/GDSA.16-tile-through-on-chip-memory.md.
 * cpp-perf-guidelines GPU.2: map adjacent lanes to adjacent data.
 * Source: guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 */
#include "kernels/operations.h"
