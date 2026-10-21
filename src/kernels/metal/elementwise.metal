/*
 * Implementation: prepared Metal target kernels.
 *
 * Axis X5: residual add, SwiGLU and fixed-row gather. Matching dense shapes,
 * BF16/FP32 storage, distinct output and checked extents are required.
 *
 * Algorithm: transformer-execution.md, One Qwen3 transformer block.
 * Add: widen a,b, compute FP32(a+b), then round to storage once.
 * SwiGLU: e=exp(-abs(gate)); sigmoid is 1/(1+e) for nonnegative gate
 * and e/(1+e) otherwise. Multiply gate*sigmoid in FP32, then round that
 * SiLU result to storage BEFORE multiplying by up. Keep division and the
 * gate multiplication separate; gate/(1+e) changes rounding.
 * Write round_storage(FP32(rounded_silu*FP32(up))). No implicit fused stage
 * may remove this intermediate rounding. Preserve finite/nonfinite handling
 * under strict shader math; use the frozen fixtures to check transcendentals.
 * Gather: copy the prepared strictly increasing row indices bit-for-bit;
 * indices are immutable metadata, not host-extracted device control.
 *
 * Mapping: lanes own contiguous 4-element vectors with scalar tails. Add and
 * SwiGLU need no shared memory or global scratch. Gather can use X4 D2D copies
 * in diagnostic execution; recorded bodies use its existing GPU copy command.
 * Cost O(elements) reads/writes. Optimization: one pass for SiLU and multiply
 * while preserving its rounding; do not fuse residual add across an advertised
 * capture/steering site. All producer values needed by dynamic hooks survive.
 *
 * Header-design and implementation consultation, 2026-10-21: both local corpora
 * searched and cited rules read. Core I.7 (#ri-post) fixes numerical/completion
 * results; SL.con.3 (#rsl-bounds) requires checked extents before native
 * indexing; R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.2: vectorized contiguous element traffic.
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md
 * GPU.6: fuse SiLU/multiply without erasing the rounding boundary.
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md
 * GDSA.2: explicit scalar arithmetic remains independent of scheduling.
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md
 */

#include "common.metal"
#include "elementary.metal"
kernel void elementwise_add(constant OperationArgs& a [[buffer(0)]],
                            uint i [[thread_position_in_grid]]) {
  if (i < a.dimensions[0])
    store(a.address[2], i,
          load(a.address[0], i, a.type) + load(a.address[1], i, a.type),
          a.type);
}
kernel void silu_multiply(constant OperationArgs& a [[buffer(0)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= a.dimensions[0]) return;
  const float x = load(a.address[0], i, a.type);
  const float exponential = exp_nonpositive(x >= 0 ? -x : x);
  const float sigmoid =
      x >= 0 ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential);
  const float silu = rounded(x * sigmoid, a.type);
  store(a.address[2], i, silu * load(a.address[1], i, a.type), a.type);
}
