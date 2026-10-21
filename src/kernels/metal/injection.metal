/*
 * Metal implementation of the shared target-generation operations.
 *
 * Axis X5: trained NLA normalization. Input h[D] in BF16/FP32, finite scale;
 * output z[D] in the declared BF16/FP32 type, FP32 source norm and status.
 * Contract: source is complete, output disjoint, all extents admitted. Zero or
 * nonfinite norm/source/output latches failure and forbids AV execution.
 *
 * Algorithm: execution-boundary.md, H2. Eight lanes visit i=lane+8*k ascending,
 * separately round FP32 h[i]*h[i] and sum. Lane zero adds the eight partials
 * left-to-right and computes FP32 sqrt. Broadcast norm; compute
 * z[i]=round_storage(FP32(FP32(h[i]/norm)*scale)). Separate divide/multiply,
 * then one storage cast. Validate source, norm, intermediate and stored output
 * on GPU; publish status only after all lane checks complete.
 *
 * Metal mapping: one group, eight reduction owners, cooperative contiguous
 * output. Shared storage is eight FP32 partials plus scalar norm/status.
 * Barriers separate reduction, validation and stores; integer status reduction
 * is associative. Cost O(D); no temporary device-wide tensor. Optimization:
 * retain normalized output directly for the marker replacement consumer.
 *
 * Header-design consultation, 2026-10-21: both local corpora searched and
 * cited rules read. Core I.7 (#ri-post) fixes numerical/completion results;
 * SL.con.3 (#rsl-bounds) requires checked extents before native indexing;
 * R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.1: normalization/status remain GPU-owned.
 * guidelines/gpu/GPU.1-keep-data-on-device.md
 * GPU.2: cooperative contiguous output stores.
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md
 * GDSA.2: eight partials and ordered final fold match the trained law.
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md
 */

#include "common.metal"
kernel void injection_normalize(constant OperationArgs& a [[buffer(0)]],
                                uint t [[thread_index_in_threadgroup]]) {
  threadgroup float sums[8], norm;
  threadgroup atomic_uint invalid;
  const ulong width = a.dimensions[0];
  if (!t) atomic_store_explicit(&invalid, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 8) {
    float sum = 0;
    for (ulong i = t; i < width; i += 8) {
      float value = load(a.address[0], i, a.type);
      if (!isfinite(value))
        atomic_store_explicit(&invalid, 1, memory_order_relaxed);
      sum += value * value;
    }
    sums[t] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!t) {
    float sum = 0;
    for (uint i = 0; i < 8; ++i) sum += sums[i];
    norm = precise::sqrt(sum);
    *((device float*)a.address[1]) = norm;
    if (!isfinite(norm) || norm == 0)
      atomic_store_explicit(&invalid, 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool valid = atomic_load_explicit(&invalid, memory_order_relaxed) == 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (valid)
    for (ulong i = t; i < width; i += 256) {
      const float value = (load(a.address[0], i, a.type) / norm) * a.scalar;
      const float result = rounded(value, a.flags);
      if (!isfinite(result))
        atomic_store_explicit(&invalid, 1, memory_order_relaxed);
      store(a.address[1] + 8, i, result, a.flags);
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!t)
    *((device uint*)(a.address[1] + 4)) =
        atomic_load_explicit(&invalid, memory_order_relaxed);
}
