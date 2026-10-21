/*
 * Metal implementation of the shared target-generation operations.
 *
 * Axis X5: reconstruction difference and selected-row residual update.
 * Inputs original/edited reconstruction vectors a[D],b[D]; target residual
 * h[B,D], FP32 direction u[D], strength alpha>=0, selected position p and
 * device position S. Output independent FP32 direction or one modified row,
 * plus device status/count. All other residual rows and source captures stay
 * bitwise unchanged. Extents and non-aliasing are checked before recording.
 *
 * Algorithm: steering.md, Algorithm and numerical boundaries.
 * 1. d[i]=FP32(b[i]-a[i]). Eight lanes sum separately rounded FP32 squares
 *    at i=lane+8*k. Combine lanes 0..7 left-to-right and take FP32 sqrt.
 *    Require finite nonzero norm, then u[i]=FP32(d[i]/norm).
 * 2. During replay, if p is outside [S,S+B), return without mutation. Otherwise
 *    row=p-S. Compute L2(h[row]) with that same eight-lane reduction.
 * 3. beta=FP32(alpha*norm); product[i]=FP32(beta*u[i]); then
 *    h[row,i]=round_storage(FP32(FP32(h[row,i])+product[i])). Separate the
 *    multiplication and addition. Validate values and increment application
 *    count once. Invalid status gates later sampling on the GPU.
 *
 * Mapping: one group per vector/update; eight reduction owners and coalesced
 * output lanes. Status/count use integer words. No host branch selects the row.
 * Cost O(D), bounded shared sums and preallocated direction. Optimization:
 * keep the direction reusable and independent of AR/target transient arenas.
 *
 * Header-design consultation, 2026-10-21: both local corpora searched and
 * cited rules read. Core I.7 (#ri-post) fixes numerical/completion results;
 * SL.con.3 (#rsl-bounds) requires checked extents before native indexing;
 * R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.1: GPU position, norm and mutation enforce the execution boundary.
 * guidelines/gpu/GPU.1-keep-data-on-device.md
 * GDSA.2: fixed reductions and separate products preserve numerical order.
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md
 */

#include "common.metal"
kernel void direction_difference(constant OperationArgs& a [[buffer(0)]],
                                 uint t [[thread_index_in_threadgroup]]) {
  threadgroup float sums[8], norm;
  const ulong width = a.dimensions[0];
  auto status = (device uint*)a.address[2];
  auto output = (device float*)(a.address[2] + 8);
  if (t < 8) {
    float sum = 0;
    for (ulong i = t; i < width; i += 8) {
      float delta =
          load(a.address[1], i, a.type) - load(a.address[0], i, a.type);
      sum += delta * delta;
    }
    sums[t] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!t) {
    float sum = 0;
    for (uint i = 0; i < 8; ++i) sum += sums[i];
    norm = precise::sqrt(sum);
    *status = !isfinite(norm) || norm == 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (!*status)
    for (ulong i = t; i < width; i += 256)
      output[i] =
          (load(a.address[1], i, a.type) - load(a.address[0], i, a.type)) /
          norm;
}
kernel void intervention_reset(constant OperationArgs& a [[buffer(0)]]) {
  auto state = (device uint*)a.address[2];
  state[0] = 0;
  state[1] = 0;
}
kernel void intervention_apply(constant OperationArgs& a [[buffer(0)]],
                               uint t [[thread_index_in_threadgroup]]) {
  const ulong first = position(a), selected = a.first, width = a.dimensions[0];
  if (selected < first || selected - first >= a.dimensions[1]) return;
  auto state = (device uint*)a.address[2];
  const ulong row = (selected - first) * width;
  threadgroup float sums[8], beta;
  threadgroup atomic_uint invalid;
  if (!t) {
    atomic_store_explicit(&invalid, state[0] || state[1] != 0,
                          memory_order_relaxed);
    ++state[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 8) {
    float sum = 0;
    for (ulong i = t; i < width; i += 8) {
      float v = load(a.address[0], row + i, a.type);
      sum += v * v;
    }
    sums[t] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!t) {
    float sum = 0;
    for (uint i = 0; i < 8; ++i) sum += sums[i];
    beta = a.scalar * precise::sqrt(sum);
    if (!isfinite(beta))
      atomic_store_explicit(&invalid, 1, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool valid = atomic_load_explicit(&invalid, memory_order_relaxed) == 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (valid)
    for (ulong i = t; i < width; i += 256) {
      const float value =
          rounded(load(a.address[0], row + i, a.type) +
                      beta * ((device const float*)a.address[1])[i],
                  a.type);
      if (!isfinite(value))
        atomic_store_explicit(&invalid, 1, memory_order_relaxed);
      store(a.address[0], row + i, value, a.type);
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!t) state[0] = atomic_load_explicit(&invalid, memory_order_relaxed);
}
