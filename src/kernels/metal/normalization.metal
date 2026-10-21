/*
 * Implementation: prepared Metal target kernels.
 *
 * Axis X5: RMSNorm for residual rows and Q/K heads. Inputs X[R,D], W[D],
 * epsilon>0; output Y[R,D]. Storage is BF16 or FP32. No input/output alias.
 *
 * Algorithm: transformer-execution.md, One Qwen3 transformer block.
 * 1. Widen X and square each value with a separately rounded FP32 multiply.
 *    Lane t has four sums. For D>128, component c visits 4*t+c+128*k in
 *    ascending k; for D<=128 it visits t+32*c if in range. Add in FP32.
 * 2. Sum each lane's four components left-to-right. Merge lanes using shuffle
 *    offsets 1,2,4,8,16. The lane-zero total defines the row's sum of squares.
 * 3. mean=FP32(total*FP32(1/D)); inverse=FP32(rsqrt(FP32(mean+epsilon))).
 * 4. normalized=round_storage(FP32(X*inverse)); then
 *    Y=round_storage(FP32(normalized*W)). Do not remove the first rounding.
 *
 * Metal mapping: one SIMD group owns the fixed reduction. For residual widths
 * that fit, a 128/256-thread group cooperatively stages D FP32 values, then
 * distributes the original 128 component sums. Groups of four combine them;
 * one SIMD group finishes the tree. A barrier publishes inverse and all lanes
 * write adjacent features. Shared bytes=4*(D+32); stage only when native limits
 * permit. Wider rows reread X for output and keep reduction sums in registers.
 * A head-width-128 path uses one SIMD group and no whole-row staging.
 *
 * Cost O(R*D), no global workspace. Optimization: cooperative traffic without
 * changing the numerical tree. Input staging is reused for output; no CPU
 * norm or scalar upload. Same row bits must survive changes in batch/chunking.
 *
 * Header-design and implementation consultation, 2026-10-21: both local corpora
 * searched and cited rules read. Core I.7 (#ri-post) fixes numerical/completion
 * results; SL.con.3 (#rsl-bounds) requires checked extents before native
 * indexing; R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.2: cooperative contiguous reads and writes.
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md
 * GPU.5: stage only widths that fit and reuse the row.
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md
 * GDSA.2: four-component sums and shuffle order remain fixed.
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md
 */

#include "common.metal"
kernel void rms_norm(constant OperationArgs& a [[buffer(0)]],
                     threadgroup float* values [[threadgroup(0)]],
                     uint lane [[thread_index_in_threadgroup]],
                     uint row [[threadgroup_position_in_grid]]) {
  const ulong width = a.dimensions[1];
  const bool staged = a.flags & 2;
  if (staged) {
    for (ulong i = lane; i < width; i += 256)
      values[i] = load(a.address[0], row * width + i, a.type);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 128) {
      float partial = 0;
      for (ulong i = lane; i < width; i += 128)
        partial += values[i] * values[i];
      const float second = simd_shuffle_down(partial, 1);
      const float third = simd_shuffle_down(partial, 2);
      const float fourth = simd_shuffle_down(partial, 3);
      if (lane % 4 == 0)
        values[width + lane / 4] = ((partial + second) + third) + fourth;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32) {
      float sum = values[width + lane];
      for (uint offset = 1; offset < 32; offset *= 2)
        sum += simd_shuffle_down(sum, offset);
      if (!lane)
        values[width] = precise::rsqrt(sum * (1.0f / float(width)) + a.scalar);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = values[width];
    for (ulong i = lane; i < width; i += 256)
      store(
          a.address[2], row * width + i,
          rounded(values[i] * inverse, a.type) * load(a.address[1], i, a.type),
          a.type);
    return;
  }
  float partials[4] = {0, 0, 0, 0};
  for (uint c = 0; c < 4; ++c) {
    for (ulong i = width <= 128 ? lane + 32 * c : 4 * lane + c; i < width;
         i += 128) {
      const float x = load(a.address[0], row * width + i, a.type);
      partials[c] += x * x;
    }
  }
  float sum = ((partials[0] + partials[1]) + partials[2]) + partials[3];
  for (uint offset = 1; offset < 32; offset *= 2)
    sum += simd_shuffle_down(sum, offset);
  const float total = simd_broadcast(sum, 0);
  const float inverse =
      precise::rsqrt(total * (1.0f / float(width)) + a.scalar);
  for (ulong i = lane; i < width; i += 32) {
    const float x = load(a.address[0], row * width + i, a.type);
    store(a.address[2], row * width + i,
          rounded(x * inverse, a.type) * load(a.address[1], i, a.type), a.type);
  }
}
