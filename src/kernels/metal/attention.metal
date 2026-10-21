/*
 * Implementation: prepared Metal target kernels.
 *
 * Axis X5: causal grouped-query attention with preallocated contiguous KV.
 * Inputs Q[B,Hq,d], K/V[C,Hkv,d], active prefix S, initialized rows [0,S+B).
 * Output Y[B,Hq,d]; Hq is divisible by Hkv. Every query i reads only j<=S+i.
 * Inputs/outputs are BF16 or FP32; tensor arithmetic below is FP32; scale setup
 * uses binary64. No read of uninitialized cache capacity, no output alias and
 * no CPU length readback.
 *
 * Algorithm: transformer-execution.md, Attention and positional semantics.
 * 1. Query (i,h) selects KV head g=floor(h/(Hq/Hkv)), with N=S+i+1 keys.
 *    Each QK dot has eight contiguous ranges of ceil(d/8) features. Visit
 *    features ascending with FMA, then add partials at offsets 4,2,1.
 *    score[j]=FP32(dot*scale), with scale=FP32(binary64(1)/sqrt_binary64(d))
 * initialized once on GPU.
 * 2. Find maximum m; compute e[j]=FP32(exp(FP32(score[j]-m))). Denominator
 *    partial t visits j=t,t+32,... ascending; combine at 16,8,4,2,1.
 * 3. Round p[j]=FP32(e[j]/denominator) BEFORE its product with V. For each
 *    output feature, accumulate keys [0,ceil(N/2)) and [ceil(N/2),N) into
 *    separate FP32 FMA sums in ascending order. Add sums and narrow once.
 *    Do not divide an unnormalized PV numerator at the end: that changes the
 *    required rounding. The split uses this query's visible N, never B/C.
 *
 * Metal mapping: decode groups parallelize independent keys with eight lanes
 * per QK dot; feature-adjacent lanes consume V. If N<=4096 and the full
 * 4*(N+d+256)-byte allocation fits native threadgroup limits, retain scores
 * on chip and reuse their storage for probabilities. This reduces K rereads.
 * Otherwise use three bounded passes: max; denominator; probabilities/PV.
 * A key tile contains 32 keys, preserving j modulo 32 ownership across tiles.
 * Recompute identical QK dots each pass, carry the canonical partials, and
 * split the PV traversal at ceil(N/2) even when that falls inside a tile.
 * No O(B*C) global score/probability matrix or per-key synchronization.
 *
 * Prefill groups own up to four queries from one query head. Stage a 32-key
 * K tile and reuse it across those queries. During PV, reuse that staging for
 * V; each query has its own max/denominator/two PV sums and causal mask. At
 * d=128, staged FP32 K or V uses 16 KiB; four Q rows use 2 KiB. A 32-key
 * score tile for four queries adds 512 bytes. Reserve reduction storage too;
 * larger heads reduce query tiling or use the single-query bounded path.
 * Shared-memory allocation must include every simultaneously live array.
 *
 * Cost O(B*(S+B)*Hq*d), bounded on-chip state. The long path reads K three
 * times and V once per query tile; no large global intermediate. Optimization:
 * reuse KV across nearby prompt queries and retain scores where they fit.
 * Reference: llama.cpp fa_vec_common.metal and vLLM-Metal pagedattention_tiled
 * supply tiling mechanisms. Their online-rescaled numerator is not this
 * algorithm; preserve our explicit probability rounding and fixed reductions.
 *
 * Header-design and implementation consultation, 2026-10-21: both local corpora
 * searched and cited rules read. Core I.7 (#ri-post) fixes numerical/completion
 * results; SL.con.3 (#rsl-bounds) requires checked extents before native
 * indexing; R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.2: adjacent lanes read contiguous head features.
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md
 * GPU.5: retain scores only within native shared-memory limits.
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md
 * GDSA.16: tile KV and count recomputation instead of a score matrix.
 * guidelines/gpu-dsa/GDSA.16-tile-through-on-chip-memory.md
 * GDSA.2: query-local fixed trees preserve full/cached equivalence.
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md
 */

#include "common.metal"
#include "elementary.metal"
#include "numeric.metal"
kernel void attention_scale(constant OperationArgs& a [[buffer(0)]]) {
  *((device float*)a.address[4]) = binary64::to_float(
      binary64::F64(1) / binary64::sqrt(binary64::F64(float(a.dimensions[3]))));
}
kernel void attention_retained(constant OperationArgs& a [[buffer(0)]],
                               threadgroup float* memory [[threadgroup(0)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint task [[threadgroup_position_in_grid]]) {
  const ulong width = a.dimensions[3], heads = a.dimensions[1],
              kvheads = a.dimensions[2];
  const ulong count = position(a) + task / heads + 1,
              head = (task % heads) / (heads / kvheads);
  threadgroup float* q = memory;
  threadgroup float* scores = q + width;
  threadgroup float* reduction = scores + a.capacity;
  const uint lane = tid % 8, group = tid / 8;
  const ulong span = (width + 7) / 8;
  const float scale = *((device const float*)a.address[4]);
  for (ulong k = tid; k < width; k += 256)
    q[k] = load(a.address[0], task * width + k, a.type);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ulong base = 0; base < count; base += 32) {
    const ulong key = base + group;
    float dot = 0;
    if (key < count)
      for (ulong k = lane * span; k < min((lane + 1) * span, width); ++k)
        dot =
            fma(q[k],
                load(a.address[1], (key * kvheads + head) * width + k, a.type),
                dot);
    dot = tree(dot, 8);
    if (!lane && key < count) scores[key] = dot * scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maximum = -INFINITY;
  for (ulong k = tid; k < count; k += 256) maximum = max(maximum, scores[k]);
  reduction[tid] = maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride; stride /= 2) {
    if (tid < stride)
      reduction[tid] = max(reduction[tid], reduction[tid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  maximum = reduction[0];
  for (ulong k = tid; k < count; k += 256)
    scores[k] = exp_nonpositive(scores[k] - maximum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32) {
    float sum = 0;
    for (ulong k = tid; k < count; k += 32) sum += scores[k];
    sum = tree(sum, 32);
    if (!tid) reduction[0] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float denominator = reduction[0];
  for (ulong k = tid; k < count; k += 256) scores[k] = scores[k] / denominator;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ulong f = tid; f < width; f += 256) {
    float left = 0, right = 0;
    const ulong split = (count + 1) / 2;
    for (ulong k = 0; k < split; ++k)
      left = fma(scores[k],
                 load(a.address[2], (k * kvheads + head) * width + f, a.type),
                 left);
    for (ulong k = split; k < count; ++k)
      right = fma(scores[k],
                  load(a.address[2], (k * kvheads + head) * width + f, a.type),
                  right);
    store(a.address[3], task * width + f, left + right, a.type);
  }
}
// Four queries share each staged K/V tile. Three passes retain exact normalized
// probabilities and the prescribed CDF/PV order without a context-sized buffer.
kernel void attention_tiled(constant OperationArgs& a [[buffer(0)]],
                            threadgroup float* memory [[threadgroup(0)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint task [[threadgroup_position_in_grid]]) {
  const ulong width = a.dimensions[3], heads = a.dimensions[1],
              kvheads = a.dimensions[2];
  const ulong first_query = (task / heads) * 4, head = task % heads,
              kvhead = head / (heads / kvheads);
  const ulong rows = min(4ul, a.dimensions[0] - first_query),
              start = position(a) + first_query;
  threadgroup float* q = memory;
  threadgroup float* kv = q + 4 * width;
  threadgroup float* scores = kv + 32 * width;
  threadgroup float* reduced = scores + 128;
  threadgroup float* maxima = reduced + 128;
  threadgroup float* denominators = maxima + 4;
  const uint lane = tid % 8, group = tid / 8;
  const ulong span = (width + 7) / 8;
  const float scale = *((device const float*)a.address[4]);
  for (ulong i = tid; i < rows * width; i += 256)
    q[i] = load(a.address[0],
                ((first_query + i / width) * heads + head) * width + i % width,
                a.type);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float left[4] = {0, 0, 0, 0}, right[4] = {0, 0, 0, 0};
  // Width <=256 is admitted for this mapping: each thread owns one PV feature.
  for (uint pass = 0; pass < 3; ++pass) {
    float partial[4] = {0, 0, 0, 0};
    if (!pass)
      for (uint r = 0; r < 4; ++r) partial[r] = -INFINITY;
    for (ulong base = 0; base < start + rows; base += 32) {
      for (ulong i = tid; i < 32 * width; i += 256)
        kv[i] = base + i / width < start + rows
                    ? load(a.address[1],
                           ((base + i / width) * kvheads + kvhead) * width +
                               i % width,
                           a.type)
                    : 0;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint r = 0; r < rows; ++r) {
        float dot = 0;
        if (base + group <= start + r)
          for (ulong k = lane * span; k < min((lane + 1) * span, width); ++k)
            dot = fma(q[r * width + k], kv[group * width + k], dot);
        dot = tree(dot, 8) * scale;
        if (!lane) {
          const bool valid = base + group <= start + r;
          if (pass == 0)
            partial[r] = max(partial[r], valid ? dot : -INFINITY);
          else if (pass == 1 && valid)
            partial[r] += exp_nonpositive(dot - maxima[r]);
          else if (pass == 2)
            scores[r * 32 + group] =
                valid ? exp_nonpositive(dot - maxima[r]) / denominators[r] : 0;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (pass == 2) {
        for (ulong i = tid; i < 32 * width; i += 256)
          kv[i] = base + i / width < start + rows
                      ? load(a.address[2],
                             ((base + i / width) * kvheads + kvhead) * width +
                                 i % width,
                             a.type)
                      : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < width)
          for (uint r = 0; r < rows; ++r) {
            const ulong count = start + r + 1, split = (count + 1) / 2;
            for (ulong k = 0; k < 32 && base + k < count; ++k) {
              if (base + k < split)
                left[r] = fma(scores[r * 32 + k], kv[k * width + tid], left[r]);
              else
                right[r] =
                    fma(scores[r * 32 + k], kv[k * width + tid], right[r]);
            }
          }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    if (pass < 2) {
      if (!lane)
        for (uint r = 0; r < rows; ++r) reduced[r * 32 + group] = partial[r];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < 32)
        for (uint r = 0; r < rows; ++r) {
          float value = reduced[r * 32 + tid];
          for (uint offset = 16; offset; offset /= 2) {
            float other = simd_shuffle_down(value, offset);
            value = pass == 0 ? max(value, other) : value + other;
          }
          if (!tid) {
            if (pass == 0)
              maxima[r] = value;
            else
              denominators[r] = value;
          }
        }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (tid < width)
    for (uint r = 0; r < rows; ++r)
      store(a.address[3], ((first_query + r) * heads + head) * width + tid,
            left[r] + right[r], a.type);
}
