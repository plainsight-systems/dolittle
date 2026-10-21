/*
 * Implementation: prepared Metal target kernels.
 *
 * Axis X5: bias-free dense projections for Q/K/V, attention output, MLP and
 * vocabulary. Inputs X[R,I], W[O,I]; output Y[R,O], all BF16 or FP32.
 * Contract: distinct output, initialized inputs, positive checked dimensions.
 * A positioned output writes Y[S+r,o] with S read from device state; capacity
 * includes every written row. All operands/parameters survive body completion.
 *
 * Algorithm: transformer-execution.md, One Qwen3 transformer block.
 * 1. Set P=16 when I<=4096 and O>1024; otherwise P=32. This choice uses only
 *    feature widths. For each (r,o), zero P FP32 partial sums.
 * 2. Partial j visits k=j,j+P,... in ascending order and performs
 *    partial[j]=fma(FP32(X[r,k]),FP32(W[o,k]),partial[j]).
 * 3. Merge upper into lower partials at offsets P/2,P/4,...,1 with separate
 *    FP32 additions. Narrow partial[0] once when writing the output.
 * 4. Ragged widths retain the existing 16-feature padding convention. Guard
 *    loads; padded zero terms follow the same oracle-tested tree in all paths.
 *
 * Metal mapping: decode assigns P adjacent lanes to one output column, with
 * 128 threads/group. A paired BF16 path may give P/2 lanes two original
 * partials each, but must reconstruct the original merge tree. Prefill stages
 * T=4 or 8 rows by K=128 input features; each weight load feeds T row
 * accumulators. Two barriers protect each stage. Shared storage is 4*T*128
 * bytes and registers hold T partials/lane. Tails never skip group barriers.
 * Decode and prefill therefore differ in data reuse, not arithmetic order.
 * For ragged I, use a padded 16x16 staging variant with the same P partials.
 *
 * Cost: approximately 2*R*I*O FLOPs. Weights are read once per T-row tile;
 * no repacking or global intermediate. Optimization: reuse weight loads across
 * prompt rows and keep adjacent lanes on adjacent features. Tune T on M3.
 * llama.cpp kernels/mul_mm.metal and mul_mv.metal motivate separate matrix-
 * matrix/vector paths. Its SIMD-group matrix accumulation is not selected:
 * the intrinsic does not expose our fixed 16/32 partial order. An alternative
 * requires a reviewed numerical policy, never a silent dispatch substitution.
 *
 * Header-design and implementation consultation, 2026-10-21: both local corpora
 * searched and cited rules read. Core I.7 (#ri-post) fixes numerical/completion
 * results; SL.con.3 (#rsl-bounds) requires checked extents before native
 * indexing; R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.2: contiguous features coalesce weight reads.
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md
 * GPU.5: stage reused inputs with bounded barriers/storage.
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md
 * GDSA.2: same dot tree in full, chunked and single-row execution.
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md
 */

#include "common.metal"
template <uint Partials, uint Rows>
void project(constant OperationArgs& a, threadgroup float* stage, uint tid,
             ulong task) {
  const ulong width = a.dimensions[1], outputs = a.dimensions[2];
  const ulong output_tiles = (outputs + 128 / Partials - 1) / (128 / Partials);
  const ulong first_row = (task / output_tiles) * Rows;
  const ulong column =
      (task % output_tiles) * (128 / Partials) + tid / Partials;
  const uint lane = tid % Partials;
  const ulong padded = (width + 15) / 16 * 16;
  float sums[Rows];
  for (uint r = 0; r < Rows; ++r) sums[r] = 0;
  if (Rows == 1) {
    for (ulong k = lane; k < padded; k += Partials) {
      const float w = column < outputs && k < width
                          ? load(a.address[1], column * width + k, a.type)
                          : 0;
      const float x =
          k < width ? load(a.address[0], first_row * width + k, a.type) : 0;
      sums[0] = fma(x, w, sums[0]);
    }
  } else {
    for (ulong base = 0; base < padded; base += 128) {
      for (uint r = 0; r < Rows; ++r)
        stage[r * 128 + tid] =
            first_row + r < a.dimensions[0] && base + tid < width
                ? load(a.address[0], (first_row + r) * width + base + tid,
                       a.type)
                : 0;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint k = lane; k < 128 && base + k < padded; k += Partials) {
        const float w =
            column < outputs && base + k < width
                ? load(a.address[1], column * width + base + k, a.type)
                : 0;
        for (uint r = 0; r < Rows; ++r)
          sums[r] = fma(stage[r * 128 + k], w, sums[r]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  const ulong output_base = a.flags & 1 ? position(a) * outputs : 0;
  for (uint r = 0; r < Rows; ++r) {
    const float sum = tree(sums[r], Partials);
    if (!lane && column < outputs && first_row + r < a.dimensions[0])
      store(a.address[2], output_base + (first_row + r) * outputs + column, sum,
            a.type);
  }
}
#define PROJECT(P, R)                                                         \
  kernel void projection_##P##_##R(constant OperationArgs& a [[buffer(0)]],   \
                                   threadgroup float* s [[threadgroup(0)]],   \
                                   uint t [[thread_index_in_threadgroup]],    \
                                   uint g [[threadgroup_position_in_grid]]) { \
    project<P, R>(a, s, t, g);                                                \
  }
PROJECT(16, 1)
PROJECT(32, 1)
PROJECT(16, 4)
PROJECT(32, 4)
PROJECT(16, 8)
PROJECT(32, 8)
#undef PROJECT
