/*
 * Implementation: prepared Metal target kernels.
 *
 * Axis X5: split-half_width RoPE and its GPU frequency setup. Input X[B,H,d],
 * even d, theta>0; output same shape or cache rows [S,S+B). S is device-owned.
 * Contract: S+B<=admitted capacity, disjoint input/output, no host calculation
 * of position, angles or frequencies. Every reader follows setup completion.
 *
 * Algorithm: transformer-execution.md, Attention and positional semantics.
 * 1. During GPU setup, for i<d/2 compute exponent=FP32(2*i/d), then
 *    frequency[i]=FP32(binary64(1)/pow_binary64(theta,exponent)). Store the
 *    d/2 frequencies in immutable FP32 device workspace. numeric.metal owns
 *    binary64 operations; uploading precomputed CPU frequencies is not used.
 * 2. Each pair reads t=S+local_row, forms angle=FP32(FP32(t)*frequency[i]),
 *    and rounds FP32 sin(angle)/cos(angle) to the activation storage type.
 * 3. For a=X[...,i], b=X[...,i+d/2], round a*c,b*s,b*c,a*s individually to
 *    storage. Compute y0=round_storage(FP32(a*c-b*s)) and
 *    y1=round_storage(FP32(b*c+a*s)) using those rounded products.
 *
 * Mapping: adjacent lanes own adjacent pairs; two contiguous half_width-head
 * loads and stores. Frequency setup runs once per prepared theta/width, reused
 * by all positions and heads. Q writes local rows; K writes its cache append
 * range. V is never rotated. No fusion across the product-rounding boundaries.
 * Cost O(B*H*d); setup O(d), persistent workspace 2*d bytes, no trig table that
 * grows with sequence length. Optimization: remove repeated binary64 pow from
 * every position/head while retaining its FP32 frequency result.
 *
 * Header-design and implementation consultation, 2026-10-21: both local corpora
 * searched and cited rules read. Core I.7 (#ri-post) fixes numerical/completion
 * results; SL.con.3 (#rsl-bounds) requires checked extents before native
 * indexing; R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.1: GPU setup and device position preserve the boundary.
 * guidelines/gpu/GPU.1-keep-data-on-device.md
 * GPU.2: contiguous half_width-head pair loads.
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md
 * GPU.9: admit immutable frequency storage before sequence start.
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md
 */

#include "common.metal"
#include "numeric.metal"
kernel void rotary_frequencies(constant OperationArgs& a [[buffer(0)]],
                               uint i [[thread_position_in_grid]]) {
  const ulong width = a.dimensions[2];
  if (i >= width / 2) return;
  const auto exponent = binary64::F64(float(2 * i) / float(width));
  ((device float*)a.address[4])[i] = binary64::to_float(
      binary64::F64(1) /
      binary64::pow(binary64::F64::raw(a.address[7]), exponent));
}
kernel void rotary(constant OperationArgs& a [[buffer(0)]],
                   uint i [[thread_position_in_grid]]) {
  const ulong heads = a.dimensions[1], width = a.dimensions[2],
              half_width = width / 2;
  if (i >= a.dimensions[0] * heads * half_width) return;
  const ulong row = i / half_width, feature = i % half_width,
              first = position(a);
  const float angle =
      float(first + row / heads) * ((device const float*)a.address[4])[feature];
  const float c = rounded(precise::cos(angle), a.type),
              s = rounded(precise::sin(angle), a.type);
  const float x = load(a.address[0], row * width + feature, a.type),
              y = load(a.address[0], row * width + feature + half_width,
                       a.type);
  const ulong out =
      (a.flags & 1 ? first * heads * width : 0) + row * width + feature;
  store(a.address[2], out, rounded(x * c, a.type) - rounded(y * s, a.type),
        a.type);
  store(a.address[2], out + half_width,
        rounded(y * c, a.type) + rounded(x * s, a.type), a.type);
}
