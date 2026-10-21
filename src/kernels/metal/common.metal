/*
 * Implementation: prepared Metal target kernels.
 *
 * Axis X3/X5: Metal scalar conversions and fixed reduction helpers used by
 * the target kernels. Contract: BF16 storage is an exact 16-bit payload; FP32
 * storage is binary32. No FP16 conversion, fast-math reassociation or implicit
 * contraction. Explicit fma is used only where the algorithm requests it.
 * Shader compilation must disable fast math and unintended contraction.
 *
 * Algorithm: transformer-execution.md, Model computation.
 * Widen BF16 by placing its bits in the high 16 bits of binary32. Narrow finite
 * FP32 using round-to-nearest/ties-to-even: add 0x7fff plus retained low bit,
 * then take the high 16 bits. Preserve signed zero/infinity; canonicalize NaN
 * to 0x7fc0 as the existing host representation does. Keep intermediate
 * storage rounding explicit even when two operations share one dispatch.
 *
 * A reduction helper takes a fixed active lane group and explicit offsets.
 * Each shuffle/add follows that order; never replace it with simd_sum when
 * the API leaves its tree unspecified. Inactive tail lanes contribute neutral
 * values and still reach every threadgroup barrier. Feature indexing uses
 * checked 64-bit offsets; no 32-bit byte-product wrap for large weight views.
 *
 * Cost: O(1) conversion, O(log lanes) shuffle tree, no global scratch.
 * Optimization: packed BF16 loads reduce address/load instructions while each
 * unpacked component keeps its original FP32 accumulator. Packed loads require
 * proved alignment and guarded tails; scalar paths preserve the same values.
 * Numerical checks include ties, subnormals, signed zeros, infinities and NaNs.
 *
 * Header-design and implementation consultation, 2026-10-21: both local corpora
 * searched and cited rules read. Core I.7 (#ri-post) fixes numerical/completion
 * results; SL.con.3 (#rsl-bounds) requires checked extents before native
 * indexing; R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GDSA.2: fixed partial order and explicit round boundaries.
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md
 * GPU.2: packed adjacent features preserve coalescing.
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md
 */

#ifndef DOLITTLE_METAL_COMMON
#define DOLITTLE_METAL_COMMON
#include <metal_stdlib>
using namespace metal;
struct OperationArgs {
  ulong address[8];
  ulong dimensions[4];
  ulong first, capacity;
  uint type, flags;
  float scalar;
  uint row_tile;
};
inline ushort bf16(float x) {
  uint b = as_type<uint>(x);
  return isnan(x) ? ushort(0x7fc0)
                  : ushort((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
inline float widen(ushort x) { return as_type<float>(uint(x) << 16); }
inline float load(ulong address, ulong index, uint type) {
  return type ? widen(((device const ushort*)address)[index])
              : ((device const float*)address)[index];
}
inline void store(ulong address, ulong index, float x, uint type) {
  if (type)
    ((device ushort*)address)[index] = bf16(x);
  else
    ((device float*)address)[index] = x;
}
inline float rounded(float x, uint type) { return type ? widen(bf16(x)) : x; }
inline ulong position(constant OperationArgs& a) {
  return a.address[5] ? *((device const ulong*)a.address[5]) : a.first;
}
inline float tree(float x, uint partials) {
  for (uint offset = partials / 2; offset; offset /= 2)
    x += simd_shuffle_down(x, offset);
  return x;
}
#endif
