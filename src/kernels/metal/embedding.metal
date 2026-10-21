/*
 * Implementation: prepared Metal target kernels.
 *
 * Axis X5: embedding lookup. Inputs uint32 IDs[B], weights[V,D]; output
 * X[B,D] in the unchanged weight storage type. Prompt IDs are validated before
 * upload; generated IDs come from the device sampler's validated result.
 * DeviceSequence copies the pending device ID into the input slot before
 * executing this body. Device failure/status prevents consuming an invalid ID.
 *
 * Algorithm: transformer-execution.md, Preparing tokens and embeddings.
 * Each row reads ID once, then copies weights[ID,0..D) into X[row,0..D).
 * BF16 payloads copy bit-for-bit; no arithmetic or dequantization. Adjacent
 * lanes copy adjacent features using aligned vectors and a guarded tail.
 * Output must not overlap weights/IDs; all resources survive completion.
 *
 * Cost O(B*D), one gathered row read plus one write per token, no workspace
 * beyond a per-group ID. Optimization: broadcast the ID and use contiguous
 * row traffic. Future AV replacement remains a separate producer hook after
 * embedding, so lookup fusion cannot hide that activation boundary.
 *
 * Header-design and implementation consultation, 2026-10-21: both local corpora
 * searched and cited rules read. Core I.7 (#ri-post) fixes numerical/completion
 * results; SL.con.3 (#rsl-bounds) requires checked extents before native
 * indexing; R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.1: generated IDs feed lookup without a host round trip.
 * guidelines/gpu/GPU.1-keep-data-on-device.md
 * GPU.2: feature-adjacent lanes copy the gathered row.
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md
 */

#include "common.metal"
kernel void embedding(constant OperationArgs& a [[buffer(0)]],
                      uint i [[thread_position_in_grid]]) {
  const ulong width = a.dimensions[1];
  if (i >= a.dimensions[0] * width) return;
  const uint token = ((device const uint*)a.address[0])[i / width];
  const ulong source = ulong(token) * width + i % width;
  // Gather storage bits, including NaN payloads, without a floating conversion.
  if (a.type)
    ((device ushort*)a.address[2])[i] =
        ((device const ushort*)a.address[1])[source];
  else
    ((device uint*)a.address[2])[i] =
        ((device const uint*)a.address[1])[source];
}
