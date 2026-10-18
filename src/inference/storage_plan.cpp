/*
 * Implementation-file header design only; no function bodies yet.
 * Axis X1: pure reusable scratch placement.
 * Contract: implement storage_plan.h with checked align-up and non-overlap for
 * inclusive lifetimes. Every requested ID is represented exactly once. A budget
 * failure returns no partial plan. Empty/zero-byte resources follow the header.
 * Algorithm: transformer overlay, "Model computation": validate IDs/ranges and
 * alignments; order starts; retire intervals whose last use is strictly earlier;
 * choose an aligned free range or grow the arena; coalesce free ranges and verify
 * containment/alignment/lifetime non-overlap before returning owned metadata.
 * Design and cost: simple O(R^2) worst-case preparation, O(R) metadata, no device
 * allocation. Optimization: reuse one arena across calls and non-overlapping
 * values. No claim of optimal packing. Unit tests assert invariants under varied
 * input order and lifetimes, never exact offsets, free-list order or pool size
 * unless required by an explicit admission bound.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): enforce the stated results and failure boundaries.
 * cpp-perf-guidelines GPU.9: prepare storage once and preserve last-use lifetimes.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "inference/storage_plan.h"
