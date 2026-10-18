/*
 * Implementation-file header design only; no function bodies yet.
 * Axis X3: checked tensor layout arithmetic.
 * Contract: implement tensor.h without reading tensor payloads or accessing a
 * device. Invalid dtype/rank/strides/null layout -> invalid_argument; overflowing
 * products/sums -> length_error. Empty shapes and rank-zero scalars follow the
 * interface's distinct rules. Returned layouts own their shape/stride vectors.
 * Algorithm: transformer overlay, "Model computation" and "Cost model": validate
 * dtype/rank/stride units, detect empty dimensions, compute reachable extent with
 * checked arithmetic, and compare HostView capacity. contiguous_layout derives
 * strides from the innermost dimension outward without unchecked multiplication.
 * Design and cost: O(rank) validation, no data copies; allocation only for owned
 * layout metadata. Optimization: callers retain prepared layouts outside loops.
 * Unit verification checks extent/containment/rejection contracts, not internal
 * branch order. Kernel-specific dense/non-overlap validation is a separate step.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): enforce the stated results and failure boundaries.
 * cpp-perf-guidelines GPU.9: prepare storage once and preserve last-use lifetimes.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "tensors/tensor.h"
