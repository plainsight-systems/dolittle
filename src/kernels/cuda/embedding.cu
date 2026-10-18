/*
 * Implementation-file header design only; no CUDA bodies yet.
 * Axis X5, CUDA embedding lookup and bit-preserving selected-row copies.
 * Algorithm: transformer overlay, "Full model flow" and "Configurable observation
 * points": Embedding reads validated uint32 IDs and copies each selected weight
 * row to its output. GatherRows uses prepared, checked row selections; coalesce
 * consecutive rows into copy ranges, and enqueue those device copies in the
 * same ordered queue. The first gather path needs no device index array.
 * Design and cost: O(rows*width*element_bytes) transfer work; embedding reads IDs
 * once per row/cooperative group. Gather preparation is O(selected rows); per-call
 * submission cost is O(contiguous runs), an explicit cost to measure for sparse
 * selections. Empty output selections enqueue nothing.
 * Optimization: contiguous feature lanes and grouped copy ranges, preserving raw
 * bits without normalization or numeric conversion. Observation-copy readers
 * extend source lifetime through completion. Kernelized sparse gathers can follow
 * if measurements justify them without changing selection semantics.
 * IDs must be validated by the executor before upload; no per-element host checks
 * or embedding-specific reinterpretation of tokens as text.
 * Contract: implement the matching operation in kernels/operations.h for dense
 * BF16/FP32 storage and FP32 accumulation. Preparation rejects unsupported shapes,
 * overflowing launch extents or insufficient device resources before submission.
 * Operands/workspace are borrowed from the prepared plan through queue completion;
 * no allocation, host readback or host wait in a kernel launch. Submission errors
 * follow operations.h; deferred device faults surface at the explicit queue wait.
 * Every address and tail lane is bounded; no cross-block spin waits. Kernel
 * selection and arithmetic policy are recorded by the prepared implementation.
 * Bodies, launch geometry and GPU validation are pending; these are design choices,
 * not measured performance or evidence of numerical equivalence.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): equations, rounding and completion are explicit.
 * cpp-perf-guidelines GPU.2: map adjacent lanes to adjacent data.
 * Source: guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 * cpp-perf-guidelines GPU.1: keep row copies on device.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "kernels/operations.h"
