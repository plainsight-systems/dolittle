/*
 * Implementation-file header design only; no CUDA bodies yet.
 * Axis X5, CUDA RMSNorm over the final dimension.
 * Algorithm: transformer overlay, "One Qwen3 transformer block": assign a row or
 * head to a block, accumulate squares and reduce in FP32, compute inverse root
 * of mean-square plus epsilon, normalize values, then multiply the matching
 * weight. In BF16 mode round the normalized value to BF16 before weight multiply
 * and round the product on store; this preserves the Qwen3 reference boundary.
 * Scalar/head tails participate safely in every reduction/barrier.
 * Design and cost: O(rows*width), bounded reduction scratch, no full-size squared
 * tensor. Optimization: reuse row data where register/shared capacity permits,
 * with adjacent lanes addressing adjacent features. Do not fuse a later residual
 * or hide a requested block boundary. Finite-input numerical comparisons include
 * zero rows, small/large magnitudes and widths at tile edges.
 * Reference: transformers v4.57.3 modeling_qwen3.py, Qwen3RMSNorm.forward
 * (https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3/modeling_qwen3.py).
 * No bitwise cross-backend reduction-equivalence promise.
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
 */
#include "kernels/operations.h"
