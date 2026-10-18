/*
 * Implementation-file header design only; no CUDA bodies yet.
 * Axis X5, CUDA unscaled split-half rotary embedding.
 * Algorithm: transformer overlay, "Attention and positional semantics": derive
 * absolute position first_position+p, pair features i and i+d/2, form FP32 angles
 * and trigonometric values, rotate Q or K, never V. In BF16 mode round sin/cos
 * before application and preserve the separate product/add rounding boundaries
 * of the reference. No FMA across a required rounding boundary.
 * Design and cost: O(N*H*d); no KV cache and no data-dependent launch choice.
 * Optimization: adjacent lanes process adjacent feature pairs; compute each
 * pair's angle once for its two outputs. Cross-head/block trig-table caching is a
 * later measured optimization with explicit storage accounting, not hidden state.
 * Checked position addition and supported argument ranges are part of preparation.
 * Reference: transformers v4.57.3 modeling_qwen3.py, Qwen3RotaryEmbedding.forward
 * and apply_rotary_pos_emb
 * (https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3/modeling_qwen3.py).
 * Numerical integration covers position zero and nonzero positions; tests do not
 * assert a particular trigonometric intrinsic or thread arrangement.
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
