/*
 * Implementation-file header design only; no CUDA bodies yet.
 * Axis X5, CUDA projection: Y = X * W^T, with W stored [out,in].
 * Algorithm: transformer overlay, "One Qwen3 transformer block": flatten trailing
 * input/output dimensions, assign each block an output tile, cooperatively load
 * bounded X and W tiles, accumulate across the reduction dimension in FP32, and
 * round once at output storage. Invalid tail loads contribute zero; all lanes
 * participate in shared-memory barriers. No bias, implicit transpose conversion,
 * TF32 substitution or external inference/matrix library in the initial path.
 * Design and cost: 2*rows*in*out FLOPs; shared tile storage is checked against the
 * device limit, with register/output footprint included in launch selection.
 * Optimization: adjacent lanes load contiguous reduction-dimension elements and
 * reuse shared tiles. The initial direct tiled path prioritizes a verified baseline;
 * tensor-core variants require explicit capability selection and comparison.
 * One integration scenario compares model/operation outputs; unit checks validate
 * shape/dtype/extent contracts without launching this kernel.
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
