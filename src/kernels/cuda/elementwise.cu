/*
 * Implementation-file header design only; no CUDA bodies yet.
 * Axis X5, CUDA residual addition and SiLU gate-times-up.
 * Algorithm: transformer overlay, "One Qwen3 transformer block": map contiguous
 * elements to lanes. Add computes each residual sum and rounds to storage.
 * SiluMultiply evaluates SiLU in FP32; in BF16 mode round the activation to BF16
 * before multiplying up, then round the product to storage. Use a numerically
 * stable sigmoid form for large-magnitude finite inputs. Tail lanes never read or
 * write outside extents. No aliasing output with a live input in the initial path.
 * Design and cost: Add reads two arrays/writes one; SiluMultiply does the same
 * with extra scalar arithmetic. No global workspace or intermediate activation
 * array is required. Optimization: fuse activation and multiplication while
 * retaining the prescribed rounding boundary, rather than fusing through a
 * requested residual observation. Backend arithmetic differences remain subject
 * to the independent numerical bounds.
 * Unit tests cover pure shape/access contracts; device arithmetic and residual
 * captures are exercised within the numerical integration scenario.
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
