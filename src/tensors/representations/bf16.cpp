/*
 * Implementation-file header design only; no function bodies yet.
 * Axis X3, BF16 kind: exact host conversion and byte encoding.
 * Contract: implement bf16.h's rounding, NaN, infinity, subnormal and signed-zero
 * rules without allocation or exceptions. Raw read/write preserves every bit.
 * Algorithm: transformer overlay, "Model computation": assert binary32 host
 * representation, bit_cast to unsigned bits, handle NaNs before rounding, apply
 * nearest/even narrowing, widen by bit placement, and encode byte order explicitly.
 * Design and cost: constant work per scalar. Optimization: no full-buffer
 * conversion on capture; callers convert only values needed for comparison.
 * Unit verification exercises ties and representational boundaries, all raw-bit
 * round trips and exact widening. No device or reference model is invoked.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines C.183 (#ru-pun): bit_cast preserves valid object representation access.
 * cpp-perf-guidelines GPU.1: keep transfers/conversions explicit and bounded.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "tensors/representations/bf16.h"
