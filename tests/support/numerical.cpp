/*
 * Implementation-file header design only; no function bodies or test cases yet.
 * Axis D3: pure array comparison supporting numerical verification.
 * Contract: numerical.h defines limits, zero-norm behavior and nonfinite-result
 * failure. Invalid references/shapes/limits throw; invalid actual values produce
 * a failed comparison. No input mutation, files, GPU work or model execution.
 * Algorithm: docs/architecture/transformer-execution.md, "Verification
 * boundaries": validate reference and limits, widen differences before
 * subtraction, accumulate max absolute error and both squared norms in double,
 * then evaluate the explicit limits with the zero-reference exception.
 * Design and cost: one linear comparison pass after validation, constant
 * auxiliary storage, no allocation. Optimization: separate diagnostic work
 * from model compute/readback time. Unit checks use ordinary small arrays and
 * metric invariants; no assertions about loop structure or intermediate sums.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): passing and undefined relative error are explicit.
 * cpp-perf-guidelines GPU.1: diagnostic host work is isolated from GPU timing.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "tests/support/numerical.h"
