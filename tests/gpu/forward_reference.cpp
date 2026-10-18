/*
 * Implementation-file header design only; no function bodies or test cases yet.
 * Axis D3: validate provenance/shape and compare completed forward results.
 * Contract: implement forward_reference.h. The caller supplies a completed
 * result; this helper launches nothing. Malformed reference inputs fail before
 * comparison; actual identity/shape/value mismatches become explicit failures.
 * Algorithm: docs/architecture/transformer-execution.md, "Verification
 * boundaries": validate fixture provenance/selections/lengths, compare run and
 * checkpoint identity, exact token IDs/mode/sites/positions/dtypes, reject
 * missing/extra outputs, decode only matched rows to FP32, and delegate error
 * metrics to support/numerical. Always emit run-provenance status even with no
 * output rows. Preserve raw bytes; decoding never rewrites captured values.
 * Design and cost: O(requested output bytes), bounded temporary conversion
 * vectors and owned diagnostics; no full-model activation readback.
 * Optimization: compare one output at a time and keep conversion outside GPU
 * compute timing. Reference fixture generation is independent of this helper.
 * The harness that obtains device results is integration work within the cap;
 * this source defines no extra scenario and numerical helper unit tests launch
 * nothing. Do not introduce one integration test per layer or operation here.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit): provenance and thresholds are explicit.
 * cpp-guidelines I.7 (#ri-post): metadata match precedes numerical acceptance.
 * cpp-perf-guidelines GPU.1: output decoding is separate diagnostic work.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "tests/gpu/forward_reference.h"
