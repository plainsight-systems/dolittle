/*
 * Implementation-file header design only; no function bodies yet.
 * Axis M2: TensorArchive base lifetime definitions; format-specific ownership
 * and decoding remain in the selected format implementation.
 * Contract: the abstract owner cannot be copied and is destroyed polymorphically.
 * Base destruction does not access derived mappings or throw. Concrete owners
 * must release payload backing only after their own view metadata is discarded.
 * Algorithm: docs/architecture/transformer-execution.md, "Checkpoint loading
 * and binding": establish the abstract ownership boundary used by the concrete
 * reader and binder; teardown follows reverse ownership, never hidden readback.
 * Design and cost: no tensor copy, allocation, parsing or I/O in the base.
 * Optimization: format-neutral lifetime dispatch adds no per-element work.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): virtual owner teardown encloses backing lifetime.
 * cpp-perf-guidelines GPU.1: host archive lifetime does not imply device transfers.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "tensors/formats/format.h"
