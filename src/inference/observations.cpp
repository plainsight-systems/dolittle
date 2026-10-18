/*
 * Implementation-file header design only; no function bodies yet.
 * Axis X1: resolve read-only observation selections.
 * Contract: implement observations.h with owned immutable metadata and checked
 * sizes. Empty requests allocate no capture payload. Invalid sites/positions or
 * duplicate requests fail explicitly; overflow/limits fail before expanding rows.
 * Algorithm: transformer overlay, "Configurable observation points": validate
 * catalog, resolve each requested semantic point, count all/explicit rows, check
 * products and total limits, then expand owned positions and assign byte ranges.
 * Wire returned spans only after backing vectors have their final size.
 * Design and cost: O(catalog+requests+selected rows) expected indexed work and
 * O(selected rows) metadata. Optimization: resolve once; never parse site strings
 * or allocate per activation element. No file access or GPU work in this module.
 * Unit checks assert valid extents, exact selected positions, no silent dropping,
 * and independence from caller metadata; not a particular lookup container.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): resource ownership survives exceptions and teardown.
 * cpp-perf-guidelines GPU.9: prepare storage once and preserve last-use lifetimes.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "inference/observations.h"
