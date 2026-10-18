/*
 * Implementation-file header design only; no function bodies yet.
 * Axis M2, safetensors kind: pure metadata decoding plus verified archive loading.
 * Contract: implement safetensors.h and the archive-owned view lifetime in
 * format.h. Parsing consumes memory; loading alone performs file I/O. Enforce
 * limits before metadata growth. All input/layout/overflow/integrity failures
 * follow the header categories; unwinding closes every acquired mapping/handle.
 * Algorithm: transformer overlay, "Checkpoint loading and binding":
 * 1. Validate explicit artifact/shard/index paths and expected digests; reject
 *    escapes/duplicates and open only the declared artifacts.
 * 2. Check actual byte counts and hashes; decode the 8-byte little-endian header
 *    length only after checking framing. Bound all file-offset arithmetic.
 * 3. Call parse_header on the bounded JSON span; check dense ranges, dtypes and
 *    complete payload coverage. Reject duplicate JSON keys rather than last-wins.
 * 4. Match the supplied index to actual shard contents, construct canonical
 *    source identity, retain backing owners and publish only the complete archive.
 * Design and cost: one verification read per file, O(tensors) owned metadata,
 * mapped payload views and bounded staging later. Optimization: no eager host
 * copy/conversion of full weights. A JSON/hash support dependency must be pinned
 * before implementation, with duplicate-key and allocation-limit behavior checked.
 * Unit tests call the same in-memory parser. File/hash/index integration coverage
 * belongs to the small integration budget, not a file-spawning unit fixture.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): resource ownership survives exceptions and teardown.
 * cpp-perf-guidelines GPU.1: keep transfers/conversions explicit and bounded.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "tensors/formats/safetensors.h"
