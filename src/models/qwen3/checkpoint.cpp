/*
 * Implementation-file header design only; no function bodies yet.
 * Axis M3, Qwen3 kind: configuration parsing and checkpoint parameter binding.
 * Contract: implement pure parse_config/bind_parameters plus bind_target's
 * ownership transfer. Reject unsupported mathematical settings and every missing,
 * duplicate, unexpected or mis-shaped parameter. Distinguish semantic config
 * errors from file integrity errors already handled by the archive.
 * Algorithm: transformer overlay, "Checkpoint loading and binding": obtain verified
 * config bytes; parse bounded JSON with explicit supported/defaulted fields;
 * validate the Qwen3 Config; derive parameter roles and serialized names; validate
 * the complete named layout catalog; bind immutable payload views; publish the
 * owner and versioned identity only after every check. Keep archive/model owners
 * alive until after all borrowed parameter metadata is destroyed.
 * Design and cost: O(blocks) metadata and no tensor copies. Optimization: resolve
 * names into parameter slots once rather than looking up strings during inference.
 * Unit checks use memory JSON/catalogs; real archive opening is integration work.
 * Pin parser/hash support before implementing, sharing the format reader's
 * validated parsing support rather than silently accepting different JSON rules.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): resource ownership survives exceptions and teardown.
 * cpp-perf-guidelines GPU.1: keep transfers/conversions explicit and bounded.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "models/qwen3/checkpoint.h"
