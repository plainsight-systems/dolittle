/*
 * Implementation-file header design only; no function bodies yet.
 * Axis X1: orchestrate one prepared full causal forward pass.
 * Contract: implement executor.h's borrowed checkpoint/device lifetime, serial
 * reuse, host-ready owned result and failure cleanup. No persistent KV cache,
 * sampler, tokenizer or NLA. Validate all token IDs before enqueuing their upload.
 * Algorithm: transformer overlay, "Preparing the first execution recipe",
 * "Configurable observation points" and "Storage lifetimes":
 * 1. Resolve selections; obtain and validate the model recipe; prepare kernels
 *    and workspace/operand requirements without allocating tensor storage.
 * 2. Insert requested row copies into the ordered schedule. Extend source last
 *    uses through these copies; place scratch and admit all weight, retained
 *    output, workspace, alignment and staging bytes before allocation/upload.
 * 3. Own one queue and admitted arenas. Upload weights through bounded staging,
 *    waiting before staging reuse during setup. No full-weight pinned duplicate.
 * 4. Per call, copy token provenance, upload IDs, execute prepared operations and
 *    capture copies, enqueue final readbacks and wait for completion. Copy
 *    completed staging into result-owned bytes with stable metadata/views.
 * 5. On failure drain submitted work before releasing its resources; publish no
 *    successful partial result. Poisoned devices invalidate plans.
 * Design and cost: model/placement setup is amortized. Per-call host allocations,
 * input upload and final readback remain explicit. Optimization: no per-layer host
 * wait or tensor allocation, no all-position vocabulary projection when unneeded.
 * Build views only after owner storage is stable. Result destruction needs no GPU.
 * Pure validators/placement are unit-tested independently. Actual forward/device
 * execution and teardown validation belong to the limited integration scenarios.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): resource ownership survives exceptions and teardown.
 * cpp-perf-guidelines GPU.7: completion tracks resource dependencies, not global waits.
 * Source: guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 * cpp-perf-guidelines GPU.9: prepare storage once and preserve last-use lifetimes.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "inference/executor.h"
