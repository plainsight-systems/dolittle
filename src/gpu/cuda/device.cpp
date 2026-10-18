/*
 * Implementation-file header design only; no function bodies yet.
 * Axis X4, CUDA kind: resource ownership, transfers and completion.
 * Contract: implement gpu/device.h and the explicit CUDA factory; preserve buffer
 * lifetime through outstanding submissions and never recycle in-use storage.
 * Vendor handles remain backend-private. Explicit waits report deferred errors;
 * noexcept cleanup drains safely and poisons unrecoverable devices.
 * Algorithm: transformer overlay, "Preparing the first execution recipe" and
 * "Storage lifetimes": initialize the selected device; create an explicit ordered
 * stream; allocate device arenas and bounded pinned staging; validate regions;
 * register borrowed resources before submitting copies/kernels; record queue
 * prefix events; wait/publish completion; release only after all uses retire.
 * Submission bookkeeping must cover partial-launch failures as well as success.
 * Design and cost: allocations/registration/event resources have explicit setup
 * and teardown costs. Optimization: reuse staging and completion bookkeeping;
 * no legacy-default-stream assumptions or blanket synchronize per observation.
 * A private bridge supplies validated native regions/stream and lifetime
 * registration to CUDA kernel launch code; it exposes no handles to model code.
 * CUDA toolchain/version selection precedes body implementation. GPU execution
 * and fault/teardown evidence are integration work, never unit-test setup.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): resource ownership survives exceptions and teardown.
 * cpp-perf-guidelines GPU.7: completion tracks resource dependencies, not global waits.
 * Source: guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 */
#include "gpu/cuda/device.h"
