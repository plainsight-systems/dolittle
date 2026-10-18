/*
 * Header design; CUDA runtime initialization is not implemented.
 * Axis X4, CUDA kind: explicit composition entry point for a stable Device.
 * Contract: select an existing nonnegative runtime ordinal, create a device
 * context/resource owner and report actual device/driver metadata. Invalid
 * ordinal -> invalid_argument; unavailable runtime/device -> runtime_error;
 * host allocation failure propagates. No fallback to a different GPU/backend.
 * Returned owner outlives all buffers, queues and prepared kernel plans.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Preparing the first
 * execution recipe": establish runtime ownership before capability selection
 * and memory admission. This factory allocates no model weights or scratch.
 * Optimization: initialize once per device lifetime; use an explicit queue
 * rather than the legacy default stream. Launch geometry belongs to kernels.
 * Cost: initialization is measured separately from prepared forward execution.
 *
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): unique ownership encloses runtime lifetime.
 * cpp-perf-guidelines GPU.7: explicit queues make synchronization intentional.
 * Source: guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 */
#pragma once
#include "gpu/device.h"
#include <memory>

namespace dolittle::gpu::cuda {
[[nodiscard]] std::unique_ptr<Device> open_device(int ordinal);
}  // namespace dolittle::gpu::cuda
