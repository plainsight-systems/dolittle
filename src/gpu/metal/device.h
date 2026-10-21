/*
 * Implemented foundation, 2026-10-21; header design approved before
 * implementation. Axis X4, Metal kind: native device composition and stable
 * runtime ownership. Contract: open the default Apple Silicon Metal device on
 * macOS 26+; require unified memory, Metal 4 queues, compute indirect command
 * buffers and the shader capabilities required by the prepared sequence.
 * Unavailable features
 * -> runtime_error; never fall back to host computation or a different mode.
 * The returned nonmoving Device outlives its buffers, queues, pipelines,
 * sequences and retained capture leases. No model storage is created here.
 *
 * Algorithm: metal-port.md, Metal scheduling. Obtain the native device, check
 * capabilities and record actual identity/limits. Construct the resource
 * owner and error state. Native device resources and precompiled shader
 * libraries are initialized before a sequence is prepared. Metal's backend
 * implements the existing DeviceBuffer/HostBuffer/Queue handles at link time;
 * native handles are available only through the private backend bridge.
 *
 * Cost: startup allocation and capability queries, separately timed from
 * inference. Optimization: one runtime owner, reusable pipelines and explicit
 * completion; no per-operation framework/context creation. Resource owners
 * retire submitted work before releasing native objects, including failures.
 *
 * Both local corpora queried and rules read for header design:
 * cpp-guidelines R.1 (#rr-raii): native resources follow owning lifetimes.
 * C.122 (#rh-separation): reuse Device's abstract boundary; Metal types stay
 * private. cpp-perf-guidelines GPU.7
 * (guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md): preparation and lifetime
 * completion are separate from live submission. Pattern: Serenity Metal/CUDA
 * device and submission owners, not its renderer frame-pacing loop.
 */
#pragma once

#include <memory>

#include "gpu/device.h"

namespace dolittle::gpu::metal {
[[nodiscard]] std::unique_ptr<Device> open_device();
}  // namespace dolittle::gpu::metal
