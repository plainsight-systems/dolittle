/*
 * Implemented foundation, 2026-10-21; verified on Apple M3 Max.
 * Axis X1/X4: private Metal sequence control bindings, shared by preparation
 * and control-shader encoding. Implements inference::DeviceSequence, rather
 * than defining another application-level generation interface.
 * Contract: bindings belong to one live Device; their owners outlive drain.
 * Position/state, prompt IDs, sampler output, normalization/intervention status
 * and capture targets stay device-owned. The CPU never reads these bindings
 * to decide whether to execute the next prepared step. forward_only omits
 * sampler/terminal bindings; all other optional regions follow the existing
 * DeviceSequence::begin contract. All bytes/alignment are checked before start.
 *
 * Algorithm: metal-port.md, Algorithm and execution boundary / Metal
 * scheduling. Initialize s=0, selected=0, pending=false, failure=0. Before a
 * prompt body, validate injection status, latch intervention failure and read
 * cancellation. If active, copy the next fixed-size prompt chunk using s, run
 * its ICB body, copy hooks and publish consumed(s,rows), then advance s by
 * rows. A barrier orders control writes before the body, and all body/capture
 * reads before the next control update. The static tail follows the same rules.
 *
 * After full prefill, execute the initial head if requested. For generation,
 * a valid sampler result increments selected, marks pending and publishes its
 * ID. For AR, the head writes a vector and performs no sampling. A decode body
 * consumes pending even if EOS/cancellation arrived after selection: gather
 * that device ID, run all blocks/hooks, publish consumed, advance s and clear
 * pending. Select again only when !failure, !EOS, !cancel and selected<G.
 * Never consume an old sampled ID in later no-op timeline entries.
 *
 * The control dispatch writes a full or empty ICB execution range. Within
 * each body, explicit dependencies order projections, normalization, KV writes,
 * attention, residuals and hook copies. A final control step checks
 * intervention count and appends terminal status once. Output windows copy
 * immutable events to their own shared buffers, then Metal completion makes
 * those readable. Sampling/reconstruction kernels remain the operation
 * families' responsibility.
 *
 * Cancellation design: an independent preprepared control dispatch stores 1
 * into a dedicated device-coherent atomic word; inference only loads it. Its
 * initialization precedes both queues using an event. No CPU write races GPU
 * loads, no data payload is published by this flag, and no kernel spins waiting
 * for it. The control queue and flag outlive both streams of work. Device-side
 * atomic visibility across these dispatches and cancellation latency must be
 * established in the existing device integration workflow before acceptance.
 * Do not substitute a host token loop if this mechanism needs revision.
 *
 * Cost: O(1) device control, O(P+G) event/command metadata and O(recipe) ICB
 * bodies. Optimization: prepare once and submit all scheduled windows without
 * collection feedback. No hot tensor allocation, CPU RNG or activation
 * readback. Metal errors terminate the sequence; teardown drains before
 * releasing leases.
 *
 * Both local corpora queried and rules read for header design:
 * cpp-guidelines I.1 (#ri-explicit), R.1 (#rr-raii): bindings and completion
 * own lifetimes. cpp-perf-guidelines GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md), GPU.6
 * (guidelines/gpu/GPU.6-batch-tiny-gpu-work.md), GPU.8
 * (guidelines/gpu/GPU.8-make-barriers-describe-real-hazards.md), GDSA.4
 * (guidelines/gpu-dsa/GDSA.4-no-cross-workgroup-waits-without-forward-progress.md):
 * GPU control, reusable commands, explicit dependencies, no cross-group waits.
 * API basis: Metal Shading Language specification sections 4.8 and 6.16;
 * MTL4ComputeCommandEncoder::executeCommandsInBuffer indirect range and
 * MTLIndirectComputeCommand::setBarrier. Device integration exercises delayed
 * collection, terminal consumption and cancellation; shader instrumentation
 * covers direct control dispatches only.
 */
#pragma once

#include "gpu/device.h"

namespace dolittle::kernels::metal {
struct ControlBindings {
  gpu::ConstDeviceRegion selection;
  gpu::ConstDeviceRegion terminal;
  gpu::DeviceRegion token_input;
  gpu::ConstDeviceRegion prompt;
  gpu::ConstDeviceRegion injection_status;
  gpu::ConstDeviceRegion injection_norm;
  gpu::ConstDeviceRegion intervention_status;
  gpu::ConstDeviceRegion intervention_count;
};
}  // namespace dolittle::kernels::metal
