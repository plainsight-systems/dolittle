/*
 * Implemented foundation, 2026-10-21; header design approved before
 * implementation. Axis X4: device memory capabilities and admission-time
 * observations. Contract: query_memory borrows a live Device and performs no
 * allocation of model/tensor storage. Limits describe this device; current
 * usage is a snapshot, not a reservation or a promise that a later allocation
 * succeeds. Zero available bytes means none, not unknown; optional absence
 * means the backend cannot report a useful value. Device failure ->
 * runtime_error.
 *
 * Algorithm: metal-port.md, Memory and model residency. At a request boundary,
 * query the backend's maximum allocation, recommended working-set budget and
 * current device allocations. CUDA reports discrete storage and current free
 * device memory; Metal reports unified storage and its recommended working set
 * separately from physical RAM. The scheduler clamps its configured budget to
 * the reported limit, reserves the new request, then considers idle weights.
 * It does not add unified host/GPU capacities as independent pools.
 *
 * Cost: one backend observation per admission, never per token. No implicit
 * device synchronization. Optimization: expose capabilities as data so pure
 * placement/admission units do not probe the driver or allocate GPU storage.
 * New fields describe byte capacities, not precision or kernel selection.
 *
 * Both local corpora queried and rules read for header design:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): unknown observations and
 * non-reserving snapshots are explicit. cpp-perf-guidelines GPU.9
 * (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md): budget
 * persistent and transient storage before execution. GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md): unified memory still needs
 * explicit ownership and ordering; no per-token memory query.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace dolittle::gpu {
class Device;
enum class MemoryTopology : std::uint8_t { discrete, unified };
// Each allocation may occupy both logical budgets. unified_bytes counts its
// unique physical backing once, including command/staging owners when present.
// On a discrete device unified_bytes is zero. Borrowed storage is not owned.
struct MemoryFootprint {
  std::size_t device_bytes{};
  std::size_t host_bytes{};
  std::size_t unified_bytes{};
};
struct MemorySnapshot {
  MemoryTopology topology{};
  std::size_t max_allocation_bytes{};
  std::size_t working_set_limit_bytes{};
  std::size_t allocated_device_bytes{};
  std::optional<std::size_t> available_device_bytes;
};
[[nodiscard]] MemorySnapshot query_memory(Device&);
}  // namespace dolittle::gpu
