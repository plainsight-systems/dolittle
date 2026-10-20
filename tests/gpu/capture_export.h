/*
 * Axis D3: explicit diagnostic export for the existing CUDA integration tests.
 * Contract: caller supplies only a completed device capture and the initialized
 * byte count. Validate owner/extent, download that prefix on a separate queue,
 * wait for completion, and return a host copy for oracle comparison. The live
 * target-to-AV path never calls this helper. Source remains unchanged and its
 * shared lease outlives the copy. Cost: O(bytes) pinned staging plus host
 * result; both allocations and the readback occur outside measured inference.
 * Algorithm: execution-boundary.md diagnostic export, after H4 readiness.
 * Header/implementation consultation 2026-10-20: Core I.1/R.1 require explicit
 * completion and owned lifetimes; Performance GPU.1/GPU.7 separate observation
 * from resident inference. Source guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Both local corpora queried and rules read in both phases.
 */
#pragma once
#include <optional>
#include <stdexcept>
#include <vector>

#include "inference/observations.h"
namespace dolittle::tests {
inline std::vector<std::byte> export_capture(
    gpu::Device& device, const std::optional<inference::DeviceCapture>& capture,
    std::span<const std::byte> host, std::size_t bytes) {
  if (!capture) {
    if (host.size() != bytes) {
      throw std::invalid_argument("host capture extent");
    }
    return {host.begin(), host.end()};
  }
  if (!host.empty() || !capture->owner || bytes > capture->bytes) {
    throw std::invalid_argument("device capture extent");
  }
  if (!bytes) {
    return {};
  }
  auto queue = device.create_queue();
  auto staging = device.allocate_staging({bytes, 1});
  auto source = capture->region();
  source.bytes = bytes;
  queue.download({&staging, 0, bytes}, source);
  queue.drain();
  return {staging.bytes().begin(), staging.bytes().end()};
}
}  // namespace dolittle::tests
