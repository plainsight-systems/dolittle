/*
 * Axis X4: model-advertised observation selection and durable capture handles.
 * Contract: ObservationPlan resolves unique advertised sites and ordered token
 * positions within a fixed sequence, with checked byte/index budgets. Empty
 * selections remain valid metadata. No hardcoded layer or interpretation law.
 * CapturedRows borrows host diagnostic bytes; OwnedCapture holds either host
 * diagnostic bytes or a DeviceCapture lease, never both for the same payload.
 * DeviceCapture owns a shared immutable DeviceBuffer plus offset/byte extent;
 * its Device outlives it. Positions expose only rows whose producer/copy has
 * completed. An unused capacity suffix is never a valid captured position.
 *
 * Algorithm: execution-boundary.md H1/H4 and transformer-execution.md hooks.
 * Resolve each requested site against the model catalog, validate selected
 * positions, derive row_bytes=width*storage_bytes, and assign checked payload
 * offsets. The executor copies the raw stored residual before source reuse.
 * The generation collector associates completed ranges with absolute consumed
 * positions. A record lookup creates a lease of one completed row. No widening,
 * normalization or text interpretation occurs during capture.
 * Cost: O(selected rows/sites) metadata and width*b per retained row. Live
 * payload stays device-resident across target teardown; explicit diagnostics
 * alone download bytes. Optimization: immutable subregions share a persistent
 * arena separate from transformer scratch and KV.
 *
 * Corpus consultation, header design and implementation, 2026-10-20: both local
 * servers queried separately in each phase and the cited rules read in full.
 * cpp-guidelines I.1 (#ri-explicit), R.1 (#rr-raii): explicit residency,
 * completion, ownership and failure contracts. cpp-perf-guidelines GPU.1,
 * GPU.6, GPU.7, GPU.9: keep tensor intermediates/decisions on device, batch
 * dispatch, decouple observation and separate persistent from transient arenas.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Related GPU.6/7/9 entries describe dispatch, staging and storage.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "gpu/device.h"
#include "models/observation.h"

namespace dolittle::inference {

struct AllPositions {};
using PositionSelection =
    std::variant<AllPositions, std::vector<models::TokenPosition>>;

struct ObservationRequest {
  models::ObservationPoint point;
  PositionSelection positions;
};

struct ObservationLimits {
  std::size_t max_payload_bytes;
  std::size_t max_position_entries;
};

struct ResolvedObservation {
  models::ObservationSite site;
  std::span<const models::TokenPosition> positions;
  std::size_t byte_offset;
  std::size_t byte_count;
};

struct CapturedRows {
  models::ObservationSite site;
  std::span<const models::TokenPosition> positions;
  std::span<const std::byte> bytes;
};

// An immutable leased device extent. Metadata is host-readable; values are
// not. Device must outlive the owner; only the backend may dereference its
// region.
struct DeviceCapture {
  std::shared_ptr<const gpu::DeviceBuffer> owner;
  std::size_t offset{}, bytes{};
  [[nodiscard]] gpu::ConstDeviceRegion region() const noexcept {
    return {owner.get(), offset, bytes};
  }
};

// Owned incremental capture result. Unlike CapturedRows, this retains its
// bytes and absolute positions across later steps and executor destruction.
// Device leases require Device to remain alive; host reference bytes do not.
struct OwnedCapture {
  models::ObservationSite site;
  std::vector<models::TokenPosition> positions;
  std::vector<std::byte> bytes;  // Explicit host-reference/export payload only.
  std::optional<DeviceCapture> device{};
};

class ObservationPlan {
 public:
  ~ObservationPlan();
  ObservationPlan(ObservationPlan&&) noexcept;
  ObservationPlan& operator=(ObservationPlan&&) noexcept;
  ObservationPlan(const ObservationPlan&) = delete;
  ObservationPlan& operator=(const ObservationPlan&) = delete;

  [[nodiscard]] std::span<const ResolvedObservation> entries() const noexcept;
  [[nodiscard]] std::size_t payload_bytes() const noexcept;
  [[nodiscard]] models::SequenceLength sequence_length() const noexcept;

 private:
  struct Impl;
  explicit ObservationPlan(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;

  friend ObservationPlan prepare_observations(
      std::span<const models::ObservationSite>, models::SequenceLength,
      std::span<const ObservationRequest>, ObservationLimits);
};

[[nodiscard]] ObservationPlan prepare_observations(
    std::span<const models::ObservationSite> catalog,
    models::SequenceLength sequence_length,
    std::span<const ObservationRequest> requests, ObservationLimits limits);

}  // namespace dolittle::inference
