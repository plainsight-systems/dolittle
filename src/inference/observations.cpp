/*
 * Observation request validation and owned metadata preparation.
 * Axis X1: resolve read-only observation selections.
 * Contract: implement observations.h with owned immutable metadata and checked
 * sizes. Empty requests allocate no capture payload. Invalid sites/positions or
 * duplicate requests fail explicitly; overflow/limits fail before expanding
 * rows. Algorithm: transformer overlay, "Configurable observation points":
 * validate catalog, resolve each requested semantic point, count all/explicit
 * rows, check products and total limits, then expand owned positions and assign
 * byte ranges. Wire returned spans only after backing vectors have their final
 * size. Design and cost: O(catalog+requests+selected rows) expected indexed
 * work and O(catalog+requests+selected rows) peak metadata. Optimization:
 * resolve once; never parse site strings or allocate per activation element. No
 * file access or GPU work in this module. Unit checks assert valid extents,
 * exact selected positions, no silent dropping, and independence from caller
 * metadata; not a particular lookup container. Overlay:
 * docs/architecture/transformer-execution.md. Corpus consultation,
 * implementation (full rules read), 2026-10-18: cpp-guidelines R.1 (#rr-raii):
 * resource ownership survives exceptions and teardown. cpp-guidelines I.7
 * (#ri-post): checked owned extents are public postconditions.
 * cpp-perf-guidelines COPY.5: noexcept plan moves transfer the unique owner.
 * Source: guidelines/copy-move/COPY.5-resource-owning-types-are-move-only.md.
 * cpp-perf-guidelines CDSA.32: transform requests once into execution order.
 * Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#include "inference/observations.h"

#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace dolittle::inference {
namespace {

struct PointHash {
  std::size_t operator()(models::ObservationPoint point) const noexcept {
    return std::hash<std::size_t>{}(point.block.value) ^
           std::hash<int>{}(static_cast<int>(point.boundary));
  }
};

struct PointEqual {
  bool operator()(models::ObservationPoint left,
                  models::ObservationPoint right) const noexcept {
    return left.boundary == right.boundary &&
           left.block.value == right.block.value;
  }
};

struct IndexedSite {
  const models::ObservationSite* site;
  bool requested = false;
};

std::size_t checked_add(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::length_error("observation size sum overflows");
  }
  return left + right;
}

std::size_t checked_multiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::length_error("observation size product overflows");
  }
  return left * right;
}

std::size_t storage_bytes(tensors::ElementType type) {
  switch (type) {
    case tensors::ElementType::bfloat16:
      return 2;
    case tensors::ElementType::float32:
      return 4;
    default:
      throw std::invalid_argument("unsupported observation storage type");
  }
}

void validate_boundary(models::ObservationBoundary boundary) {
  switch (boundary) {
    case models::ObservationBoundary::block_input:
    case models::ObservationBoundary::block_output:
      return;
  }
  throw std::invalid_argument("unknown observation boundary");
}

std::size_t row_count(const PositionSelection& selection,
                      models::SequenceLength length) {
  if (selection.valueless_by_exception()) {
    throw std::invalid_argument("missing observation position selection");
  }
  if (std::holds_alternative<AllPositions>(selection)) {
    return length.value;
  }
  return std::get<std::vector<models::TokenPosition>>(selection).size();
}

std::size_t payload_size(const models::ObservationSite& site,
                         std::size_t rows) {
  return checked_multiply(checked_multiply(rows, site.vector_width),
                          storage_bytes(site.storage_type));
}

}  // namespace

struct ObservationPlan::Impl {
  models::SequenceLength length{};
  std::size_t payload_bytes{};
  std::vector<models::TokenPosition> positions;
  std::vector<ResolvedObservation> entries;
};

ObservationPlan::ObservationPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ObservationPlan::~ObservationPlan() = default;
ObservationPlan::ObservationPlan(ObservationPlan&&) noexcept = default;
ObservationPlan& ObservationPlan::operator=(ObservationPlan&&) noexcept =
    default;

std::span<const ResolvedObservation> ObservationPlan::entries() const noexcept {
  return impl_->entries;
}

std::size_t ObservationPlan::payload_bytes() const noexcept {
  return impl_->payload_bytes;
}

models::SequenceLength ObservationPlan::sequence_length() const noexcept {
  return impl_->length;
}

ObservationPlan prepare_observations(
    std::span<const models::ObservationSite> catalog,
    models::SequenceLength sequence_length,
    std::span<const ObservationRequest> requests, ObservationLimits limits) {
  if (sequence_length.value == 0) {
    throw std::invalid_argument("observation sequence must be nonempty");
  }

  std::unordered_map<models::ObservationPoint, IndexedSite, PointHash,
                     PointEqual>
      sites;
  sites.reserve(catalog.size());
  for (const auto& site : catalog) {
    validate_boundary(site.point.boundary);
    if (site.vector_width == 0) {
      throw std::invalid_argument("observation vector width must be positive");
    }
    (void)storage_bytes(site.storage_type);
    if (!sites.emplace(site.point, IndexedSite{&site}).second) {
      throw std::invalid_argument("duplicate observation catalog point");
    }
  }

  std::size_t total_rows = 0;
  std::size_t total_bytes = 0;
  for (const auto& request : requests) {
    validate_boundary(request.point.boundary);
    const auto found = sites.find(request.point);
    if (found == sites.end() || found->second.requested) {
      throw std::invalid_argument("missing or duplicate observation request");
    }
    found->second.requested = true;
    const auto count = row_count(request.positions, sequence_length);
    if (const auto* positions = std::get_if<std::vector<models::TokenPosition>>(
            &request.positions)) {
      for (std::size_t i = 0; i < positions->size(); ++i) {
        if ((*positions).at(i).value >= sequence_length.value ||
            (i != 0 &&
             (*positions).at(i - 1).value >= (*positions).at(i).value)) {
          throw std::invalid_argument(
              "observation positions must increase in range");
        }
      }
    }
    total_rows = checked_add(total_rows, count);
    total_bytes =
        checked_add(total_bytes, payload_size(*found->second.site, count));
    if (total_rows > limits.max_position_entries ||
        total_bytes > limits.max_payload_bytes) {
      throw std::length_error("observation budget exceeded");
    }
  }

  auto impl = std::make_unique<ObservationPlan::Impl>();
  impl->length = sequence_length;
  impl->payload_bytes = total_bytes;
  impl->positions.reserve(total_rows);
  impl->entries.reserve(requests.size());
  for (const auto& request : requests) {
    if (std::holds_alternative<AllPositions>(request.positions)) {
      for (std::size_t position = 0; position < sequence_length.value;
           ++position) {
        impl->positions.push_back({position});
      }
    } else {
      const auto& positions =
          std::get<std::vector<models::TokenPosition>>(request.positions);
      impl->positions.insert(impl->positions.end(), positions.begin(),
                             positions.end());
    }
  }

  std::size_t row_offset = 0;
  std::size_t byte_offset = 0;
  for (const auto& request : requests) {
    const auto& site = *sites.at(request.point).site;
    const auto count = row_count(request.positions, sequence_length);
    const auto bytes = payload_size(site, count);
    impl->entries.push_back(
        {site,
         std::span<const models::TokenPosition>(impl->positions)
             .subspan(row_offset, count),
         byte_offset, bytes});
    row_offset += count;  // Totals were checked before allocation.
    byte_offset += bytes;
  }
  return ObservationPlan(std::move(impl));
}

}  // namespace dolittle::inference
