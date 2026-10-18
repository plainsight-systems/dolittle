/*
 * Header design; request resolution and device copies are not implemented.
 * Axis: X1, observation preparation within forward execution planning.
 *
 * Contract: prepare_observations consumes a model-advertised catalog, a
 * positive full-sequence length, requests and explicit limits. Empty requests
 * are valid and produce zero capture payload. An empty explicit position list
 * selects no rows; AllPositions selects [0, sequence_length). Explicit lists
 * must be strictly increasing and in range. Requested points must be unique,
 * advertised, and have valid shape/type. Reject malformed/unsupported inputs
 * with invalid_argument, overflow or exceeded limits with length_error, and
 * propagate allocation failure. Preparation never drops requests silently.
 *
 * The returned plan owns all resolved metadata, retaining no borrowed input
 * spans. CapturedRows is a read-only view into its ForwardResult owner; both
 * positions and bytes become invalid on owner destruction or move. Payload is
 * contiguous [selected_positions, vector_width] in original storage dtype.
 * Input plan views likewise require a live, unmoved plan. Moved-from plans
 * support destruction/reassignment only. No GPU handles or callbacks escape.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Configurable
 * observation points" and "Storage lifetimes":
 * 1. Validate the catalog, unique requests and position selections.
 * 2. Count rows and checked byte products/sums; reject overflow/limits BEFORE
 *    allocating expanded metadata. No wraparound in any offset or extent.
 * 3. Resolve sites, expand position lists, assign non-overlapping byte ranges
 *    in request order, and retain immutable metadata for execution.
 * 4. The executor copies those rows before scratch reuse, preserves raw bits,
 *    and exposes CapturedRows only after host-copy completion.
 *
 * Design and cost: R requests and P total selected rows require O(R+P)
 * resolved metadata; payload bytes are sum(rows * width * element_bytes).
 * max_position_entries bounds P independently of max_payload_bytes. Estimate
 * metadata and allocator/staging overhead separately in overall admission.
 * Optimization: resolve/allocate once per prepared pass, batch row copies,
 * and issue no capture allocation/readback when R is zero. No cost is measured.
 *
 * Corpus guidance queried for header design, 2026-10-17:
 * cpp-guidelines I.7 (#ri-post): state ready/owned-plan and borrowed-view rules.
 * cpp-guidelines R.1 (#rr-raii): plan metadata has automatic resource lifetime.
 * cpp-perf-guidelines COPY.5: plans are move-only; noexcept moves transfer
 * ownership. Source: guidelines/copy-move/COPY.5-resource-owning-types-are-move-only.md.
 * cpp-perf-guidelines CDSA.32: pre-resolve immutable capture metadata.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 * cpp-perf-guidelines GPU.1: batch required readbacks, preserve device intermediates.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include "models/observation.h"

#include <cstddef>
#include <memory>
#include <span>
#include <variant>
#include <vector>

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
    std::span<const ObservationRequest> requests,
    ObservationLimits limits);

}  // namespace dolittle::inference
