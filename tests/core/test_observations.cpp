/*
 * Axis D3: isolated verification of X1 observation preparation contracts.
 * Contract: small in-memory catalogs and requests exercise exact selected rows,
 * bounded payload ranges, rejected malformed inputs, and owned-plan lifetime.
 * No filesystem, model, device, subprocess or application flow is involved.
 * Failures throw through the shared check helpers; diagnostic text is not tested.
 * Algorithm: docs/architecture/transformer-execution.md, "Configurable
 * observation points": prepare caller-selected semantic sites, inspect owned
 * results, then reject invalid points/rows and arithmetic/budget excesses.
 * Design and cost: bounded cases with tiny successful outputs. Huge dimensions
 * appear only in rejection cases to check overflow before row expansion.
 * Optimization: no model fixtures or tensor payload allocations are needed.
 * Corpus consultation, design and implementation, 2026-10-18 (full rules read):
 * cpp-guidelines I.7 (#ri-post): test public postconditions, not private storage.
 * cpp-guidelines R.1 (#rr-raii): owning inputs leave scope before plan inspection.
 * cpp-perf-guidelines COPY.5: test move ownership and reassignment invariants.
 * Source: guidelines/copy-move/COPY.5-resource-owning-types-are-move-only.md.
 * cpp-perf-guidelines CDSA.32: compare resolved rows to requested logical order.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#include "inference/observations.h"
#include "tests/support/check.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

using namespace dolittle;
using tests::check;
using tests::expect_throws;
using models::ObservationBoundary;
using tensors::ElementType;

constexpr models::ObservationSite input_site{
    {ObservationBoundary::block_input, {3}}, 3, ElementType::bfloat16};
constexpr models::ObservationSite output_site{
    {ObservationBoundary::block_output, {17}}, 2, ElementType::float32};
constexpr std::array catalog{input_site, output_site};
constexpr inference::ObservationLimits generous_limits{4096, 128};

static_assert(!std::is_copy_constructible_v<inference::ObservationPlan>);
static_assert(!std::is_copy_assignable_v<inference::ObservationPlan>);
static_assert(std::is_nothrow_move_constructible_v<inference::ObservationPlan>);
static_assert(std::is_nothrow_move_assignable_v<inference::ObservationPlan>);

void selected_rows_and_lifetime() {
    auto plan = [] {
        auto local_catalog = catalog;
        std::array requests{
            inference::ObservationRequest{output_site.point,
                std::vector<models::TokenPosition>{{0}, {3}}},
            inference::ObservationRequest{input_site.point, inference::AllPositions{}}};
        auto result = inference::prepare_observations(local_catalog, {4}, requests, {40, 6});
        local_catalog[1].vector_width = 100;
        std::get<std::vector<models::TokenPosition>>(requests[0].positions)[0].value = 2;
        return result;
    }();
    check(plan.sequence_length().value == 4, "sequence length retained");
    check(plan.payload_bytes() == 40, "mixed dtype payload extent");
    const auto entries = plan.entries();
    check(entries.size() == 2, "all requests retained");
    check(entries[0].site.point.block.value == 17 &&
          entries[0].site.point.boundary == ObservationBoundary::block_output &&
          entries[0].site.vector_width == 2 &&
          entries[0].site.storage_type == ElementType::float32,
          "site metadata retained independently of caller");
    check(entries[0].positions.size() == 2 && entries[0].positions[0].value == 0 &&
          entries[0].positions[1].value == 3, "explicit rows retained independently");
    check(entries[1].positions.size() == 4, "all positions expands sequence");
    for (std::size_t i = 0; i < entries[1].positions.size(); ++i) {
        check(entries[1].positions[i].value == i, "all positions preserves consumed order");
    }
    std::size_t end = 0;
    for (const auto& entry : entries) {
        check(entry.byte_offset == end, "payload is contiguous in request order");
        check(entry.byte_offset <= plan.payload_bytes() &&
              entry.byte_count <= plan.payload_bytes() - entry.byte_offset,
              "payload range contained");
        end += entry.byte_count;
    }
    check(end == plan.payload_bytes(), "payload ranges cover total");
    auto moved = std::move(plan);
    check(moved.entries()[0].positions[1].value == 3, "move retains position ownership");
    plan = std::move(moved);
    check(plan.entries()[1].positions[3].value == 3, "moved-from owner can be reassigned");
    auto destination = inference::prepare_observations({}, {1}, {}, {0, 0});
    destination = std::move(plan);
    check(destination.payload_bytes() == 40, "move assignment transfers populated plan");
}

void empty_selections() {
    const auto empty = inference::prepare_observations({}, {3}, {}, {0, 0});
    check(empty.entries().empty() && empty.payload_bytes() == 0, "empty requests have no payload");
    const std::array requests{inference::ObservationRequest{
        input_site.point, std::vector<models::TokenPosition>{}}};
    const auto plan = inference::prepare_observations(catalog, {3}, requests, {0, 0});
    check(plan.entries().size() == 1 && plan.entries()[0].positions.empty() &&
          plan.entries()[0].byte_count == 0 && plan.payload_bytes() == 0,
          "explicit empty request retained without payload");
    auto wide = input_site;
    wide.vector_width = std::numeric_limits<std::size_t>::max();
    const std::array wide_catalog{wide};
    const auto wide_empty = inference::prepare_observations(wide_catalog, {3}, requests, {0, 0});
    check(wide_empty.payload_bytes() == 0, "zero selected rows require no width product");
}

void malformed_inputs() {
    expect_throws<std::invalid_argument>([] {
        (void)inference::prepare_observations(catalog, {0}, {}, generous_limits);
    });
    const std::array duplicate_catalog{input_site, input_site};
    expect_throws<std::invalid_argument>([&] {
        (void)inference::prepare_observations(duplicate_catalog, {4}, {}, generous_limits);
    });
    for (auto invalid : std::array{
             models::ObservationSite{{static_cast<ObservationBoundary>(99), {3}}, 3,
                                     ElementType::bfloat16},
             models::ObservationSite{input_site.point, 0, ElementType::bfloat16},
             models::ObservationSite{input_site.point, 3, static_cast<ElementType>(99)},
             models::ObservationSite{input_site.point, 3, ElementType::uint32}}) {
        const std::array invalid_catalog{invalid};
        expect_throws<std::invalid_argument>([&] {
            (void)inference::prepare_observations(invalid_catalog, {4}, {}, generous_limits);
        });
    }
    for (const auto point : std::array{
             models::ObservationPoint{ObservationBoundary::block_output, {3}},
             models::ObservationPoint{static_cast<ObservationBoundary>(99), {3}}}) {
        const std::array requests{inference::ObservationRequest{point, inference::AllPositions{}}};
        expect_throws<std::invalid_argument>([&] {
            (void)inference::prepare_observations(catalog, {4}, requests, generous_limits);
        });
    }
    const std::array duplicates{
        inference::ObservationRequest{input_site.point, inference::AllPositions{}},
        inference::ObservationRequest{input_site.point, std::vector<models::TokenPosition>{}}};
    expect_throws<std::invalid_argument>([&] {
        (void)inference::prepare_observations(catalog, {4}, duplicates, generous_limits);
    });
    for (const auto& positions : std::array{
             std::vector<models::TokenPosition>{{1}, {1}},
             std::vector<models::TokenPosition>{{2}, {1}},
             std::vector<models::TokenPosition>{{4}}}) {
        const std::array requests{inference::ObservationRequest{input_site.point, positions}};
        expect_throws<std::invalid_argument>([&] {
            (void)inference::prepare_observations(catalog, {4}, requests, generous_limits);
        });
    }
}

void checked_limits() {
    const std::array requests{
        inference::ObservationRequest{input_site.point, inference::AllPositions{}},
        inference::ObservationRequest{output_site.point, inference::AllPositions{}}};
    const auto exact = inference::prepare_observations(catalog, {4}, requests, {56, 8});
    check(exact.payload_bytes() == 56, "inclusive exact budget accepted");
    for (const auto limits : std::array{
             inference::ObservationLimits{55, 8}, inference::ObservationLimits{56, 7}}) {
        expect_throws<std::length_error>([&] {
            (void)inference::prepare_observations(catalog, {4}, requests, limits);
        });
    }
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    expect_throws<std::length_error>([&] {
        (void)inference::prepare_observations(catalog, {maximum}, requests, {maximum, maximum});
    });
    auto wide_catalog = catalog;
    wide_catalog[0].vector_width = maximum / 4;
    wide_catalog[0].storage_type = ElementType::float32;
    wide_catalog[1].vector_width = maximum / 4;
    expect_throws<std::length_error>([&] {
        (void)inference::prepare_observations(wide_catalog, {1}, requests, {maximum, maximum});
    });
    // Row metadata has its own bound, independent of the generous byte budget.
    expect_throws<std::length_error>([&] {
        (void)inference::prepare_observations(catalog, {maximum / 16}, requests, {maximum, 1});
    });
}

}  // namespace

int main() {
    selected_rows_and_lifetime();
    empty_selections();
    malformed_inputs();
    checked_limits();
}
