/*
 * Axis D3: isolated verification of X3 tensor layout contracts.
 * Contract: exercise only in-memory layout metadata and byte views; failed
 * postconditions throw through the test helper. No files, devices or processes.
 * Algorithm: docs/architecture/transformer-execution.md, "Model computation"
 * and "Cost model": compare logical element addresses with reachable extent,
 * check dense construction and borrowed capacity, then reject invalid metadata
 * and arithmetic overflow. Scalars, empty and overlapping layouts are distinct.
 * Design/cost: bounded small layouts and O(rank) metadata; no payload conversion.
 * Optimization: cases validate prepared layout results, not per-element machinery.
 * Corpus consultation, test design and implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): assert public postconditions, not helper choices.
 * cpp-guidelines ES.103 (#res-overflow): cover products and extent-sum overflow.
 * cpp-perf-guidelines CDSA.32: verify documented shape/stride consumer contracts.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#include "tensors/tensor.h"
#include "tests/support/check.h"

#include <array>
#include <limits>
#include <stdexcept>

using namespace dolittle::tensors;
using dolittle::tests::check;
using dolittle::tests::expect_throws;

int main() {
    for (const auto type : {ElementType::bfloat16, ElementType::float32,
                            ElementType::uint32}) {
        const auto width = type == ElementType::bfloat16 ? 2U : 4U;
        check(element_bytes(type) == width, "stored element width");
        const auto scalar = contiguous_layout(type, {});
        check(scalar.shape.empty() && scalar.byte_strides.empty(), "scalar rank");
        check(required_bytes(scalar) == width, "scalar contains one element");

        std::array<std::size_t, 3> shape{2, 3, 5};
        const auto dense = contiguous_layout(type, shape);
        check(dense.shape == std::vector<std::size_t>(shape.begin(), shape.end()),
              "layout preserves logical shape");
        check(required_bytes(dense) == 30 * width, "dense extent");
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                for (std::size_t k = 0; k < 5; ++k) {
                    const auto address = i * dense.byte_strides[0] +
                        j * dense.byte_strides[1] + k * dense.byte_strides[2];
                    check(address == ((i * 3 + j) * 5 + k) * width,
                          "row-major logical addressing");
                }
            }
        }
        shape[0] = 9;
        check(dense.shape[0] == 2, "layout owns shape independently");

        const Layout padded{type, {2, 3}, {8 * width, width}};
        check(required_bytes(padded) == 11 * width, "padded extent includes gaps");
        const Layout transposed{type, {3, 2}, {width, 8 * width}};
        check(required_bytes(transposed) == required_bytes(padded),
              "axis permutation preserves addressed extent");
        const Layout overlapping{type, {100, 2}, {0, width}};
        check(required_bytes(overlapping) == 2 * width,
              "extent alone does not reject overlap");

        std::array<std::byte, 128> bytes{};
        const auto extent = required_bytes(padded);
        validate_view({&padded, std::span<const std::byte>(bytes).first(extent)});
        validate_view({&padded, bytes});
        expect_throws<std::invalid_argument>([&] {
            validate_view({&padded, std::span<const std::byte>(bytes).first(extent - 1)});
        });
        // Byte storage need not be aligned to the numerical representation.
        validate_view({&scalar, std::span<const std::byte>(bytes).subspan(1, width)});

        for (const auto empty_shape : {std::array<std::size_t, 3>{0, 2, 3},
                                      std::array<std::size_t, 3>{2, 0, 3},
                                      std::array<std::size_t, 3>{2, 3, 0}}) {
            const auto empty = contiguous_layout(type, empty_shape);
            check(required_bytes(empty) == 0, "zero dimension means empty");
            validate_view({&empty, {}});
        }
    }

    const auto maximum = std::numeric_limits<std::size_t>::max();
    const auto aligned_maximum = maximum - maximum % 4;
    const auto empty = contiguous_layout(ElementType::float32,
        std::array<std::size_t, 3>{0, maximum, maximum});
    check(required_bytes(empty) == 0, "empty addresses do not overflow");
    const Layout empty_strided{ElementType::float32, {maximum, 0},
                               {aligned_maximum, 4}};
    check(required_bytes(empty_strided) == 0, "empty strided extent is zero");

    expect_throws<std::invalid_argument>([] { validate_view({nullptr, {}}); });
    expect_throws<std::invalid_argument>([] {
        (void)element_bytes(static_cast<ElementType>(-1));
    });
    expect_throws<std::invalid_argument>([] {
        (void)required_bytes({ElementType::float32, {2}, {}});
    });
    expect_throws<std::invalid_argument>([] {
        (void)required_bytes({ElementType::float32, {0}, {3}});
    });
    expect_throws<std::length_error>([&] {
        (void)required_bytes({ElementType::float32, {maximum}, {4}});
    });
    expect_throws<std::length_error>([&] {
        (void)required_bytes({ElementType::float32, {2}, {aligned_maximum}});
    });
    expect_throws<std::length_error>([&] {
        (void)required_bytes({ElementType::float32, {2, 2},
                             {aligned_maximum / 2 - (aligned_maximum / 2) % 4,
                              aligned_maximum}});
    });
    expect_throws<std::length_error>([&] {
        (void)contiguous_layout(ElementType::float32,
                               std::array<std::size_t, 1>{maximum});
    });
}
