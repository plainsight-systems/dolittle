/*
 * Axis D3: isolated numerical-comparison contract tests.
 * Contract: small in-memory arrays exercise defined metric results, tolerance
 * boundaries and invalid-input rejection. No files, devices or model execution.
 * Failure throws through the test helper; success returns zero.
 * Algorithm: docs/architecture/transformer-execution.md, "Verification
 * boundaries": compare hand-derived norms and errors, distinguish undefined
 * relative error from a failed comparison, and reject invalid references.
 * Cost: fixed small inputs; no performance claim or accumulation-order checks.
 * Corpus consultation, design and implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): assert public numerical outcomes and failures.
 * cpp-perf-guidelines COPY.7: borrow arrays and capture test inputs by reference.
 * Source: guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 */
#include "tests/support/check.h"
#include "tests/support/numerical.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

using dolittle::tests::check;
using dolittle::tests::compare_values;
using dolittle::tests::expect_throws;

int main() {
    const std::array<float, 2> reference{3, 4};
    const std::array<float, 2> twice{6, 8};
    const auto exact = compare_values(reference, reference, {0, 0});
    check(exact.passed && exact.all_values_finite &&
          exact.maximum_absolute_error == 0 && exact.relative_l2 == 0,
          "identical nonzero arrays have zero error");
    const auto boundary = compare_values(reference, twice, {4, 1});
    check(boundary.passed && boundary.maximum_absolute_error == 4 &&
          boundary.relative_l2 == 1, "both limits are inclusive");
    check(!compare_values(reference, twice, {3.99, 1}).passed,
          "absolute error can fail independently");
    check(!compare_values(reference, twice, {4, 0.99}).passed,
          "relative error can fail independently");

    const std::array<float, 2> zeros{0, -0.0F};
    const auto zero_reference = compare_values(zeros, reference, {4, 0});
    check(zero_reference.passed && !zero_reference.relative_l2 &&
          zero_reference.maximum_absolute_error == 4,
          "zero reference norm uses only the absolute limit");
    const auto empty = compare_values({}, {}, {0, 0});
    check(empty.passed && empty.all_values_finite &&
          empty.maximum_absolute_error == 0 && !empty.relative_l2,
          "empty arrays satisfy the zero-error contract");

    const float maximum = std::numeric_limits<float>::max();
    const std::array<float, 1> positive{maximum};
    const std::array<float, 1> negative{-maximum};
    const auto wide = compare_values(positive, negative,
                                    {2.0 * static_cast<double>(maximum), 2});
    check(wide.passed && wide.all_values_finite && wide.relative_l2 == 2 &&
          wide.maximum_absolute_error == 2.0 * static_cast<double>(maximum),
          "subtraction widens before float overflow");
    const std::array<float, 1> tiny{std::numeric_limits<float>::denorm_min()};
    const std::array<float, 1> zero{0};
    const auto tiny_result = compare_values(tiny, zero, {tiny[0], 1});
    check(tiny_result.passed && tiny_result.relative_l2 == 1,
          "subnormal reference is not mistaken for zero norm");

    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (float invalid : {infinity, -infinity, nan}) {
        const std::array<float, 2> nonfinite{invalid, 4};
        const auto result = compare_values(reference, nonfinite, {100, 100});
        check(!result.passed && !result.all_values_finite &&
              result.maximum_absolute_error == std::numeric_limits<double>::infinity() &&
              !result.relative_l2, "nonfinite actual values produce a failed result");
        expect_throws<std::invalid_argument>([&] {
            (void)compare_values(nonfinite, reference, {100, 100});
        });
    }
    const std::array<float, 2> invalid_reference{3, nan};
    const std::array<float, 2> invalid_actual{infinity, 4};
    expect_throws<std::invalid_argument>([&] {
        (void)compare_values(invalid_reference, invalid_actual, {100, 100});
    });
    expect_throws<std::invalid_argument>([&] {
        (void)compare_values(reference, zero, {0, 0});
    });
    for (double invalid : {-1.0, static_cast<double>(infinity), static_cast<double>(nan)}) {
        expect_throws<std::invalid_argument>([&] {
            (void)compare_values(reference, reference, {invalid, 0});
        });
        expect_throws<std::invalid_argument>([&] {
            (void)compare_values(reference, reference, {0, invalid});
        });
    }
}
