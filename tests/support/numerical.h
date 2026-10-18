/*
 * Header design only; pure numerical comparison helper, no test runner.
 * Axis D3: error metrics independent of inference, files or hardware.
 * Contract: equal-length FP32 spans and finite nonnegative tolerance limits;
 * reference values must be finite. Invalid inputs throw invalid_argument.
 * Inputs are borrowed only during the call. Nonfinite actual values yield
 * all_values_finite=false, passed=false, maximum_absolute_error=+infinity,
 * relative_l2 absent. Otherwise max absolute error and relative L2 are computed
 * with double intermediates. If the reference norm is zero, relative L2 is
 * absent and the absolute limit alone applies; otherwise both limits apply.
 * Empty equal spans pass with zero absolute error and absent relative L2.
 * No implicit threshold selection or tolerance adjustment after a failure.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Verification
 * boundaries" and "Model computation": validate limits/reference, accumulate
 * max |a-r|, sum (a-r)^2 and sum r^2 in double, form the norm ratio when
 * defined, then compare with supplied limits. No tensor execution or readback.
 * Cost: O(elements), O(1) auxiliary storage, no allocation. Optimization: the
 * integration harness decodes only requested outputs; this pass is reported
 * separately from model compute. Unit tests check metric identities, limits
 * and invalid inputs; no snapshots of accumulation order or diagnostics.
 *
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit): thresholds are explicit inputs.
 * cpp-guidelines I.7 (#ri-post): zero-norm and nonfinite outcomes are defined.
 * cpp-perf-guidelines GPU.1: diagnostic comparisons are outside GPU timing.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include <optional>
#include <span>

namespace dolittle::tests {
struct NumericalTolerance {
    double max_absolute_error;
    double max_relative_l2;
};
struct NumericalComparison {
    bool all_values_finite;
    double maximum_absolute_error;
    std::optional<double> relative_l2;
    bool passed;
};
[[nodiscard]] NumericalComparison compare_values(
    std::span<const float> reference, std::span<const float> actual,
    NumericalTolerance tolerance);
}  // namespace dolittle::tests
