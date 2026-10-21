/*
 * Native application and numerical acceptance design, 2026-10-21:
 * Core I.1 (#ri-explicit) makes backend selection and comparison policy
 * explicit. Performance GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md)
 * keeps production inference on the GPU; test metrics consume completed
 * readbacks. Both local corpora queried and read for design and implementation.
 * Pure numerical comparison helper, no test runner.
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
 * Elementwise comparison instead requires |a-r| <= atol + rtol*|r| at every
 * element. It reports mismatch count independently of the whole-array norm;
 * rtol is never interpreted as a relative L2 threshold. Empty inputs pass.
 * Both modes reject invalid reference values and limits before comparison.
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
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit): thresholds are explicit inputs.
 * cpp-guidelines I.7 (#ri-post): zero-norm and nonfinite outcomes are defined.
 * cpp-perf-guidelines GPU.1: diagnostic comparisons are outside GPU timing.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Portability header design, 2026-10-21: Core I.1 (#ri-explicit) distinguishes
 * elementwise and norm tolerances. Performance GDSA.2 separates accuracy from
 * determinism
 * (guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md).
 * Whole-model diagnostic design, 2026-10-21: I.1 and GDSA.2 above keep
 * measured differences separate from acceptance. These functions borrow equal,
 * nonempty rows; malformed shapes/nonfinite references throw invalid_argument.
 * Nonfinite actual rows return nullopt. Geometry reports cosine and norm ratio
 * (undefined at zero norms); logits report KL(reference || actual), total
 * variation, argmax IDs and the rank of each argmax in the other row.
 * Algorithm: numerical-portability.md, "Whole-model comparisons": accumulate
 * norms/dot product in double; for logits subtract each row's maximum, sum
 * exponentials, then compare normalized probabilities at temperature one.
 * Ranks are one-based, with lower token IDs first for tied logits. There are
 * no acceptance thresholds. Cost: O(row width), constant space, no allocation;
 * GPU.1 keeps this work on completed test readbacks, outside inference timing.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace dolittle::tests {
struct NumericalTolerance {
  double max_absolute_error;
  double max_relative_l2;
};
struct NumericalComparison {
  bool all_values_finite{};
  double maximum_absolute_error{};
  std::optional<double> relative_l2;
  bool passed{};
};
// Whole-model policy leaves measured limits intact. Diagnostic acceptance
// requires finite values; operation checks always enforce their own limits.
// Constant work, no allocation. Shapes are validated by the comparison caller.
enum class ModelComparisonPolicy : std::uint8_t {
  enforce_limits,
  report_differences
};
[[nodiscard]] bool accepts_model_comparison(
    const NumericalComparison& comparison, ModelComparisonPolicy policy);
struct ElementwiseTolerance {
  double absolute;
  double relative;
};
struct ElementwiseComparison {
  std::size_t mismatched_elements{};
  bool all_values_finite{};
  double maximum_absolute_error{};
  bool passed{};
};
struct RowGeometry {
  std::optional<double> cosine_similarity;
  std::optional<double> norm_ratio;
};
struct LogitDiagnostics {
  double reference_to_actual_kl;
  double total_variation;
  std::size_t reference_argmax;
  std::size_t actual_argmax;
  std::size_t reference_argmax_rank_in_actual;
  std::size_t actual_argmax_rank_in_reference;
};
[[nodiscard]] std::optional<RowGeometry> row_geometry(
    std::span<const float> reference, std::span<const float> actual);
[[nodiscard]] std::optional<LogitDiagnostics> logit_diagnostics(
    std::span<const float> reference, std::span<const float> actual);
[[nodiscard]] ElementwiseComparison compare_elementwise(
    std::span<const float> reference, std::span<const float> actual,
    ElementwiseTolerance tolerance);
[[nodiscard]] NumericalComparison compare_values(
    std::span<const float> reference, std::span<const float> actual,
    NumericalTolerance tolerance);
}  // namespace dolittle::tests
