/*
 * Native application and numerical acceptance design, 2026-10-21:
 * Core I.1 (#ri-explicit) makes backend selection and comparison policy
 * explicit. Performance GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md)
 * keeps production inference on the GPU; test metrics consume completed
 * readbacks. Both local corpora queried and read for design and implementation.
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
 * cpp-perf-guidelines COPY.7: borrow arrays and capture test inputs by
 * reference. Source: guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 * Portability design/implementation consultation, 2026-10-21: Core I.1/I.7
 * (#ri-explicit/#ri-post), performance GDSA.2/GPU.1: test elementwise accuracy
 * independently of norm metrics and device execution. Sources:
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md;
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Whole-model diagnostics, 2026-10-21: Core I.1 (#ri-explicit) for design
 * and I.7 (#ri-post) for implementation separate metrics from acceptance;
 * performance GPU.1 keeps host comparisons outside inference and its timing
 * (guidelines/gpu/GPU.1-keep-data-on-device.md). Both corpora queried/read.
 * numerical-portability.md, "Whole-model comparisons": compare row direction
 * and magnitude, then temperature-one distributions and reciprocal argmax
 * ranks. Use stable centered log-softmax and double host accumulators. Equal
 * nonempty shapes and finite reference values are required; invalid actual
 * rows produce unavailable metrics. No thresholds or production math changes.
 * Linear work, constant auxiliary storage; reuse completed host readbacks.
 */
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "tests/support/check.h"
#include "tests/support/numerical.h"

using dolittle::tests::check;
using dolittle::tests::compare_elementwise;
using dolittle::tests::compare_values;
using dolittle::tests::expect_throws;
using dolittle::tests::logit_diagnostics;
using dolittle::tests::row_geometry;

int main() try {
  const std::array<float, 3> element_reference{0, -8, 8};
  const std::array<float, 3> element_actual{0.25F, -9.25F, 6.75F};
  check(compare_elementwise(element_reference, element_actual, {0.25, 0.125})
            .passed,
        "elementwise limits include absolute plus relative boundary");
  const auto outside =
      compare_elementwise(element_reference, element_actual, {0.125, 0.125});
  check(!outside.passed && outside.mismatched_elements == 3 &&
            outside.maximum_absolute_error == 1.25,
        "every element must satisfy its own reference-scaled bound");
  const std::array<float, 2> large_and_zero{1000, 0};
  const std::array<float, 2> small_outlier{1000, 1};
  check(compare_values(large_and_zero, small_outlier, {1, 0.01}).passed &&
            !compare_elementwise(large_and_zero, small_outlier, {0.01, 0.01})
                 .passed,
        "small norm error cannot hide an elementwise failure near zero");
  check(compare_elementwise({}, {}, {0, 0}).passed,
        "empty elementwise arrays pass");
  const std::array<float, 2> reference{3, 4};
  const std::array<float, 2> twice{6, 8};
  const auto scaled = row_geometry(reference, twice);
  check(scaled && scaled->cosine_similarity == 1 && scaled->norm_ratio == 2,
        "parallel rows distinguish direction from magnitude");
  const std::array<float, 2> perpendicular{-4, 3};
  const auto orthogonal = row_geometry(reference, perpendicular);
  check(orthogonal && orthogonal->cosine_similarity == 0 &&
            orthogonal->norm_ratio == 1,
        "orthogonal rows have zero cosine despite equal norms");
  const std::array<float, 3> scores{0, 1, 1};
  const std::array<float, 3> shifted{1000, 1001, 1001};
  const auto shift = logit_diagnostics(scores, shifted);
  check(shift && shift->reference_to_actual_kl == 0 &&
            shift->total_variation == 0 && shift->reference_argmax == 1 &&
            shift->actual_argmax == 1 &&
            shift->reference_argmax_rank_in_actual == 1,
        "logit offsets preserve probabilities and ties choose lower IDs");
  const std::array<float, 3> reordered{0, 0, 1};
  const auto ranks = logit_diagnostics(scores, reordered);
  check(ranks && ranks->reference_argmax_rank_in_actual == 3 &&
            ranks->actual_argmax_rank_in_reference == 2,
        "reciprocal ranks account for deterministic ties");
  const std::array<float, 2> binary_scores{0, 1};
  const std::array<float, 2> uniform{0, 0};
  const double probability = 1 / (1 + std::exp(-1.0));
  const double expected_kl =
      probability * std::log(2 * probability) +
      (1 - probability) * std::log(2 * (1 - probability));
  const auto distribution = logit_diagnostics(binary_scores, uniform);
  check(
      distribution &&
          std::abs(distribution->reference_to_actual_kl - expected_kl) <
              1e-14 &&
          std::abs(distribution->total_variation - (probability - 0.5)) < 1e-14,
      "binary distribution matches analytic KL and total variation");
  const auto exact = compare_values(reference, reference, {0, 0});
  check(exact.passed && exact.all_values_finite &&
            exact.maximum_absolute_error == 0 && exact.relative_l2 == 0,
        "identical nonzero arrays have zero error");
  const auto boundary = compare_values(reference, twice, {4, 1});
  check(boundary.passed && boundary.maximum_absolute_error == 4 &&
            boundary.relative_l2 == 1,
        "both limits are inclusive");
  check(!compare_values(reference, twice, {3.99, 1}).passed,
        "absolute error can fail independently");
  check(!compare_values(reference, twice, {4, 0.99}).passed,
        "relative error can fail independently");

  const std::array<float, 2> zeros{0, -0.0F};
  const auto zero_geometry = row_geometry(zeros, reference);
  const auto zero_actual = row_geometry(reference, zeros);
  check(zero_geometry && !zero_geometry->cosine_similarity &&
            !zero_geometry->norm_ratio && zero_actual &&
            !zero_actual->cosine_similarity && zero_actual->norm_ratio == 0,
        "zero norms leave only defined geometry available");
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
  const std::array<float, 2> extreme{maximum, -maximum};
  const std::array<float, 2> opposite{-maximum, maximum};
  const auto separated = logit_diagnostics(extreme, opposite);
  check(separated && std::isfinite(separated->reference_to_actual_kl) &&
            separated->total_variation == 1,
        "extreme logits use finite centered log probabilities");
  const auto wide = compare_values(positive, negative,
                                   {2.0 * static_cast<double>(maximum), 2});
  check(wide.passed && wide.all_values_finite && wide.relative_l2 == 2 &&
            wide.maximum_absolute_error == 2.0 * static_cast<double>(maximum),
        "subtraction widens before float overflow");
  const std::array<float, 1> tiny{std::numeric_limits<float>::denorm_min()};
  const std::array<float, 1> zero{0};
  const auto tiny_result = compare_values(tiny, zero, {tiny.at(0), 1});
  check(tiny_result.passed && tiny_result.relative_l2 == 1,
        "subnormal reference is not mistaken for zero norm");

  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (float invalid : {infinity, -infinity, nan}) {
    const std::array<float, 2> nonfinite{invalid, 4};
    check(!row_geometry(reference, nonfinite) &&
              !logit_diagnostics(reference, nonfinite),
          "nonfinite actual values yield unavailable diagnostics");
    expect_throws<std::invalid_argument>(
        [&] { (void)row_geometry(nonfinite, reference); });
    expect_throws<std::invalid_argument>(
        [&] { (void)logit_diagnostics(nonfinite, reference); });
    const auto result = compare_values(reference, nonfinite, {100, 100});
    const auto elements = compare_elementwise(reference, nonfinite, {100, 100});
    check(!elements.passed && !elements.all_values_finite &&
              elements.mismatched_elements == 1,
          "elementwise comparison rejects nonfinite actuals");
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_elementwise(nonfinite, reference, {100, 100}); });
    check(!result.passed && !result.all_values_finite &&
              result.maximum_absolute_error ==
                  std::numeric_limits<double>::infinity() &&
              !result.relative_l2,
          "nonfinite actual values produce a failed result");
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_values(nonfinite, reference, {100, 100}); });
  }
  const std::array<float, 2> invalid_reference{3, nan};
  const std::array<float, 2> invalid_actual{infinity, 4};
  expect_throws<std::invalid_argument>([&] {
    (void)compare_values(invalid_reference, invalid_actual, {100, 100});
  });
  expect_throws<std::invalid_argument>(
      [&] { (void)compare_values(reference, zero, {0, 0}); });
  expect_throws<std::invalid_argument>(
      [&] { (void)compare_elementwise(reference, zero, {0, 0}); });
  expect_throws<std::invalid_argument>([] { (void)row_geometry({}, {}); });
  expect_throws<std::invalid_argument>([] { (void)logit_diagnostics({}, {}); });
  expect_throws<std::invalid_argument>(
      [&] { (void)row_geometry(reference, zero); });
  expect_throws<std::invalid_argument>(
      [&] { (void)logit_diagnostics(reference, zero); });
  expect_throws<std::invalid_argument>([&] {
    (void)compare_elementwise(invalid_reference, invalid_actual, {100, 100});
  });
  for (double invalid :
       {-1.0, static_cast<double>(infinity), static_cast<double>(nan)}) {
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_values(reference, reference, {invalid, 0}); });
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_values(reference, reference, {0, invalid}); });
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_elementwise(reference, reference, {invalid, 0}); });
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_elementwise(reference, reference, {0, invalid}); });
  }
  for (const bool finite : {false, true}) {
    for (const bool within_limits : {false, true}) {
      const dolittle::tests::NumericalComparison measured{
          finite, 0, {}, within_limits};
      check(accepts_model_comparison(
                measured,
                dolittle::tests::ModelComparisonPolicy::enforce_limits) ==
                (finite && within_limits),
            "strict model acceptance");
      check(accepts_model_comparison(
                measured,
                dolittle::tests::ModelComparisonPolicy::report_differences) ==
                finite,
            "diagnostics never accept nonfinite model values");
      check(measured.passed == within_limits,
            "acceptance preserves measured verdict");
    }
  }
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
