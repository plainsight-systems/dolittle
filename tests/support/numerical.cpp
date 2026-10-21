/*
 * Native application and numerical acceptance design, 2026-10-21:
 * Core I.1 (#ri-explicit) makes backend selection and comparison policy
 * explicit. Performance GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md)
 * keeps production inference on the GPU; test metrics consume completed
 * readbacks. Both local corpora queried and read for design and implementation.
 * Pure numerical comparison implementation.
 * Axis D3: pure array comparison supporting numerical verification.
 * Contract: numerical.h defines limits, zero-norm behavior and nonfinite-result
 * failure. Invalid references/shapes/limits throw; invalid actual values
 * produce a failed comparison. No input mutation, files, GPU work or model
 * execution. Algorithm: docs/architecture/transformer-execution.md,
 * "Verification boundaries": validate reference and limits, widen differences
 * before subtraction, accumulate max absolute error and both squared norms in
 * double, then evaluate the explicit limits with the zero-reference exception.
 * Design and cost: one linear comparison pass after validation, constant
 * auxiliary storage, no allocation. Optimization: separate diagnostic work
 * from model compute/readback time. Unit checks use ordinary small arrays and
 * metric invariants; no assertions about loop structure or intermediate sums.
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): passing and undefined relative error are
 * explicit. cpp-perf-guidelines GPU.1: diagnostic host work is isolated from
 * GPU timing. Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Portability implementation consultation, 2026-10-21: Core I.7 (#ri-post)
 * defines elementwise failures separately from norm error; GPU.1 keeps these
 * linear host comparisons outside the production computation and its timing.
 * Elementwise algorithm: validate all references and limits, widen subtraction,
 * count |a-r| > atol + rtol*|r|, and reject every nonfinite actual element.
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
#include "tests/support/numerical.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "src/bounds.h"

namespace dolittle::tests {
bool accepts_model_comparison(const NumericalComparison& comparison,
                              ModelComparisonPolicy policy) {
  return comparison.all_values_finite &&
         (policy == ModelComparisonPolicy::report_differences ||
          comparison.passed);
}

namespace {
bool validate_diagnostic_rows(std::span<const float> reference,
                              std::span<const float> actual) {
  if (reference.empty() || reference.size() != actual.size()) {
    throw std::invalid_argument("diagnostics require equal nonempty rows");
  }
  for (const float value : reference) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("reference values must be finite");
    }
  }
  return std::ranges::all_of(actual,
                             [](float value) { return std::isfinite(value); });
}
std::size_t token_rank(std::span<const float> logits, std::size_t token) {
  const float value = dolittle::at(logits, token);
  std::size_t rank = 1;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const float candidate = dolittle::at(logits, i);
    if (candidate > value || (candidate == value && i < token)) {
      ++rank;
    }
  }
  return rank;
}
}  // namespace

std::optional<RowGeometry> row_geometry(std::span<const float> reference,
                                        std::span<const float> actual) {
  if (!validate_diagnostic_rows(reference, actual)) {
    return std::nullopt;
  }
  double reference_squared = 0;
  double actual_squared = 0;
  double dot = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double expected = dolittle::at(reference, i);
    const double observed = dolittle::at(actual, i);
    reference_squared += expected * expected;
    actual_squared += observed * observed;
    dot += expected * observed;
  }
  RowGeometry result;
  if (reference_squared > 0) {
    result.norm_ratio = std::sqrt(actual_squared / reference_squared);
    if (actual_squared > 0) {
      result.cosine_similarity = std::clamp(
          dot / (std::sqrt(reference_squared) * std::sqrt(actual_squared)),
          -1.0, 1.0);
    }
  }
  return result;
}

std::optional<LogitDiagnostics> logit_diagnostics(
    std::span<const float> reference, std::span<const float> actual) {
  if (!validate_diagnostic_rows(reference, actual)) {
    return std::nullopt;
  }
  const auto reference_max = std::ranges::max_element(reference);
  const auto actual_max = std::ranges::max_element(actual);
  double reference_sum = 0;
  double actual_sum = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    reference_sum += std::exp(static_cast<double>(dolittle::at(reference, i)) -
                              *reference_max);
    actual_sum +=
        std::exp(static_cast<double>(dolittle::at(actual, i)) - *actual_max);
  }
  const double reference_log_sum = std::log(reference_sum);
  const double actual_log_sum = std::log(actual_sum);
  double kl = 0;
  double variation = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double log_p = static_cast<double>(dolittle::at(reference, i)) -
                         *reference_max - reference_log_sum;
    const double log_q = static_cast<double>(dolittle::at(actual, i)) -
                         *actual_max - actual_log_sum;
    const double probability = std::exp(log_p);
    kl += probability * (log_p - log_q);
    variation += std::abs(probability - std::exp(log_q));
  }
  const auto reference_id =
      static_cast<std::size_t>(reference_max - reference.begin());
  const auto actual_id = static_cast<std::size_t>(actual_max - actual.begin());
  return LogitDiagnostics{std::max(0.0, kl),
                          variation * 0.5,
                          reference_id,
                          actual_id,
                          token_rank(actual, reference_id),
                          token_rank(reference, actual_id)};
}

ElementwiseComparison compare_elementwise(std::span<const float> reference,
                                          std::span<const float> actual,
                                          ElementwiseTolerance tolerance) {
  if (reference.size() != actual.size() || !std::isfinite(tolerance.absolute) ||
      !std::isfinite(tolerance.relative) || tolerance.absolute < 0 ||
      tolerance.relative < 0) {
    throw std::invalid_argument("invalid elementwise shape or tolerance");
  }
  for (const float value : reference) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("reference values must be finite");
    }
  }
  ElementwiseComparison result{0, true, 0, true};
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double expected = dolittle::at(reference, i);
    const double observed = dolittle::at(actual, i);
    if (!std::isfinite(observed)) {
      result.all_values_finite = false;
      result.maximum_absolute_error = std::numeric_limits<double>::infinity();
      ++result.mismatched_elements;
      continue;
    }
    const double error = std::abs(observed - expected);
    result.maximum_absolute_error =
        std::max(result.maximum_absolute_error, error);
    if (error > tolerance.absolute + tolerance.relative * std::abs(expected)) {
      ++result.mismatched_elements;
    }
  }
  result.passed = result.mismatched_elements == 0;
  return result;
}
NumericalComparison compare_values(std::span<const float> reference,
                                   std::span<const float> actual,
                                   NumericalTolerance tolerance) {
  if (reference.size() != actual.size() ||
      !std::isfinite(tolerance.max_absolute_error) ||
      !std::isfinite(tolerance.max_relative_l2) ||
      tolerance.max_absolute_error < 0 || tolerance.max_relative_l2 < 0) {
    throw std::invalid_argument("invalid comparison shape or tolerance");
  }
  // Validate the whole reference even if an earlier actual value is invalid.
  for (const float value : reference) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("reference values must be finite");
    }
  }

  double maximum = 0;
  double squared_error = 0;
  double squared_reference = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    if (!std::isfinite(dolittle::at(actual, i))) {
      return {false, std::numeric_limits<double>::infinity(), std::nullopt,
              false};
    }
    const double expected = dolittle::at(reference, i);
    const double difference =
        static_cast<double>(dolittle::at(actual, i)) - expected;
    maximum = std::max(maximum, std::abs(difference));
    squared_error += difference * difference;
    squared_reference += expected * expected;
  }
  std::optional<double> relative;
  if (squared_reference != 0) {
    relative = std::sqrt(squared_error / squared_reference);
  }
  const bool passed = maximum <= tolerance.max_absolute_error &&
                      (!relative || *relative <= tolerance.max_relative_l2);
  return {true, maximum, relative, passed};
}
}  // namespace dolittle::tests
