/*
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
 */
#include "tests/support/numerical.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "src/bounds.h"

namespace dolittle::tests {
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
