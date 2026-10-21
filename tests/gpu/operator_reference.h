/*
 * Axis D3: isolated mathematical checks inside the target numerical workflow.
 * Contract: borrow a live Device for this call; execute production operations
 * on deterministic FP32/BF16 inputs, independently compute the equations in
 * FP32 with explicit storage rounding, and return named comparison reports.
 * All queued work completes before owned storage is released. Allocation and
 * device errors propagate. Numerical mismatches report passed=false.
 * Algorithm: numerical-portability.md, Operation checks: construct stored
 * operands, compute the independent equation, upload/execute/download, compare
 * each element using the published operation/dtype tolerance, and separately
 * report norm error. No production reduction tree is emulated. This is part
 * of the existing target integration scenario, not a unit or standalone test.
 * Cost: bounded diagnostic inputs and allocations, outside inference timing.
 * No production CPU arithmetic, fallback, or change to the frozen oracle.
 * Header-design consultation, 2026-10-21: cpp-guidelines I.1 (#ri-explicit)
 * makes reference and tolerance explicit; cpp-perf-guidelines GDSA.2 separates
 * repeatability from accuracy. Source:
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 */
#pragma once

#include <nlohmann/json.hpp>

namespace dolittle::gpu {
class Device;
}
namespace dolittle::tests {
[[nodiscard]] nlohmann::json check_operator_reference(gpu::Device& device);
}
