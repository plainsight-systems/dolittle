/*
 * Completed-result comparison for the first numerical harness.
 * Axis: D3, independent forward-pass verification, not production inference.
 *
 * Contract: a reference case owns exact token IDs, the verified checkpoint
 * identity, output selections, expected original dtypes and FP32-decoded
 * reference values. Fixtures must come from an independently pinned reference
 * implementation, not from the implementation under test. Expected row counts,
 * positions and site identities must match the request.
 * Reference provenance records generator revision, dependency/device environment,
 * attention backend and numerical policy; these fields must be nonempty.
 * Reference and actual values must be finite. Tolerances are explicit, finite
 * and nonnegative;
 * numerical bounds must be justified from reference evidence, never selected
 * after a failing result merely to make it pass.
 *
 * compare_forward requires exact execution/checkpoint identity, token IDs,
 * numerical mode, sites, row positions, widths and original dtypes before
 * comparing numbers. It reports
 * shape/identity/nonfinite failures distinctly. Compare max absolute error and
 * relative L2 in double accumulation. Relative L2 is absent for a zero-norm
 * reference and the absolute bound applies; otherwise both bounds must pass.
 * No narrative or reconstruction score is used as evidence of correctness.
 * Invalid fixtures/tolerances throw invalid_argument; checked size overflow
 * throws length_error; allocation failure propagates. Actual-result mismatch
 * returns failed comparisons rather than throwing. Results include one entry
 * for run provenance even when no output rows were requested.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "First forward pass"
 * and "Verification boundaries". Supply fixed multi-token inputs, execute a
 * prepared no-cache pass, decode completed host bytes without normalization,
 * compare logits and selected sites, and retain named per-output failures.
 * Include block inputs/outputs, alternate blocks, empty capture requests,
 * prefix causality and capture-on/off cases. Separately test invalid sites,
 * bounds/overflow, copied metadata lifetime and output lifetime after teardown.
 *
 * Design and cost: fixture loading, FP32 conversion and comparisons are host
 * verification work, excluded from reported kernel time. Fixture vectors own
 * their data; comparisons borrow immutable results. Inspect only requested
 * outputs; full activation readback is not a production requirement.
 * compare_forward inspects an already completed result; it never executes a
 * model. tests/support/numerical.h owns the pure array error metrics. Device
 * execution that produces the result is integration work and counts against
 * the repository-wide limit; this helper does not create additional scenarios.
 * CompletedForwardView borrows immutable, completed host data only for this
 * call; no spans are retained. The ForwardResult overload adapts its accessors
 * to the identical comparison exercised by in-memory unit tests.
 *
 * Corpus guidance queried/read for implementation, 2026-10-18:
 * cpp-guidelines I.4 (#ri-typed): semantic point and token positions distinguish
 * outputs even when their widths match.
 * cpp-guidelines I.7 (#ri-post): comparison states exactly what passing means.
 * cpp-perf-guidelines GPU.1: isolate diagnostic readback/conversion from compute
 * timing. Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include "inference/executor.h"
#include "tests/support/numerical.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dolittle::tests {

struct ExpectedRows {
    std::vector<models::TokenPosition> positions;
    std::size_t width;
    tensors::ElementType original_storage_type;
    std::vector<float> values;
};

struct ExpectedObservation {
    models::ObservationPoint point;
    ExpectedRows rows;
};

struct ReferenceProvenance {
    std::string implementation_revision;
    std::string environment_identity;
    std::string attention_backend;
    std::string numerical_policy;
};

struct ForwardReferenceCase {
    std::string name;
    ReferenceProvenance reference;
    std::string checkpoint_identity;
    inference::ExecutionId execution_id;
    std::vector<std::uint32_t> token_ids;
    inference::ForwardOptions options;
    ExpectedRows logits;
    std::vector<ExpectedObservation> captures;
};

struct OutputComparison {
    std::string output_name;
    bool identity_and_shape_match;
    bool all_values_finite;
    double maximum_absolute_error;
    std::optional<double> relative_l2;
    bool passed;
    std::string failure;
};

struct CompletedForwardView {
    inference::ExecutionId execution_id;
    std::string_view checkpoint_identity;
    std::span<const std::uint32_t> token_ids;
    tensors::ComputationMode numerical_mode;
    inference::LogitRows logits;
    std::span<const inference::CapturedRows> captures;
};

[[nodiscard]] std::vector<OutputComparison> compare_forward(
    const ForwardReferenceCase& reference,
    CompletedForwardView actual,
    NumericalTolerance logit_tolerance,
    NumericalTolerance observation_tolerance);

[[nodiscard]] inline std::vector<OutputComparison> compare_forward(
    const ForwardReferenceCase& reference, const inference::ForwardResult& actual,
    NumericalTolerance logit_tolerance, NumericalTolerance observation_tolerance) {
    return compare_forward(reference,
        CompletedForwardView{actual.execution_id(), actual.checkpoint_identity(),
            actual.token_ids(), actual.numerical_mode(), actual.logits(), actual.captures()},
        logit_tolerance, observation_tolerance);
}

}  // namespace dolittle::tests
