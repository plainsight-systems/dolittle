/*
 * Axis D3: validate provenance/shape and compare completed forward results.
 * Contract: implement forward_reference.h. The caller supplies a completed
 * result; this helper launches nothing. Malformed reference inputs fail before
 * comparison; actual identity/shape/value mismatches become explicit failures.
 * Algorithm: docs/architecture/transformer-execution.md, "Verification
 * boundaries": validate fixture provenance/selections/lengths, compare run and
 * checkpoint identity, exact token IDs/mode/sites/positions/dtypes, reject
 * missing/extra outputs, decode only matched rows to FP32, and delegate error
 * metrics to support/numerical. Always emit run-provenance status even with no
 * output rows. Preserve raw bytes; decoding never rewrites captured values.
 * Design and cost: O(requested output bytes + C log C) for C capture sites,
 * bounded temporary conversion
 * vectors and owned diagnostics; no full-model activation readback.
 * Optimization: compare one output at a time and keep conversion outside GPU
 * compute timing. Reference fixture generation is independent of this helper.
 * The harness that obtains device results is integration work within the cap;
 * this source defines no extra scenario and numerical helper unit tests launch
 * nothing. Do not introduce one integration test per layer or operation here.
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit): provenance and thresholds are explicit.
 * cpp-guidelines I.7 (#ri-post): metadata match precedes numerical acceptance.
 * cpp-guidelines C.183 (#ru-pun): decode raw FP32 with bit_cast, never aliasing casts.
 * cpp-perf-guidelines GPU.1: output decoding is separate diagnostic work.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "bounds.h"
#include "tests/gpu/forward_reference.h"
#include "tensors/representations/bf16.h"
#include "tensors/tensor.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace dolittle::tests {
namespace {
void require(bool condition, const char *message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::length_error("reference output extent overflows");
    }
    return a * b;
}
bool positions_equal(std::span<const models::TokenPosition> a, std::span<const models::TokenPosition> b) {
    return std::ranges::equal(a, b, [](auto x, auto y) { return x.value == y.value; });
}
using SiteKey = std::pair<models::ObservationBoundary, std::size_t>;
SiteKey key(models::ObservationPoint point) { return {point.boundary, point.block.value}; }

void validate_rows(const ExpectedRows &rows, std::size_t length, tensors::ElementType storage) {
    require(rows.width != 0, "reference row width must be positive");
    require(rows.original_storage_type == storage, "reference output mode differs");
    const auto elements = product(rows.positions.size(), rows.width);
    (void)product(elements, tensors::element_bytes(storage));
    require(rows.values.size() == elements, "reference value count differs from shape");
    for (std::size_t i = 0; i < rows.positions.size(); ++i) {
        require(rows.positions.at(i).value < length &&
                    (i == 0 || rows.positions.at(i - 1).value < rows.positions.at(i).value),
                "reference row positions are invalid");
    }
    for (float value : rows.values) {
        require(std::isfinite(value), "reference values must be finite");
    }
}

struct OutputTolerances {
    NumericalTolerance logits;
    NumericalTolerance observations;
};
void validate_fixture(const ForwardReferenceCase &r, OutputTolerances tolerances) {
    const auto [logits, observations] = tolerances;
    (void)compare_values({}, {}, logits);
    (void)compare_values({}, {}, observations);
    require(!r.reference.implementation_revision.empty() && !r.reference.environment_identity.empty() &&
                !r.reference.attention_backend.empty() && !r.reference.numerical_policy.empty(),
            "reference provenance must be complete");
    require(!r.checkpoint_identity.empty() && !r.token_ids.empty(),
            "reference requires checkpoint and token provenance");
    const auto mode = r.options.numerical_mode;
    require((mode.activation_storage == tensors::ElementType::bfloat16 ||
             mode.activation_storage == tensors::ElementType::float32) &&
                mode.reduction_accumulation == tensors::ElementType::float32,
            "reference numerical mode is unsupported");
    validate_rows(r.logits, r.token_ids.size(), mode.activation_storage);
    for (auto token : r.token_ids) {
        require(token < r.logits.width, "reference token is outside vocabulary");
    }
    if (std::holds_alternative<inference::AllPositions>(r.options.logit_positions)) {
        require(r.logits.positions.size() == r.token_ids.size(), "reference logit selection differs");
        for (std::size_t i = 0; i < r.token_ids.size(); ++i) {
            require(r.logits.positions.at(i).value == i, "reference logit selection differs");
        }
    } else {
        require(positions_equal(r.logits.positions,
                                std::get<std::vector<models::TokenPosition>>(r.options.logit_positions)),
                "reference logit selection differs");
    }

    std::map<SiteKey, const ExpectedRows *> expected;
    for (const auto &capture : r.captures) {
        validate_rows(capture.rows, r.token_ids.size(), mode.activation_storage);
        require(capture.point.boundary == models::ObservationBoundary::block_input ||
                    capture.point.boundary == models::ObservationBoundary::block_output,
                "reference observation boundary is invalid");
        require(expected.emplace(key(capture.point), &capture.rows).second,
                "reference repeats a capture site");
    }
    require(r.options.observations.size() == r.captures.size(), "reference capture selection differs");
    for (const auto &request : r.options.observations) {
        const auto found = expected.find(key(request.point));
        require(found != expected.end(), "reference capture request is absent or repeated");
        const auto &positions = found->second->positions;
        if (std::holds_alternative<inference::AllPositions>(request.positions)) {
            require(positions.size() == r.token_ids.size(), "reference capture row selection differs");
            for (std::size_t i = 0; i < positions.size(); ++i) {
                require(positions.at(i).value == i, "reference capture row selection differs");
            }
        } else {
            require(
                positions_equal(positions, std::get<std::vector<models::TokenPosition>>(request.positions)),
                "reference capture row selection differs");
        }
        expected.erase(found);
    }
}

OutputComparison mismatch(std::string name, std::string failure) {
    return {std::move(name),   false, false, std::numeric_limits<double>::infinity(), std::nullopt, false,
            std::move(failure)};
}

OutputComparison compare_rows(std::string name, const ExpectedRows &expected,
                              std::span<const models::TokenPosition> positions, std::size_t width,
                              tensors::ElementType storage, std::span<const std::byte> bytes,
                              NumericalTolerance tolerance, bool run_matches) {
    if (!run_matches) {
        return mismatch(std::move(name), "run provenance differs");
    }
    const auto expected_bytes =
        product(expected.values.size(), tensors::element_bytes(expected.original_storage_type));
    if (width != expected.width || storage != expected.original_storage_type ||
        !positions_equal(positions, expected.positions) || bytes.size() != expected_bytes) {
        return mismatch(std::move(name), "output metadata or byte extent differs");
    }
    std::vector<float> decoded;
    decoded.reserve(expected.values.size());
    if (storage == tensors::ElementType::bfloat16) {
        for (std::size_t i = 0; i < bytes.size(); i += 2) {
            decoded.push_back(tensors::to_float(
                tensors::read_le(std::span<const std::byte, 2>(bytes.subspan(i).data(), 2))));
        }
    } else {
        for (std::size_t i = 0; i < bytes.size(); i += 4) {
            std::uint32_t bits = 0;
            for (unsigned byte = 0; byte < 4; ++byte) {
                bits |= std::to_integer<std::uint32_t>(dolittle::at(bytes, i + byte)) << (8 * byte);
            }
            decoded.push_back(std::bit_cast<float>(bits));
        }
    }
    const auto numerical = compare_values(expected.values, decoded, tolerance);
    return {std::move(name),
            true,
            numerical.all_values_finite,
            numerical.maximum_absolute_error,
            numerical.relative_l2,
            numerical.passed,
            numerical.passed ? ""
                             : (numerical.all_values_finite ? "numerical bounds exceeded"
                                                            : "actual values are nonfinite")};
}
std::string capture_name(models::ObservationPoint point) {
    return std::string("capture.") +
           (point.boundary == models::ObservationBoundary::block_input ? "input." : "output.") +
           std::to_string(point.block.value);
}
} // namespace

std::vector<OutputComparison> compare_forward(const ForwardReferenceCase &reference,
                                              CompletedForwardView actual, NumericalTolerance logit_tolerance,
                                              NumericalTolerance observation_tolerance) {
    validate_fixture(reference, {.logits = logit_tolerance, .observations = observation_tolerance});
    const auto mode = reference.options.numerical_mode;
    const bool run_matches = reference.execution_id.value == actual.execution_id.value &&
                             reference.checkpoint_identity == actual.checkpoint_identity &&
                             std::ranges::equal(reference.token_ids, actual.token_ids) &&
                             mode.activation_storage == actual.numerical_mode.activation_storage &&
                             mode.reduction_accumulation == actual.numerical_mode.reduction_accumulation;
    std::vector<OutputComparison> result;
    result.push_back(run_matches
                         ? OutputComparison{"run", true, true, 0, std::nullopt, true, ""}
                         : mismatch("run", "execution/checkpoint/token/numerical provenance differs"));
    result.push_back(compare_rows("logits", reference.logits, actual.logits.positions,
                                  actual.logits.vocabulary_size, actual.logits.storage_type,
                                  actual.logits.bytes, logit_tolerance, run_matches));
    std::map<SiteKey, const inference::CapturedRows *> captures;
    for (const auto &capture : actual.captures) {
        if (!captures.emplace(key(capture.site.point), &capture).second) {
            result.push_back(mismatch(capture_name(capture.site.point), "duplicate actual capture site"));
        }
    }
    for (const auto &expected : reference.captures) {
        const auto found = captures.find(key(expected.point));
        if (found == captures.end()) {
            result.push_back(mismatch(capture_name(expected.point), "missing actual capture site"));
            continue;
        }
        const auto &capture = *found->second;
        result.push_back(compare_rows(capture_name(expected.point), expected.rows, capture.positions,
                                      capture.site.vector_width, capture.site.storage_type, capture.bytes,
                                      observation_tolerance, run_matches));
        captures.erase(found);
    }
    for (const auto &[point, capture] : captures) {
        (void)point;
        result.push_back(mismatch(capture_name(capture->site.point), "unexpected actual capture site"));
    }
    return result;
}
} // namespace dolittle::tests
