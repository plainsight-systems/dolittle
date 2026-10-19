/*
 * Axis D3: isolated completed-result comparison contract verification.
 * Contract: all fixtures and completed views live in memory; no model, files,
 * processes or GPU execution. Assertions cover provenance, shape, finite values
 * and numerical bounds rather than diagnostic wording or private helper order.
 * Algorithm: transformer overlay "Verification boundaries": compare matching
 * synthetic arrays, alter one public metadata/value property, and ensure the
 * appropriate acceptance category changes; malformed references must throw.
 * Design/cost: tiny owned arrays, byte encodings and borrowed views; constant
 * test workload. Optimization: no checkpoint download or model execution is
 * necessary to establish the comparison unit's postconditions.
 * Corpus consultation, design/implementation, 2026-10-18:
 * cpp-guidelines I.4 (#ri-typed): preserve semantic sites and token positions.
 * cpp-guidelines I.7 (#ri-post): test exact public acceptance/error contracts.
 * cpp-guidelines C.183 (#ru-pun): byte fixtures inspect FP32 with bit_cast.
 * cpp-perf-guidelines GPU.1: isolate numerical diagnostics from GPU execution.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <stdexcept>

#include "tensors/representations/bf16.h"
#include "tests/gpu/forward_reference.h"
#include "tests/support/check.h"

namespace tensors = dolittle::tensors;
namespace models = dolittle::models;
namespace inference = dolittle::inference;
using dolittle::tests::check;
using dolittle::tests::compare_forward;
using dolittle::tests::CompletedForwardView;
using dolittle::tests::expect_throws;
using dolittle::tests::ForwardReferenceCase;
using dolittle::tests::NumericalTolerance;
using dolittle::tests::OutputComparison;

namespace {
std::vector<std::byte> encode(std::span<const float> values,
                              tensors::ElementType type) {
  std::vector<std::byte> result;
  for (float value : values) {
    if (type == tensors::ElementType::bfloat16) {
      std::array<std::byte, 2> bytes{};
      tensors::write_le(tensors::from_float(value), bytes);
      result.insert(result.end(), bytes.begin(), bytes.end());
    } else {
      const auto bits = std::bit_cast<std::uint32_t>(value);
      for (unsigned i = 0; i < 4; ++i) {
        result.push_back(static_cast<std::byte>((bits >> (8 * i)) & 0xffU));
      }
    }
  }
  return result;
}
bool passed(const std::vector<OutputComparison>& comparisons) {
  return !comparisons.empty() &&
         std::ranges::all_of(comparisons,
                             [](const auto& c) { return c.passed; });
}
}  // namespace

int main() try {
  constexpr NumericalTolerance exact{0, 0};
  for (const auto storage :
       {tensors::ElementType::bfloat16, tensors::ElementType::float32}) {
    const models::ObservationPoint site{
        models::ObservationBoundary::block_output, {1}};
    ForwardReferenceCase reference{
        "synthetic comparison contract",
        {"independent-v1", "unit-memory", "none", "exact"},
        "checkpoint-identity",
        {7},
        {0, 1},
        {{storage, tensors::ElementType::float32},
         std::vector<models::TokenPosition>{{1}},
         {{site, inference::AllPositions{}}},
         {std::size_t{1024}, 16},
         4096},
        {{{1}}, 4, storage, {1, 2, -1, 0}},
        {{site, {{{0}, {1}}, 2, storage, {0, 1, 2, -1}}}}};
    auto logits = encode(reference.logits.values, storage);
    auto captured = encode(reference.captures.at(0).rows.values, storage);
    std::vector<inference::CapturedRows> captures{
        {{site, 2, storage},
         reference.captures.at(0).rows.positions,
         captured}};
    const CompletedForwardView actual{
        reference.execution_id,
        reference.checkpoint_identity,
        reference.token_ids,
        reference.options.numerical_mode,
        {reference.logits.positions, 4, storage, logits},
        captures};
    check(passed(compare_forward(reference, actual, exact, exact)),
          "matching complete result passes");

    auto changed = actual;
    changed.execution_id.value++;
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "execution identity matters");
    changed = actual;
    changed.checkpoint_identity = "another-checkpoint";
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "checkpoint identity matters");
    changed = actual;
    const std::array<std::uint32_t, 2> other_tokens{1, 0};
    changed.token_ids = other_tokens;
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "exact token provenance matters");
    changed = actual;
    changed.numerical_mode.reduction_accumulation =
        tensors::ElementType::bfloat16;
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "numerical mode matters");
    changed = actual;
    changed.logits.vocabulary_size++;
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "row width matters");
    changed = actual;
    changed.logits.storage_type = tensors::ElementType::uint32;
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "original dtype matters");
    changed = actual;
    const std::array<models::TokenPosition, 1> wrong_position{{{0}}};
    changed.logits.positions = wrong_position;
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "consumed positions matter");
    changed = actual;
    changed.logits.bytes =
        std::span<const std::byte>(logits).first(logits.size() - 1);
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "byte extent matters");
    changed = actual;
    changed.captures = {};
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "missing sites fail");
    auto wrong_captures = captures;
    wrong_captures.at(0).site.point.boundary =
        models::ObservationBoundary::block_input;
    changed.captures = wrong_captures;
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "site semantics matter");
    wrong_captures = captures;
    wrong_captures.push_back(captures.at(0));
    changed.captures = wrong_captures;
    check(!passed(compare_forward(reference, changed, exact, exact)),
          "duplicate sites fail");

    changed = actual;
    const std::array<float, 4> differing_values{2, 2, -1, 0};
    auto differing_bytes = encode(differing_values, storage);
    changed.logits.bytes = differing_bytes;
    const auto differing = compare_forward(reference, changed, exact, exact);
    check(std::ranges::any_of(differing,
                              [](const auto& c) {
                                return c.identity_and_shape_match &&
                                       c.all_values_finite && !c.passed &&
                                       c.maximum_absolute_error == 1;
                              }),
          "numerical mismatch differs from metadata mismatch");
    check(passed(compare_forward(reference, changed, {1, 1}, exact)),
          "explicit numerical limits apply");
    const std::array<float, 4> nonfinite_values{
        std::numeric_limits<float>::infinity(), 2, -1, 0};
    auto nonfinite_bytes = encode(nonfinite_values, storage);
    changed.logits.bytes = nonfinite_bytes;
    const auto nonfinite = compare_forward(reference, changed, exact, exact);
    check(std::ranges::any_of(nonfinite,
                              [](const auto& c) {
                                return c.identity_and_shape_match &&
                                       !c.all_values_finite && !c.passed;
                              }),
          "nonfinite actual is a reported failure");

    auto bad_reference = reference;
    bad_reference.reference.implementation_revision.clear();
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_forward(bad_reference, actual, exact, exact); });
    bad_reference = reference;
    bad_reference.logits.values.at(0) = std::numeric_limits<float>::quiet_NaN();
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_forward(bad_reference, actual, exact, exact); });
    bad_reference = reference;
    bad_reference.captures.at(0).rows.values.pop_back();
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_forward(bad_reference, actual, exact, exact); });
    bad_reference = reference;
    bad_reference.options.logit_positions = inference::AllPositions{};
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_forward(bad_reference, actual, exact, exact); });
    expect_throws<std::invalid_argument>(
        [&] { (void)compare_forward(reference, actual, {-1, 0}, exact); });
    bad_reference = reference;
    bad_reference.logits.width = std::numeric_limits<std::size_t>::max();
    expect_throws<std::length_error>(
        [&] { (void)compare_forward(bad_reference, actual, exact, exact); });

    auto empty = reference;
    empty.options.logit_positions = std::vector<models::TokenPosition>{};
    empty.options.observations.clear();
    empty.logits.positions.clear();
    empty.logits.values.clear();
    empty.captures.clear();
    changed = actual;
    changed.logits.positions = {};
    changed.logits.bytes = {};
    changed.captures = {};
    check(passed(compare_forward(empty, changed, exact, exact)),
          "empty requested outputs retain provenance");
    changed.execution_id.value++;
    check(!passed(compare_forward(empty, changed, exact, exact)),
          "empty outputs still reject wrong run");
  }
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
