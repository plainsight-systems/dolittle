/*
 * Axis R4 isolated record invariants: owned rows survive source moves, unavailable
 * rows remain absent, valid variants append, rejected variants leave state intact.
 * Construct only in-memory records; no application flow or GPU is run.
 * Algorithm: logical-overview.md, "3 Retain the experiment and change phases".
 * Optimization: two tiny rows exercise ownership without model-sized fixtures.
 * Corpus implementation consultation, 2026-10-19: cpp-guidelines R.1 (#rr-raii),
 * I.7 (#ri-post) test ownership and atomic publication. cpp-perf-guidelines GPU.1
 * tests operate on already-owned host rows; no replay or transfer.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "experiment/record.h"
#include "tests/support/check.h"

namespace tensors = dolittle::tensors;
namespace models = dolittle::models;
namespace tests = dolittle::tests;
namespace nla = dolittle::nla;
namespace experiment = dolittle::experiment;
namespace inference = dolittle::inference;
namespace text = dolittle::text;

namespace {
dolittle::experiment::TargetRecord target() {

    experiment::TargetRecord t{};
    t.run = {1};
    t.pair.convention = "test";
    t.backend_identity = "unit";
    t.capture_site = {{models::ObservationBoundary::block_output, {0}}, 2, tensors::ElementType::float32};
    auto &g = t.generation;
    g.checkpoint_identity = "weights";
    g.tokenizer_identity = "tokens";
    g.protocol_identity = "prompt";
    g.settings = {{inference::sampling::SelectionMode::greedy, 1, 0, 1}, 1, 1};
    g.stop_ids = {3};
    g.tokens = {1, 2};
    g.prompt_tokens = 1;
    g.consumed_tokens = 2;
    g.reason = inference::StopReason::token_limit;
    g.captures.push_back({t.capture_site, {{0}, {1}}, std::vector<std::byte>(16, std::byte{1})});
    t.decoded = {"ab", {{1, 0, 1, true}, {2, 1, 2, true}}};
    t.segments = {{1, text::SegmentKind::answer}};
    return t;
}
} // namespace
int main() try {

    auto input = target();
    experiment::Record record(std::move(input), {65536, 2});
    const experiment::CaptureKey key{{models::ObservationBoundary::block_output, {0}}, {1}};
    const auto view = record.capture(key);
    tests::check(view && view->bytes.size() == 8 && view->bytes.front() == std::byte{1}, "owned capture row");
    tests::check(!record.capture({key.site, {2}}), "absent rows remain absent");
    experiment::Interpretation failed{};
    failed.capture = key;
    failed.requested_settings = {{inference::sampling::SelectionMode::greedy, 1, 0, 1}, 4, 8};
    failed.failure_message = "loader failed";
    failed.outcome = experiment::AttemptOutcome::failed;
    if (!view) {
        throw std::runtime_error("missing capture");
    }
    const auto id = record.append(failed);
    const auto *stable = &record.variant(id);
    auto invalid = failed;
    invalid.capture.position = {9};
    tests::expect_throws<std::invalid_argument>([&] { (void)record.append(invalid); });
    tests::check(record.variant_count() == 1 && &record.variant(id) == stable, "rejected append is atomic");
    (void)record.append(failed);
    tests::check(&record.variant(id) == stable && view->bytes.front() == std::byte{1},
                 "views survive append");
    tests::expect_throws<std::length_error>([&] { (void)record.append(failed); });
    experiment::Record completed_record(target(), {65536, 2});
    experiment::Interpretation completed{};
    completed.capture = key;
    completed.generation = target().generation;
    completed.generation->captures.clear();
    completed.requested_settings = completed.generation->settings;
    completed.injection = experiment::InjectionProvenance{{0}, 5, 10, tensors::ElementType::float32, "test"};
    completed.decoded = {"b", {{2, 0, 1, true}}};
    completed.outcome = experiment::AttemptOutcome::completed;
    completed.stage = experiment::AttemptStage::complete;
    const auto completed_id = completed_record.append(completed);
    const auto bytes_before = completed_record.retained_bytes();
    completed.decoded.tokens.at(0).id = 99;
    tests::expect_throws<std::invalid_argument>([&] { (void)completed_record.append(completed); });
    tests::check(completed_record.variant_count() == 1 &&
                     completed_record.retained_bytes() == bytes_before &&
                     completed_record.variant(completed_id).interpretation.decoded.bytes == "b",
                 "misaligned output cannot alter a completed variant");
    auto duplicate = target();
    duplicate.generation.captures.at(0).positions.at(1) = {0};
    tests::expect_throws<std::invalid_argument>(
        [&] { experiment::Record r(std::move(duplicate), {65536, 2}); });
    tests::expect_throws<std::length_error>([] { experiment::Record r(target(), {1, 2}); });
} catch (const std::exception &error) {
    return dolittle::tests::report_failure(error.what());
} catch (...) {
    return dolittle::tests::report_failure("unexpected non-standard exception");
}
