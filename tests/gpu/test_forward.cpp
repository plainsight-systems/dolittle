/*
 * Axis D3: real checkpoint-to-CUDA forward integration driver.
 * Contract: one fixture bundle produced by the independent reference generator;
 * artifacts must pass the production size/hash loader before binding. A run
 * succeeds only if every provenance, shape, finite-value and fixed numerical
 * bound passes, and capture selection does not change logits. Invalid fixtures
 * or runtime failures exit nonzero; report JSON contains measured errors/costs.
 * No weights/reference values are manufactured by the engine under test.
 * Algorithm: transformer-execution.md, "First forward pass" and "Verification
 * boundaries": load verified weights, prepare configurable observations, run
 * complete causal passes, retain results across teardown, compare independent
 * expected rows and prefix/adjacent-boundary identities. Empty selections and
 * rejected requests exercise the same full-forward interaction. Tiny shapes and
 * the pinned 8B model are two integration scenarios, not host unit tests.
 * Cost: fixture parsing/comparison excluded from execution timing; wall times
 * distinguish loading, preparation/upload and host-ready execution. Optimization:
 * one resident plan at a time, reuse a plan for equal-shaped alternate tokens;
 * verification readbacks remain bounded by fixture selections.
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): nested owners survive failures and release in
 * dependency order. I.7 (#ri-post): success requires all comparison contracts.
 * cpp-perf-guidelines GPU.1: separate verification readbacks from compute claims.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "gpu/cuda/device.h"
#include "models/qwen3/checkpoint.h"
#include "tensors/formats/safetensors.h"
#include "tensors/tensor.h"
#include "tests/gpu/forward_reference.h"
#include "tests/support/check.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

namespace {
using namespace dolittle;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
tensors::ElementType dtype(const Json& value) {
    if (value == "float32") return tensors::ElementType::float32;
    if (value == "bfloat16") return tensors::ElementType::bfloat16;
    throw std::invalid_argument("unsupported reference dtype");
}
std::vector<models::TokenPosition> positions(const Json& values) {
    std::vector<models::TokenPosition> result;
    for (const auto& value : values) result.push_back({value.get<std::size_t>()});
    return result;
}
models::ObservationPoint point(const Json& value) {
    const auto boundary = value.at("boundary").get<std::string>();
    if (boundary != "block_input" && boundary != "block_output")
        throw std::invalid_argument("unknown observation boundary");
    return {boundary == "block_input" ? models::ObservationBoundary::block_input :
            models::ObservationBoundary::block_output, {value.at("block").get<std::size_t>()}};
}
tests::ExpectedRows rows(const Json& value) {
    return {positions(value.at("positions")), value.at("width").get<std::size_t>(),
            dtype(value.at("original_storage_type")), value.at("values").get<std::vector<float>>()};
}
tests::NumericalTolerance tolerance(const Json& value) {
    return {value.at("max_absolute_error").get<double>(), value.at("max_relative_l2").get<double>()};
}
bool equal_bytes(std::span<const std::byte> a, std::span<const std::byte> b) {
    return std::ranges::equal(a, b);
}
tests::ForwardReferenceCase reference_case(const Json& bundle, const Json& value,
    const std::string& identity, std::uint64_t id) {
    const auto& provenance = bundle.at("reference");
    tests::ForwardReferenceCase result;
    result.name = value.at("name");
    result.reference = {provenance.at("implementation_revision"), provenance.at("environment_identity"),
                        provenance.at("attention_backend"), provenance.at("numerical_policy")};
    result.checkpoint_identity = identity;
    result.execution_id = {id};
    result.token_ids = value.at("token_ids").get<std::vector<std::uint32_t>>();
    result.options = {{dtype(bundle.at("dtype")), tensors::ElementType::float32},
        positions(value.at("logit_positions")), {}, {64 * 1024 * 1024, 1024 * 1024},
        std::size_t{23} * 1024 * 1024 * 1024};
    result.logits = rows(value.at("logits"));
    for (const auto& capture : value.at("captures")) {
        result.captures.push_back({point(capture), rows(capture)});
        result.options.observations.push_back({point(capture), positions(capture.at("positions"))});
    }
    return result;
}
Json comparisons(const tests::ForwardReferenceCase& expected, const inference::ForwardResult& actual,
    const Json& bundle, const Json& value, bool& passed) {
    auto metrics = tests::compare_forward(expected, actual,
        tolerance(bundle.at("tolerances").at("logits")),
        tolerance(bundle.at("tolerances").at("observations")));
    Json outputs = Json::array();
    for (std::size_t i = 0; i < metrics.size(); ++i) {
        auto& metric = metrics[i];
        // Tighten the bundle fallback with independently fixed per-output scales.
        const Json* source = i == 0 ? nullptr : i == 1 ? &value.at("logits") : &value.at("captures").at(i - 2);
        if (source && source->contains("tolerance")) {
            const auto limit = tolerance(source->at("tolerance"));
            metric.passed = metric.passed && metric.maximum_absolute_error <= limit.max_absolute_error &&
                (!metric.relative_l2 || *metric.relative_l2 <= limit.max_relative_l2);
        }
        passed = passed && metric.passed;
        outputs.push_back({{"output", metric.output_name}, {"passed", metric.passed},
            {"maximum_absolute_error", metric.maximum_absolute_error},
            {"relative_l2", metric.relative_l2 ? Json(*metric.relative_l2) : Json(nullptr)},
            {"failure", metric.failure}});
    }
    return outputs;
}
// A semantic block input is exactly the preceding block output, where selected.
void check_boundaries(const inference::ForwardResult& result) {
    for (const auto& input : result.captures()) {
        if (input.site.point.boundary != models::ObservationBoundary::block_input || !input.site.point.block.value) continue;
        for (const auto& output : result.captures()) {
            if (output.site.point.boundary != models::ObservationBoundary::block_output ||
                output.site.point.block.value + 1 != input.site.point.block.value) continue;
            if (std::ranges::equal(input.positions, output.positions, {}, &models::TokenPosition::value,
                                  &models::TokenPosition::value))
                tests::check(equal_bytes(input.bytes, output.bytes), "adjacent block boundary values differ");
        }
    }
}
void check_prefix(const inference::ForwardResult& shorter, const inference::ForwardResult& longer) {
    if (shorter.token_ids().size() >= longer.token_ids().size() ||
        !std::ranges::equal(shorter.token_ids(), longer.token_ids().first(shorter.token_ids().size()))) return;
    for (const auto& a : shorter.captures()) {
        for (const auto& b : longer.captures()) {
            if (a.site.point.block.value != b.site.point.block.value || a.site.point.boundary != b.site.point.boundary) continue;
            const auto row_bytes = a.site.vector_width * tensors::element_bytes(a.site.storage_type);
            for (std::size_t i = 0; i < a.positions.size(); ++i)
                for (std::size_t j = 0; j < b.positions.size(); ++j)
                    if (a.positions[i].value == b.positions[j].value)
                        tests::check(equal_bytes(a.bytes.subspan(i * row_bytes, row_bytes),
                            b.bytes.subspan(j * row_bytes, row_bytes)), "future tokens changed earlier capture");
        }
    }
}
Json run(const std::filesystem::path& fixture) {
    tests::check(std::filesystem::file_size(fixture) <= 128 * 1024 * 1024, "fixture exceeds harness budget");
    std::ifstream stream(fixture);
    const auto bundle = Json::parse(stream);
    tests::check(bundle.at("schema_version") == 1, "unsupported fixture schema");
    tensors::formats::safetensors::OpenRequest request{fixture.parent_path(), bundle.at("repository"),
        bundle.at("revision"), {}, bundle.at("shards").get<std::vector<std::string>>(),
        std::nullopt, 16 * 1024 * 1024, 128 * 1024 * 1024, 100000};
    if (!bundle.at("index").is_null()) request.index = bundle.at("index").get<std::string>();
    for (const auto& artifact : bundle.at("artifacts"))
        request.artifacts.push_back({artifact.at("relative_path"), artifact.at("byte_size"), artifact.at("sha256")});
    const auto load_start = Clock::now();
    auto checkpoint = models::qwen3::bind_target(tensors::formats::safetensors::open(request),
                                                bundle.at("config_artifact").get<std::string>());
    Json report{{"load_seconds", seconds(load_start)}, {"checkpoint_identity", checkpoint->identity()},
                {"cases", Json::array()}, {"reference", bundle.at("reference")},
                {"tolerances", bundle.at("tolerances")}};
    auto device = gpu::cuda::open_device(0);
    report["device"] = {{"name", device->info().name}, {"architecture", device->info().architecture},
                         {"driver", device->info().driver_version}};
    std::vector<tests::ForwardReferenceCase> expected;
    std::vector<inference::ForwardResult> results;
    std::uint64_t id = 1;
    for (const auto& value : bundle.at("cases")) {
        auto reference = reference_case(bundle, value, std::string(checkpoint->identity()), id++);
        const models::SequenceLength length{reference.token_ids.size()};
        auto invalid = reference.options;
        invalid.max_device_bytes = 0;
        tests::expect_throws<std::length_error>([&] { (void)inference::prepare_full_forward(*device, *checkpoint, length, invalid); });
        Json measurement{{"name", reference.name}};
        {
            const auto prepare_start = Clock::now();
            auto plan = inference::prepare_full_forward(*device, *checkpoint, length, reference.options);
            measurement["prepare_seconds"] = seconds(prepare_start);
            measurement["estimated_peak_device_bytes"] = plan.estimated_peak_device_bytes();
            measurement["host_output_bytes"] = plan.host_output_bytes();
            auto bad_tokens = reference.token_ids;
            bad_tokens[0] = static_cast<std::uint32_t>(checkpoint->model().vocabulary_size());
            tests::expect_throws<std::invalid_argument>([&] { (void)plan.execute(bad_tokens, reference.execution_id); });
            const auto execute_start = Clock::now();
            results.push_back(plan.execute(reference.token_ids, reference.execution_id));
            measurement["execute_host_ready_seconds"] = seconds(execute_start);
            // Reuse with different contents, then reproduce the original pass.
            auto changed = reference.token_ids;
            changed[0] = (changed[0] + 1) % checkpoint->model().vocabulary_size();
            (void)plan.execute(changed, {10000 + reference.execution_id.value});
            auto repeated = plan.execute(reference.token_ids, reference.execution_id);
            tests::check(equal_bytes(repeated.logits().bytes, results.back().logits().bytes), "serial plan reuse changed logits");
            for (std::size_t i = 0; i < repeated.captures().size(); ++i)
                tests::check(equal_bytes(repeated.captures()[i].bytes, results.back().captures()[i].bytes), "serial reuse changed captures");
        }
        {
            auto no_capture = reference.options;
            no_capture.observations.clear();
            auto plan = inference::prepare_full_forward(*device, *checkpoint, length, no_capture);
            auto result = plan.execute(reference.token_ids, reference.execution_id);
            tests::check(result.captures().empty() && equal_bytes(result.logits().bytes, results.back().logits().bytes),
                         "observation selection changed logits");
        }
        expected.push_back(std::move(reference));
        report["cases"].push_back(std::move(measurement));
    }
    tests::check(!results.empty(), "fixture contains no cases");
    {
        auto empty = expected.front().options;
        empty.logit_positions = std::vector<models::TokenPosition>{};
        empty.observations.clear();
        const auto site = checkpoint->model().observation_sites(empty.numerical_mode).front();
        empty.observations.push_back({site.point, std::vector<models::TokenPosition>{}});
        auto plan = inference::prepare_full_forward(*device, *checkpoint,
            {expected.front().token_ids.size()}, empty);
        auto result = plan.execute(expected.front().token_ids, {99999});
        tests::check(result.logits().bytes.empty() && result.captures().size() == 1 && result.captures()[0].bytes.empty(),
                     "empty selections did not preserve empty metadata");
    }
    // All comparisons below deliberately run after the execution owners die.
    checkpoint.reset();
    device.reset();
    bool passed = true;
    for (std::size_t i = 0; i < results.size(); ++i) {
        report["cases"][i]["outputs"] = comparisons(expected[i], results[i], bundle, bundle.at("cases")[i], passed);
        check_boundaries(results[i]);
        for (const auto& other : results) check_prefix(results[i], other);
    }
    report["passed"] = passed;
    return report;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::invalid_argument("usage: test_forward FIXTURE.json [FIXTURE.json ...]");
        Json reports = Json::array();
        bool passed = true;
        for (int i = 1; i < argc; ++i) {
            auto report = run(argv[i]);
            passed = passed && report.at("passed").get<bool>();
            reports.push_back(std::move(report));
        }
        std::cout << reports.dump(2) << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "forward integration failed: " << error.what() << '\n';
        return 1;
    }
}
