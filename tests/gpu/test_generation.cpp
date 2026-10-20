/*
 * Axis D3: real target-to-AV CUDA experiment integration driver.
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
 * Generation continuation: verify reference tokenizer/prompt IDs, replay the
 * complete fixed trace with uneven cached prefill and single-row decode,
 * compare every consumed capture and chunk-final logit row against independent
 * values, then greedily generate the trace. Check final-ID consumption, owned
 * teardown, no-capture equivalence, and tiny stop/cancel/selector-failure
 * boundaries. Compare each cached chunk byte-for-byte with an uncached pass
 * over the same trace, then compare both paths with the independent numerical
 * reference. Fixed numerical failures accumulate in report JSON and still exit
 * nonzero. For generation, decode each returned row of width W and compare its
 * W values with the oracle at the same absolute token position. Require both
 * the maximum absolute error and relative L2 limit for every row; a whole-chunk
 * norm cannot hide a bad row. Report row metrics as well as the aggregate. The
 * uncached trace also checks every logit row, including rows not returned by
 * chunked prefill. These checks reuse completed host readbacks, outside compute
 * timing. AV continuation: after releasing target execution, check the frozen
 * capture's FP32 normalization and stored replacement bytes. Compare injected
 * full/cached AV logits and captures on those identical inputs, then run two
 * interpretations of an engine-retained row with fresh caches and one resident
 * AV weight owner. Capture bytes must survive model switching and close. AV
 * fixture expectations are independent; target acceptance limits and reference
 * arithmetic stay fixed. The tiny and 8B experiment scenarios replace the two
 * generation scenarios. Implementation consultation, 2026-10-19: R.1 and I.7
 * below govern retained ownership and fixed acceptance gates; GPU.1 keeps
 * comparisons off the GPU path. Cost: fixture parsing/comparison excluded from
 * execution timing; wall times distinguish loading, preparation/upload and
 * host-ready execution. Optimization: one resident plan at a time, reuse a plan
 * for equal-shaped alternate tokens; verification readbacks remain bounded by
 * fixture selections. Corpus consultation, header design and implementation,
 * 2026-10-18: cpp-guidelines R.1 (#rr-raii): nested owners survive failures and
 * release in dependency order. I.7 (#ri-post): success requires all comparison
 * contracts. cpp-perf-guidelines GPU.1: separate verification readbacks from
 * compute claims. Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Diagnostic consultation, 2026-10-19: I.7 and GPU.1 above: failed numerical
 * postconditions identify the output, absolute rows, measured errors and fixed
 * limits using already-read host data; diagnostic work is outside GPU timing.
 * cpp-perf-guidelines GDSA.2: verify row-count-independent reductions by
 * comparing cached chunks with the same full trace, including capture bytes.
 * Source: guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 * Per-row design/implementation consultation, 2026-10-19: I.7 above makes
 * every row a postcondition; GPU.1 keeps comparisons outside GPU timing;
 * GDSA.2 keeps exact cache/full-trace equality distinct from oracle accuracy.
 * Prepared-buffer/device-selection contract consultation, 2026-10-20:
 * Core I.7 (#ri-post) and MEM.9
 * (guidelines/memory/MEM.9-embedded-allocate-at-init.md), both corpora queried
 * and full rules read for design/implementation. Exercise public results and
 * capacity/failure contracts; independent reference inputs remain unchanged.
 * Device-feedback verification, 2026-10-20: Core I.7 (#ri-post) checks
 * the committed prefix on EOS, cancellation and failed publication. GPU.7
 * (guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md) requires safe retirement
 * of queued output slots. Both corpora queried/read for design/implementation.
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

#include "bounds.h"
#include "gpu/cuda/device.h"
#include "inference/generation.h"
#include "inference/sampling/categorical.h"
#include "models/qwen3/checkpoint.h"
#include "tensors/formats/safetensors.h"
#include "tensors/representations/bf16.h"
#include "tensors/tensor.h"
#include "tests/gpu/capture_export.h"
#include "tests/gpu/forward_reference.h"
#include "tests/gpu/interpretation.h"
#include "tests/support/check.h"
#include "text/qwen3_chat.h"
#include "tokenizers/byte_bpe.h"

namespace {
namespace tensors = dolittle::tensors;
namespace models = dolittle::models;
namespace inference = dolittle::inference;
namespace gpu = dolittle::gpu;
namespace text = dolittle::text;
namespace tokenizers = dolittle::tokenizers;
namespace tests = dolittle::tests;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
tensors::ElementType dtype(const Json& value) {
  if (value == "float32") {
    return tensors::ElementType::float32;
  }
  if (value == "bfloat16") {
    return tensors::ElementType::bfloat16;
  }
  throw std::invalid_argument("unsupported reference dtype");
}
std::vector<models::TokenPosition> positions(const Json& values) {
  std::vector<models::TokenPosition> result;
  for (const auto& value : values) {
    result.push_back({value.get<std::size_t>()});
  }
  return result;
}
models::ObservationPoint point(const Json& value) {
  const auto boundary = value.at("boundary").get<std::string>();
  if (boundary != "block_input" && boundary != "block_output") {
    throw std::invalid_argument("unknown observation boundary");
  }
  return {boundary == "block_input" ? models::ObservationBoundary::block_input
                                    : models::ObservationBoundary::block_output,
          {value.at("block").get<std::size_t>()}};
}
tests::ExpectedRows rows(const Json& value) {
  return {positions(value.at("positions")),
          value.at("width").get<std::size_t>(),
          dtype(value.at("original_storage_type")),
          value.at("values").get<std::vector<float>>()};
}
tests::NumericalTolerance tolerance(const Json& value) {
  return {value.at("max_absolute_error").get<double>(),
          value.at("max_relative_l2").get<double>()};
}
bool equal_bytes(std::span<const std::byte> a, std::span<const std::byte> b) {
  return std::ranges::equal(a, b);
}
tests::ForwardReferenceCase reference_case(const Json& bundle,
                                           const Json& value,
                                           const std::string& identity,
                                           std::uint64_t id) {
  const auto& provenance = bundle.at("reference");
  tests::ForwardReferenceCase result;
  result.name = value.at("name");
  result.reference = {provenance.at("implementation_revision"),
                      provenance.at("environment_identity"),
                      provenance.at("attention_backend"),
                      provenance.at("numerical_policy")};
  result.checkpoint_identity = identity;
  result.execution_id = {id};
  result.token_ids = value.at("token_ids").get<std::vector<std::uint32_t>>();
  result.options = {{dtype(bundle.at("dtype")), tensors::ElementType::float32},
                    positions(value.at("logit_positions")),
                    {},
                    {64 * std::size_t{1024} * std::size_t{1024},
                     std::size_t{1024} * std::size_t{1024}},
                    std::size_t{23} * std::size_t{1024} * std::size_t{1024} *
                        std::size_t{1024}};
  result.logits = rows(value.at("logits"));
  for (const auto& capture : value.at("captures")) {
    result.captures.push_back({point(capture), rows(capture)});
    result.options.observations.push_back(
        {point(capture), positions(capture.at("positions"))});
  }
  return result;
}
struct ComparisonSources {
  std::reference_wrapper<const Json> bundle;
  std::reference_wrapper<const Json> value;
};
Json comparisons(const tests::ForwardReferenceCase& expected,
                 const inference::ForwardResult& actual,
                 ComparisonSources sources, bool& passed) {
  const auto& bundle = sources.bundle.get();
  const auto& value = sources.value.get();
  auto metrics = tests::compare_forward(
      expected, actual, tolerance(bundle.at("tolerances").at("logits")),
      tolerance(bundle.at("tolerances").at("observations")));
  Json outputs = Json::array();
  for (std::size_t i = 0; i < metrics.size(); ++i) {
    auto& metric = metrics.at(i);
    // Tighten the bundle fallback with independently fixed per-output scales.
    const Json* source = i == 0   ? nullptr
                         : i == 1 ? &value.at("logits")
                                  : &value.at("captures").at(i - 2);
    if (source && source->contains("tolerance")) {
      const auto limit = tolerance(source->at("tolerance"));
      metric.passed =
          metric.passed &&
          metric.maximum_absolute_error <= limit.max_absolute_error &&
          (!metric.relative_l2 || *metric.relative_l2 <= limit.max_relative_l2);
    }
    passed = passed && metric.passed;
    outputs.push_back(
        {{"output", metric.output_name},
         {"passed", metric.passed},
         {"maximum_absolute_error", metric.maximum_absolute_error},
         {"relative_l2",
          metric.relative_l2 ? Json(*metric.relative_l2) : Json(nullptr)},
         {"failure", metric.failure}});
  }
  return outputs;
}
// A semantic block input is exactly the preceding block output, where selected.
void check_boundaries(const inference::ForwardResult& result) {
  for (const auto& input : result.captures()) {
    if (input.site.point.boundary != models::ObservationBoundary::block_input ||
        !input.site.point.block.value) {
      continue;
    }
    for (const auto& output : result.captures()) {
      if (output.site.point.boundary !=
              models::ObservationBoundary::block_output ||
          output.site.point.block.value + 1 != input.site.point.block.value) {
        continue;
      }
      if (std::ranges::equal(input.positions, output.positions, {},
                             &models::TokenPosition::value,
                             &models::TokenPosition::value)) {
        tests::check(equal_bytes(input.bytes, output.bytes),
                     "adjacent block boundary values differ");
      }
    }
  }
}
void check_prefix(const inference::ForwardResult& shorter,
                  const inference::ForwardResult& longer) {
  if (shorter.token_ids().size() >= longer.token_ids().size() ||
      !std::ranges::equal(
          shorter.token_ids(),
          longer.token_ids().first(shorter.token_ids().size()))) {
    return;
  }
  for (const auto& a : shorter.captures()) {
    for (const auto& b : longer.captures()) {
      if (a.site.point.block.value != b.site.point.block.value ||
          a.site.point.boundary != b.site.point.boundary) {
        continue;
      }
      const auto row_bytes =
          a.site.vector_width * tensors::element_bytes(a.site.storage_type);
      for (std::size_t i = 0; i < a.positions.size(); ++i) {
        for (std::size_t j = 0; j < b.positions.size(); ++j) {
          if (dolittle::at(a.positions, i).value ==
              dolittle::at(b.positions, j).value) {
            tests::check(equal_bytes(a.bytes.subspan(i * row_bytes, row_bytes),
                                     b.bytes.subspan(j * row_bytes, row_bytes)),
                         "future tokens changed earlier capture");
          }
        }
      }
    }
  }
}
std::vector<float> widen(std::span<const std::byte> bytes,
                         tensors::ElementType type) {
  const auto width = tensors::element_bytes(type);
  tests::check(bytes.size() % width == 0, "readback element extent");
  std::vector<float> values(bytes.size() / width);
  if (type == tensors::ElementType::float32) {
    std::memcpy(values.data(), bytes.data(), bytes.size());
  } else {
    for (std::size_t i = 0; i < values.size(); ++i) {
      values.at(i) = tensors::to_float(tensors::read_le(
          std::span<const std::byte, 2>(bytes.subspan(2 * i).data(), 2)));
    }
  }
  return values;
}
Json compare_rows(const Json& reference, std::span<const std::byte> bytes,
                  tensors::ElementType type, std::size_t first,
                  std::size_t count, bool& passed) {
  const auto width = reference.at("width").get<std::size_t>();
  const auto all = reference.at("values").get<std::vector<float>>();
  const auto actual = widen(bytes, type);
  tests::check(
      first <= all.size() / width && count <= all.size() / width - first,
      "reference row bounds");
  const auto comparison = tests::compare_values(
      std::span(all).subspan(first * width, count * width), actual,
      tolerance(reference.at("tolerance")));
  Json row_metrics = Json::array();
  bool rows_passed = comparison.passed;
  for (std::size_t row = 0; row < count; ++row) {
    const auto metric = tests::compare_values(
        std::span(all).subspan((first + row) * width, width),
        std::span(actual).subspan(row * width, width),
        tolerance(reference.at("tolerance")));
    rows_passed = rows_passed && metric.passed;
    row_metrics.push_back(
        {{"position", first + row},
         {"passed", metric.passed},
         {"max_absolute_error", metric.maximum_absolute_error},
         {"relative_l2",
          metric.relative_l2 ? Json(*metric.relative_l2) : Json(nullptr)}});
  }
  if (!rows_passed) {
    const Json diagnostic{
        {"first", first},
        {"count", count},
        {"width", width},
        {"boundary", reference.value("boundary", "logits")},
        {"block", reference.value("block", Json(nullptr))},
        {"max_absolute_error", comparison.maximum_absolute_error},
        {"relative_l2", comparison.relative_l2 ? Json(*comparison.relative_l2)
                                               : Json(nullptr)},
        {"tolerance", reference.at("tolerance")},
        {"rows", row_metrics}};
    std::cerr << "numerical comparison failed: " << diagnostic.dump() << '\n';
  }
  passed = passed && rows_passed;
  return {{"passed", rows_passed},
          {"rows", std::move(row_metrics)},
          {"max_absolute_error", comparison.maximum_absolute_error},
          {"relative_l2", comparison.relative_l2 ? Json(*comparison.relative_l2)
                                                 : Json(nullptr)}};
}
std::string_view artifact_text(const tensors::formats::TensorArchive& archive,
                               std::string_view name) {
  auto bytes = archive.artifact(name);
  // Native API byte/address boundary; no unrelated typed object is
  // dereferenced. NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}
struct TextPair {
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
  std::unique_ptr<text::Protocol> protocol;
};
TextPair prepare_text(const tensors::formats::TensorArchive& archive,
                      const Json& generation) {
  if (generation.at("text").is_null()) {
    return {};
  }
  const auto source = archive.identity().repository + "/" +
                      archive.identity().revision + "/" +
                      archive.identity().artifact_digest;
  auto tokenizer = tokenizers::make_byte_bpe(
      {.tokenizer_json = artifact_text(archive, "tokenizer.json"),
       .tokenizer_config_json = artifact_text(archive, "tokenizer_config.json"),
       .verified_source_identity = source},
      {128 * std::size_t{1024} * std::size_t{1024},
       std::size_t{1024} * std::size_t{1024},
       std::size_t{1024} * std::size_t{1024},
       std::size_t{1024} * std::size_t{1024}});
  auto protocol = text::make_qwen3_chat(
      {.tokenizer_config_json = artifact_text(archive, "tokenizer_config.json"),
       .generation_config_json =
           artifact_text(archive, "generation_config.json"),
       .verified_source_identity = source},
      std::size_t{1024} * std::size_t{1024});
  for (const auto& value : generation.at("text").at("vectors")) {
    tests::check(tokenizer->encode(value.at("text").get<std::string>()) ==
                     value.at("ids").get<std::vector<std::uint32_t>>(),
                 "tokenizer IDs differ from pinned reference");
  }
  for (const auto& value : generation.at("text").at("prompts")) {
    std::optional<std::string> system;
    if (!value.at("system").is_null()) {
      system = value.at("system").get<std::string>();
    }
    const auto prepared = protocol->prepare(
        {system, value.at("user"),
         value.at("thinking").get<bool>() ? text::ThinkingMode::enabled
                                          : text::ThinkingMode::disabled},
        *tokenizer);
    tests::check(
        prepared.formatted_utf8 == value.at("formatted").get<std::string>() &&
            prepared.ids == value.at("ids").get<std::vector<std::uint32_t>>(),
        "formatted prompt differs from pinned reference");
  }
  std::cerr << "Verified tokenizer vectors: "
            << generation.at("text").at("vectors").size()
            << "; prompt vectors: "
            << generation.at("text").at("prompts").size() << '\n';
  return {std::move(tokenizer), std::move(protocol)};
}
Json check_generation(gpu::Device& device,
                      const models::BoundCheckpoint& checkpoint,
                      const Json& bundle, const TextPair& text_pair,
                      bool& passed) {
  const auto& fixture = bundle.at("generation");
  const auto& reference = fixture.at("reference");
  const auto tokens =
      reference.at("token_ids").get<std::vector<std::uint32_t>>();
  const auto prompt_count = fixture.at("prompt_tokens").get<std::size_t>();
  const tensors::ComputationMode mode{dtype(bundle.at("dtype")),
                                      tensors::ElementType::float32};
  inference::CachedForwardOptions options{
      mode,
      {500},
      tokens.size(),
      4,
      {},
      {64 * std::size_t{1024} * std::size_t{1024},
       std::size_t{1024} * std::size_t{1024}},
      std::size_t{23} * std::size_t{1024} * std::size_t{1024} *
          std::size_t{1024},
      256 * std::size_t{1024} * std::size_t{1024}};
  for (const auto& capture : reference.at("captures")) {
    options.capture_sites.push_back(point(capture));
  }
  Json report{{"chunks", Json::array()}};
  auto full_reference = reference_case(bundle, reference,
                                       std::string(checkpoint.identity()), 499);
  inference::ForwardResult full_result = [&] {
    auto plan = inference::prepare_full_forward(
        device, checkpoint, {tokens.size()}, full_reference.options);
    return plan.execute(tokens, {499});
  }();
  report.emplace(
      "full_trace",
      Json(comparisons(full_reference, full_result,
                       {.bundle = bundle, .value = reference}, passed)));
  report.emplace(
      "full_trace_logit_rows",
      Json(compare_rows(reference.at("logits"), full_result.logits().bytes,
                        mode.activation_storage, 0, tokens.size(), passed)));
  for (const auto& metric : report.at("full_trace")) {
    if (!metric.at("passed").get<bool>()) {
      std::cerr << "uncached trace comparison failed: " << metric.dump()
                << '\n';
    }
  }
  {
    auto impossible = options;
    impossible.max_device_bytes = 0;
    tests::expect_throws<std::length_error>([&] {
      (void)inference::prepare_cached_forward(device, checkpoint, impossible);
    });
    impossible = options;
    impossible.max_host_bytes = 0;
    tests::expect_throws<std::length_error>([&] {
      (void)inference::prepare_cached_forward(device, checkpoint, impossible);
    });
    const auto started = Clock::now();
    auto plan = inference::prepare_cached_forward(device, checkpoint, options);
    report.emplace("prepare_seconds", Json(seconds(started)));
    report.emplace("estimated_peak_device_bytes",
                   Json(plan.estimated_peak_device_bytes()));
    tests::expect_throws<std::invalid_argument>(
        [&] { (void)plan.consume({}, true); });
    const std::array bad{
        static_cast<std::uint32_t>(checkpoint.model().vocabulary_size())};
    tests::expect_throws<std::invalid_argument>(
        [&] { (void)plan.consume(bad, true); });
    tests::check(plan.prefix().usable && !plan.prefix().committed_positions,
                 "preflight rejection changed cache prefix");
    const auto execution = Clock::now();
    double prefill_seconds = 0, decode_seconds = 0;
    std::size_t logit_bytes = 0, capture_bytes = 0;
    for (std::size_t first = 0; first < tokens.size();) {
      // Uneven prefill chunks exercise active rows below the prepared bound;
      // subsequent single rows use nonzero absolute positions and cache reads.
      const auto count = first < prompt_count
                             ? std::min(std::size_t{3}, prompt_count - first)
                             : std::size_t{1};
      const auto consume_start = Clock::now();
      const auto chunk =
          plan.consume(std::span(tokens).subspan(first, count), true);
      const auto elapsed = seconds(consume_start);
      if (!chunk.next_logits) {
        throw std::runtime_error("requested chunk logits are missing");
      }
      (first < prompt_count ? prefill_seconds : decode_seconds) += elapsed;
      logit_bytes += chunk.next_logits->bytes.size();
      for (const auto& capture : chunk.captures) {
        capture_bytes += capture.bytes.size();
      }
      tests::check(chunk.first_position.value == first &&
                       chunk.tokens.size() == count &&
                       plan.prefix().committed_positions == first + count,
                   "completed chunk advances prefix exactly");
      Json metrics{
          {"first", first},
          {"count", count},
          {"logits",
           compare_rows(reference.at("logits"), chunk.next_logits->bytes,
                        mode.activation_storage, first + count - 1, 1, passed)},
          {"captures", Json::array()}};
      for (std::size_t i = 0; i < chunk.captures.size(); ++i) {
        const auto& capture = chunk.captures.at(i);
        for (std::size_t j = 0; j < count; ++j) {
          tests::check(capture.positions.at(j).value == first + j,
                       "absolute capture position");
        }
        metrics.at("captures")
            .push_back(compare_rows(reference.at("captures").at(i),
                                    capture.bytes, mode.activation_storage,
                                    first, count, passed));
      }
      const auto logit_row_bytes =
          checkpoint.model().vocabulary_size() *
          tensors::element_bytes(mode.activation_storage);
      tests::check(
          equal_bytes(
              chunk.next_logits->bytes,
              full_result.logits().bytes.subspan(
                  (first + count - 1) * logit_row_bytes, logit_row_bytes)),
          "cached logits differ from uncached execution of the same trace");
      for (std::size_t i = 0; i < chunk.captures.size(); ++i) {
        const auto& capture = chunk.captures.at(i);
        const auto row_bytes = capture.site.vector_width *
                               tensors::element_bytes(mode.activation_storage);
        tests::check(
            equal_bytes(capture.bytes, dolittle::at(full_result.captures(), i)
                                           .bytes.subspan(first * row_bytes,
                                                          count * row_bytes)),
            "cached captures differ from uncached execution of the same trace");
      }
      report.at("chunks").push_back(std::move(metrics));
      first += count;
    }
    report.emplace("replay_with_comparisons_seconds", Json(seconds(execution)));
    report.emplace("prefill_host_ready_seconds", Json(prefill_seconds));
    report.emplace("decode_host_ready_seconds", Json(decode_seconds));
    report.emplace("logit_readback_bytes", Json(logit_bytes));
    report.emplace("capture_readback_bytes", Json(capture_bytes));
    tests::expect_throws<std::invalid_argument>(
        [&] { (void)plan.consume(std::span(tokens).first(1), true); });
    plan.finish();
    tests::check(!plan.prefix().usable, "finish closes cache session");
  }
  text::PreparedPrompt prompt{
      "fixture/protocol",
      "fixture/tokenizer",
      "",
      {tokens.begin(),
       tokens.begin() + static_cast<std::ptrdiff_t>(prompt_count)},
      {static_cast<std::uint32_t>(checkpoint.model().vocabulary_size() - 1)},
      text::ThinkingMode::enabled};
  if (text_pair.protocol) {
    prompt = text_pair.protocol->prepare(
        {{}, "What is 2 + 2?", text::ThinkingMode::enabled},
        *text_pair.tokenizer);
    tests::check(prompt.ids == std::vector<std::uint32_t>(
                                   tokens.begin(),
                                   tokens.begin() + static_cast<std::ptrdiff_t>(
                                                        prompt_count)),
                 "generation prompt identity");
  }
  const inference::GenerationSettings settings{
      {inference::sampling::SelectionMode::greedy, 1, 0, 1},
      123,
      fixture.at("max_new_tokens")};
  inference::sampling::Categorical sampler(
      checkpoint.model().vocabulary_size());
  const inference::sampling::TokenSelector select =
      [&](auto logits, auto sampling, auto& rng) {
        return sampler.select(logits, sampling, rng);
      };
  inference::GenerationResult generated;
  {
    auto plan = inference::prepare_cached_forward(device, checkpoint, options);
    const auto start = Clock::now();
    generated = inference::generate(plan, prompt, settings, select, {500}, {},
                                    inference::sampling::Execution::device);
    report.emplace("generation_seconds", Json(seconds(start)));
    if (!generated.failure_message.empty() || generated.tokens != tokens) {
      std::cerr << Json{{"failure", generated.failure_message},
                        {"actual_ids", generated.tokens},
                        {"expected_ids", tokens},
                        {"consumed", generated.consumed_tokens},
                        {"reason", static_cast<unsigned>(generated.reason)}}
                       .dump()
                << '\n';
    }
    tests::check(generated.failure_message.empty() &&
                     generated.reason == inference::StopReason::token_limit &&
                     generated.tokens == tokens &&
                     generated.consumed_tokens == tokens.size(),
                 "greedy generation trace/finalization differs from reference");
    tests::check(!plan.prefix().usable, "generation finishes plan");
  }
  report.emplace(
      "generated_ids",
      Json(std::vector<std::uint32_t>(
          generated.tokens.begin() + static_cast<std::ptrdiff_t>(prompt_count),
          generated.tokens.end())));
  report.emplace("captures", Json(Json::array()));
  // Owned output is checked after the plan/cache has been destroyed.
  for (std::size_t i = 0; i < generated.captures.size(); ++i) {
    const auto& capture = generated.captures.at(i);
    tests::check(capture.positions.size() == tokens.size() &&
                     capture.positions.back().value == tokens.size() - 1,
                 "final emitted token capture missing");
    const auto exported = tests::export_capture(
        device, capture.device, capture.bytes,
        capture.positions.size() * capture.site.vector_width *
            tensors::element_bytes(capture.site.storage_type));
    report.at("captures")
        .push_back(compare_rows(reference.at("captures").at(i), exported,
                                mode.activation_storage, 0, tokens.size(),
                                passed));
  }
  {
    auto unobserved = options;
    unobserved.capture_sites.clear();
    unobserved.max_prefill_chunk = 2;
    auto plan =
        inference::prepare_cached_forward(device, checkpoint, unobserved);
    const auto result =
        inference::generate(plan, prompt, settings, select, {500}, {},
                            inference::sampling::Execution::device);
    tests::check(result.tokens == generated.tokens && result.captures.empty() &&
                     result.reason == generated.reason,
                 "capture selection or chunking changed generated IDs");
  }
  if (!text_pair.protocol) {
    // Termination/failure boundaries on the same small end-to-end generation
    // flow.
    for (int scenario = 0; scenario < 4; ++scenario) {
      auto plan =
          inference::prepare_cached_forward(device, checkpoint, options);
      auto bounded = settings;
      auto request = prompt;
      std::stop_source stop;
      if (scenario == 0) {
        bounded.max_new_tokens = 0;
      }
      if (scenario == 1) {
        request.stop_ids = {tokens.at(prompt_count)};
      }
      const inference::sampling::TokenSelector controlled =
          [&](auto logits, auto sampling, auto& rng) -> std::uint32_t {
        if (scenario == 3) {
          throw std::runtime_error("intentional selector failure");
        }
        if (scenario == 2) {
          stop.request_stop();
        }
        return sampler.select(logits, sampling, rng);
      };
      const auto result = inference::generate(
          plan, request, bounded, controlled, {500}, stop.get_token(),
          inference::sampling::Execution::host_reference);
      const std::array reasons{inference::StopReason::token_limit,
                               inference::StopReason::stop_token,
                               inference::StopReason::user_stop,
                               inference::StopReason::execution_failure};
      const auto consumed =
          prompt_count + ((scenario == 1 || scenario == 2) ? 1 : 0);
      tests::check(
          result.reason == reasons.at(static_cast<std::size_t>(scenario)) &&
              result.consumed_tokens == consumed &&
              result.tokens.size() == consumed,
          "stop/failure boundary consumed count");
      for (const auto& capture : result.captures) {
        tests::check(capture.positions.size() == consumed,
                     "stop/failure durable capture prefix");
      }
    }
  }
  if (!text_pair.protocol) {
    struct CancellingObserver final : inference::GenerationObserver {
      std::stop_source& stop;
      bool fail;
      explicit CancellingObserver(std::stop_source& source, bool failure)
          : stop(source), fail(failure) {}
      bool update(const inference::GenerationUpdate& update) noexcept override {
        if (update.first_position >= update.prompt_tokens &&
            !update.appended.empty()) {
          stop.request_stop();
          return !fail;
        }
        return true;
      }
    };
    for (int scenario = 0; scenario < 4; ++scenario) {
      auto plan =
          inference::prepare_cached_forward(device, checkpoint, options);
      auto bounded = settings;
      auto request = prompt;
      std::stop_source stop;
      if (scenario == 0) {
        bounded.max_new_tokens = 0;
      } else if (scenario == 1) {
        request.stop_ids = {tokens.at(prompt_count)};
      }
      CancellingObserver observer(stop, scenario == 3);
      const auto result =
          scenario < 2
              ? inference::generate(plan, request, bounded, select, {500},
                                    stop.get_token(),
                                    inference::sampling::Execution::device)
              : inference::generate(plan, request, bounded, select, {500},
                                    observer, stop.get_token(),
                                    inference::sampling::Execution::device);
      const auto consumed = result.consumed_tokens;
      const bool bounded_stop =
          scenario < 2
              ? consumed == prompt_count + (scenario == 0 ? 0 : 1)
              : consumed >= prompt_count + 1 && consumed <= tokens.size();
      const bool reason_ok =
          scenario == 0 ? result.reason == inference::StopReason::token_limit
          : scenario == 1
              ? result.reason == inference::StopReason::stop_token
              : result.reason == inference::StopReason::user_stop ||
                    result.reason == inference::StopReason::token_limit;
      tests::check(
          result.failure_message.empty() && reason_ok && bounded_stop &&
              result.tokens.size() == consumed,
          "device stop/cancel lost a selected token or exceeded budget");
      for (const auto& capture : result.captures) {
        tests::check(capture.positions.size() == consumed,
                     "device feedback discarded capture leaked into results");
      }
    }
  }
  report.emplace("tokenizer_vectors",
                 Json(fixture.at("text").is_null()
                          ? 0
                          : fixture.at("text").at("vectors").size()));
  report.emplace("prompt_vectors",
                 Json(fixture.at("text").is_null()
                          ? 0
                          : fixture.at("text").at("prompts").size()));
  return report;
}

Json run(const std::filesystem::path& fixture) {
  tests::check(std::filesystem::file_size(fixture) <=
                   128 * std::size_t{1024} * std::size_t{1024},
               "fixture exceeds harness budget");
  std::ifstream stream(fixture);
  const auto bundle = Json::parse(stream);
  tests::check(bundle.at("schema_version") == 1, "unsupported fixture schema");
  tensors::formats::safetensors::OpenRequest request{
      fixture.parent_path(),
      bundle.at("repository"),
      bundle.at("revision"),
      {},
      bundle.at("shards").get<std::vector<std::string>>(),
      std::nullopt,
      16 * std::size_t{1024} * std::size_t{1024},
      128 * std::size_t{1024} * std::size_t{1024},
      100000};
  if (!bundle.at("index").is_null()) {
    request.index = bundle.at("index").get<std::string>();
  }
  for (const auto& artifact : bundle.at("artifacts")) {
    request.artifacts.push_back({artifact.at("relative_path"),
                                 artifact.at("byte_size"),
                                 artifact.at("sha256")});
  }
  const auto load_start = Clock::now();
  auto archive = tensors::formats::safetensors::open(request);
  auto text_pair = prepare_text(*archive, bundle.at("generation"));
  auto checkpoint = models::qwen3::bind_target(
      std::move(archive), bundle.at("config_artifact").get<std::string>());
  Json report{{"load_seconds", seconds(load_start)},
              {"checkpoint_identity", checkpoint->identity()},
              {"cases", Json::array()},
              {"reference", bundle.at("reference")},
              {"tolerances", bundle.at("tolerances")}};
  auto device = gpu::cuda::open_device(0);
  report.emplace("device", Json({{"name", device->info().name},
                                 {"architecture", device->info().architecture},
                                 {"driver", device->info().driver_version}}));
  std::vector<tests::ForwardReferenceCase> expected;
  std::vector<inference::ForwardResult> results;
  std::uint64_t id = 1;
  for (const auto& value : bundle.at("cases")) {
    auto reference = reference_case(bundle, value,
                                    std::string(checkpoint->identity()), id++);
    const models::SequenceLength length{reference.token_ids.size()};
    auto invalid = reference.options;
    invalid.max_device_bytes = 0;
    tests::expect_throws<std::length_error>([&] {
      (void)inference::prepare_full_forward(*device, *checkpoint, length,
                                            invalid);
    });
    Json measurement{{"name", reference.name}};
    {
      const auto prepare_start = Clock::now();
      auto plan = inference::prepare_full_forward(*device, *checkpoint, length,
                                                  reference.options);
      measurement.emplace("prepare_seconds", Json(seconds(prepare_start)));
      measurement.emplace("estimated_peak_device_bytes",
                          Json(plan.estimated_peak_device_bytes()));
      measurement.emplace("host_output_bytes", Json(plan.host_output_bytes()));
      auto bad_tokens = reference.token_ids;
      bad_tokens.at(0) =
          static_cast<std::uint32_t>(checkpoint->model().vocabulary_size());
      tests::expect_throws<std::invalid_argument>(
          [&] { (void)plan.execute(bad_tokens, reference.execution_id); });
      const auto execute_start = Clock::now();
      results.push_back(
          plan.execute(reference.token_ids, reference.execution_id));
      measurement.emplace("execute_host_ready_seconds",
                          Json(seconds(execute_start)));
      // Reuse with different contents, then reproduce the original pass.
      auto changed = reference.token_ids;
      changed.at(0) =
          (changed.at(0) + 1) % checkpoint->model().vocabulary_size();
      (void)plan.execute(changed, {10000 + reference.execution_id.value});
      auto repeated = plan.execute(reference.token_ids, reference.execution_id);
      tests::check(
          equal_bytes(repeated.logits().bytes, results.back().logits().bytes),
          "serial plan reuse changed logits");
      for (std::size_t i = 0; i < repeated.captures().size(); ++i) {
        tests::check(
            equal_bytes(dolittle::at(repeated.captures(), i).bytes,
                        dolittle::at(results.back().captures(), i).bytes),
            "serial reuse changed captures");
      }
    }
    {
      auto no_capture = reference.options;
      no_capture.observations.clear();
      auto plan = inference::prepare_full_forward(*device, *checkpoint, length,
                                                  no_capture);
      auto result = plan.execute(reference.token_ids, reference.execution_id);
      tests::check(
          result.captures().empty() &&
              equal_bytes(result.logits().bytes, results.back().logits().bytes),
          "observation selection changed logits");
    }
    expected.push_back(std::move(reference));
    report.at("cases").push_back(std::move(measurement));
  }
  tests::check(!results.empty(), "fixture contains no cases");
  {
    auto empty = expected.front().options;
    empty.logit_positions = std::vector<models::TokenPosition>{};
    empty.observations.clear();
    const auto site =
        checkpoint->model().observation_sites(empty.numerical_mode).front();
    empty.observations.push_back(
        {site.point, std::vector<models::TokenPosition>{}});
    auto plan = inference::prepare_full_forward(
        *device, *checkpoint, {expected.front().token_ids.size()}, empty);
    auto result = plan.execute(expected.front().token_ids, {99999});
    tests::check(result.logits().bytes.empty() &&
                     result.captures().size() == 1 &&
                     dolittle::at(result.captures(), 0).bytes.empty(),
                 "empty selections did not preserve empty metadata");
  }
  bool passed = true;
  report.emplace(
      "generation",
      Json(check_generation(*device, *checkpoint, bundle, text_pair, passed)));
  // All comparisons below deliberately run after the execution owners die.
  checkpoint.reset();
  auto interpretation = tests::check_interpretation(*device, fixture);
  passed = passed && interpretation.at("passed").get<bool>();
  report.emplace("interpretation", std::move(interpretation));
  device.reset();
  for (std::size_t i = 0; i < results.size(); ++i) {
    report.at("cases").at(i).emplace(
        "outputs",
        Json(comparisons(expected.at(i), results.at(i),
                         {.bundle = bundle, .value = bundle.at("cases").at(i)},
                         passed)));
    check_boundaries(results.at(i));
    for (const auto& other : results) {
      check_prefix(results.at(i), other);
    }
  }
  report.emplace("passed", Json(passed));
  return report;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::invalid_argument(
          "usage: test_generation FIXTURE.json [FIXTURE.json ...]");
    }
    Json reports = Json::array();
    bool passed = true;
    for (int i = 1; i < argc; ++i) {
      auto report =
          run(dolittle::at(std::span(argv, static_cast<std::size_t>(argc)),
                           static_cast<std::size_t>(i)));
      passed = passed && report.at("passed").get<bool>();
      reports.push_back(std::move(report));
    }
    std::cout << reports.dump(2) << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "generation integration failed: " << error.what() << '\n';
    return 1;
  }
}
