/*
 * Native application and numerical acceptance design, 2026-10-21:
 * Core I.1 (#ri-explicit) makes backend selection and comparison policy
 * explicit. Performance GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md)
 * keeps production inference on the GPU; test metrics consume completed
 * readbacks. Both local corpora queried and read for design and implementation.
 * Steering verification extends the same target/AV experiment: independently
 * frozen AR vectors and difference direction, fresh replay at prompt/generated
 * positions, zero-strength control and changed-row comparisons. AR chunk sizes
 * 1 and 7 must yield identical vectors. Reject zero difference before target
 * loading. Diagnostics download completed rows only; live math stays on GPU.
 * Steering corpus consultation, header design and implementation 2026-10-20:
 * Core I.1 (#ri-explicit), typed contracts; performance MEM.9, bounded setup
 * before execution (guidelines/memory/MEM.9-embedded-allocate-at-init.md).
 * Both local corpora queried and full rules read.
 *
 * Verification axis D3: AV continuation of the existing tiny/8B experiment
 * scenarios. Input is a frozen target fixture; the sibling AV fixture supplies
 * independently frozen injection/logit/capture expectations. Returns owned
 * metrics. No sixth scenario is registered. The implementation owns temporary
 * checkpoints and releases each plan before switching model roles; caller's
 * device outlives it. Algorithm: docs/architecture/logical-overview.md,
 * sections 2 through 4:
 * 1. Read pinned fixtures and verify model artifacts. Normalize the independent
 *    target row; compare FP32 values and exact stored replacement bytes.
 * 2. Generate through Run, release target execution and retain the selected
 * row. Compare it with the unchanged target limits, then snapshot its owned
 * bytes.
 * 3. Feed identical reference injection bytes to AV full and cached execution.
 *    Compare selected activations and logits against both fixed error bounds
 * for every row. Require exact final cached/full logits on the same consumed
 * IDs.
 * 4. Interpret the retained engine row twice. Require expected greedy IDs and
 *    fresh sequence state with retained AV weights, then check pre-request
 *    cancellation, rejected calls and capture lifetime after model close.
 * Cost: bounded fixture passes; comparison work and reference copies stay
 * outside inference timing. Provider callbacks borrow immutable requests that
 * outlive Run. Optimization: compare completed host bytes without additional
 * diagnostic waits. Corpus implementation consultation, 2026-10-19:
 * cpp-guidelines R.1 (#rr-raii), I.7 (#ri-post) enforce ownership and numerical
 * postconditions. Performance GPU.1 separates fixture/readback work from
 * compute; source guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Prepared-buffer/device-selection contract consultation, 2026-10-20:
 * Core I.7 (#ri-post) and MEM.9
 * (guidelines/memory/MEM.9-embedded-allocate-at-init.md), both corpora queried
 * and full rules read for design/implementation. Exercise public results and
 * capacity/failure contracts; independent reference inputs remain unchanged.
 * Upload pipeline design/implementation consultation, 2026-10-20: both
 * corpora queried in both phases. Core R.1 (#rr-raii) and I.7 (#ri-post)
 * require completed resident weights and owned staging lifetimes; performance
 * GPU.7 (guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md) requires safe slot
 * reuse. The tiny cached AV pass uses an odd, bounded staging allowance so
 * weights cross upload boundaries; the unchanged numerical oracle checks
 * values after slot reuse. This extends the existing experiment scenario.
 */
#include "tests/gpu/interpretation.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>

#include "bounds.h"
#include "experiment/run.h"
#include "inference/injection/device.h"
#include "inference/sampling/categorical.h"
#include "inference/steering/device.h"
#include "models/qwen3/checkpoint.h"
#include "nla/pairs/qwen3_8b.h"
#include "tensors/formats/safetensors.h"
#include "tensors/representations/bf16.h"
#include "tensors/tensor.h"
#include "tests/gpu/capture_export.h"
#include "tests/support/check.h"
#include "tests/support/numerical.h"
#include "text/qwen3_chat.h"
#include "tokenizers/byte_bpe.h"

namespace dolittle::tests {
namespace {
using Json = nlohmann::json;
constexpr std::size_t mib = std::size_t{1024} * 1024, gib = 1024 * mib;
Json read(const std::filesystem::path& p) {
  std::ifstream f(p);
  if (!f) {
    throw std::runtime_error("missing interpretation fixture: " + p.string());
  }
  return Json::parse(f);
}
tensors::ElementType type(const Json& j) {
  return j == "float32" ? tensors::ElementType::float32
                        : tensors::ElementType::bfloat16;
}
std::string source_text(const tensors::formats::SourceIdentity& s) {
  return s.repository + "/" + s.revision + "/" + s.artifact_digest;
}
tensors::formats::safetensors::OpenRequest request(
    const Json& j, const std::filesystem::path& root) {
  tensors::formats::safetensors::OpenRequest r{
      root,
      j.at("repository"),
      j.at("revision"),
      {},
      j.at("shards").get<std::vector<std::string>>(),
      {},
      16 * mib,
      128 * mib,
      100000};
  if (!j.at("index").is_null()) {
    r.index = j.at("index").get<std::string>();
  }
  for (const auto& f : j.at("artifacts")) {
    r.artifacts.push_back(
        {f.at("relative_path"), f.at("byte_size"), f.at("sha256")});
  }
  return r;
}
std::string artifact(const tensors::formats::TensorArchive& a,
                     std::string_view name) {
  const auto bytes = a.artifact(name);
  std::string s(bytes.size(), '\0');
  std::memcpy(s.data(), bytes.data(), bytes.size());
  return s;
}
struct Text {
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
  std::unique_ptr<text::Protocol> protocol;
};
Text text_helpers(const tensors::formats::TensorArchive& archive) {
  const auto tokenizer = artifact(archive, "tokenizer.json"),
             config = artifact(archive, "tokenizer_config.json"),
             generation = artifact(archive, "generation_config.json"),
             identity = source_text(archive.identity());
  return {tokenizers::make_byte_bpe({tokenizer, config, identity},
                                    {128 * mib, mib, mib, mib}),
          text::make_qwen3_chat({config, generation, identity}, mib)};
}
struct FixtureTokenizer final : tokenizers::Tokenizer {
  std::string source;
  explicit FixtureTokenizer(std::string s) : source(std::move(s)) {}
  std::string_view identity() const noexcept override { return source; }
  std::vector<std::uint32_t> encode(std::string_view) const override {
    throw std::logic_error("fixture protocol supplies IDs");
  }
  tokenizers::DecodedBytes decode(
      std::span<const std::uint32_t> ids) const override {
    tokenizers::DecodedBytes out;
    out.tokens.reserve(ids.size());
    out.bytes.reserve(ids.size());
    decode_into(ids, out);
    return out;
  }
  std::size_t max_token_bytes() const noexcept override { return 1; }
  void decode_into(std::span<const std::uint32_t> ids,
                   tokenizers::DecodedBytes& out) const override {
    if (ids.size() > out.tokens.capacity() ||
        ids.size() > out.bytes.capacity()) {
      throw std::length_error("fixture decode capacity");
    }
    out.bytes.clear();
    out.tokens.clear();
    for (auto id : ids) {
      const auto begin = out.bytes.size();
      out.bytes += 'x';
      out.tokens.push_back({id, begin, out.bytes.size(), true});
    }
  }
  bool contains(std::uint32_t) const noexcept override { return true; }
};
struct FixtureProtocol final : text::Protocol {
  text::PreparedPrompt prompt;
  text::PreparedPrompt prepare(const text::Prompt&,
                               const tokenizers::Tokenizer&) const override {
    return prompt;
  }
  std::vector<text::TokenSegment> segment(std::span<const std::uint32_t> ids,
                                          std::size_t start,
                                          text::ThinkingMode) const override {
    std::vector<text::TokenSegment> result;
    result.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      result.push_back({start + i, text::SegmentKind::unclassified});
    }
    return result;
  }
};
struct FixturePair final : nla::Pair {
  nla::PairIdentity provenance;
  models::ObservationSite site{};
  text::PreparedPrompt prompt;
  models::TokenPosition injection_position{};
  float scale{};
  const nla::PairIdentity& identity() const noexcept override {
    return provenance;
  }
  models::ObservationSite target_site() const noexcept override { return site; }
  const text::PreparedPrompt& verbalizer_prompt() const noexcept override {
    return prompt;
  }
  void validate_model(nla::ModelRole role,
                      const tensors::formats::SourceIdentity& s,
                      const models::BoundCheckpoint& c,
                      tensors::ComputationMode mode) const override {
    const auto& expected = role == nla::ModelRole::target
                               ? provenance.target
                               : provenance.verbalizer;
    check(s.artifact_digest == expected.artifact_digest &&
              c.source_identity().artifact_digest == s.artifact_digest &&
              mode.activation_storage == site.storage_type,
          "tiny pairing identity");
  }
  nla::Injection prepare_injection(nla::CaptureRow row) const override {
    if (row.device) {
      return {{injection_position,
               row.site.vector_width,
               site.storage_type,
               {},
               inference::DeviceNormalization{*row.device,
                                              row.site.storage_type, scale}},
              0,
              scale};
    }
    return nla::normalize_activation(row, injection_position, scale,
                                     site.storage_type);
  }
  nla::ExplanationRange explanation(std::string_view raw) const override {
    return nla::explanation_tags(raw);
  }
};
std::vector<std::byte> encode(std::span<const float> values,
                              tensors::ElementType storage) {
  std::vector<std::byte> bytes(values.size() * tensors::element_bytes(storage));
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (storage == tensors::ElementType::float32) {
      std::memcpy(std::span(bytes).subspan(i * sizeof(float)).data(),
                  &dolittle::at(values, i), sizeof(float));
    } else {
      tensors::write_le(
          tensors::from_float(dolittle::at(values, i)),
          std::span<std::byte, 2>(std::span(bytes).subspan(i * 2).data(), 2));
    }
  }
  return bytes;
}
std::vector<float> decode(std::span<const std::byte> bytes,
                          tensors::ElementType storage) {
  std::vector<float> values(bytes.size() / tensors::element_bytes(storage));
  if (storage == tensors::ElementType::float32) {
    std::memcpy(values.data(), bytes.data(), bytes.size());
  } else {
    for (std::size_t i = 0; i < values.size(); ++i) {
      values.at(i) = tensors::to_float(tensors::read_le(
          std::span<const std::byte, 2>(bytes.subspan(i * 2).data(), 2)));
    }
  }
  return values;
}
Json compare(const Json& expected, std::span<const std::byte> bytes,
             tensors::ElementType storage, ModelComparisonPolicy policy) {
  const auto values = expected.at("values").get<std::vector<float>>();
  const auto actual = decode(bytes, storage);
  const auto& limit = expected.at("tolerance");
  const auto m = compare_values(
      values, actual,
      {limit.at("max_absolute_error"), limit.at("max_relative_l2")});
  bool passed = m.passed;
  bool accepted = accepts_model_comparison(m, policy);
  Json rows = Json::array();
  const auto width = expected.at("width").get<std::size_t>();
  for (std::size_t row = 0; row < expected.at("positions").size(); ++row) {
    const auto metric = compare_values(
        std::span(values).subspan(row * width, width),
        std::span(actual).subspan(row * width, width),
        {limit.at("max_absolute_error"), limit.at("max_relative_l2")});
    accepted = accepted && accepts_model_comparison(metric, policy);
    passed = passed && metric.passed;
    rows.push_back(
        {{"position", expected.at("positions").at(row)},
         {"passed", metric.passed},
         {"max_absolute", metric.maximum_absolute_error},
         {"relative_l2",
          metric.relative_l2 ? Json(*metric.relative_l2) : Json(nullptr)}});
  }
  return {
      {"passed", passed},
      {"accepted", accepted},
      {"kind", expected.contains("block") ? "capture" : "logits"},
      {"block", expected.value("block", Json(nullptr))},
      {"boundary", expected.value("boundary", Json(nullptr))},
      {"rows", rows},
      {"max_absolute", m.maximum_absolute_error},
      {"relative_l2", m.relative_l2 ? Json(*m.relative_l2) : Json(nullptr)}};
}
// Requests are immutable locals and outlive Run and every provider call.
experiment::ModelProvider provider(
    const tensors::formats::safetensors::OpenRequest& r) {
  return [&r](std::size_t allowance) {
    if (allowance < r.max_metadata_bytes) {
      throw std::length_error("loader metadata allowance");
    }
    auto archive = tensors::formats::safetensors::open(r);
    auto source = archive->identity();
    return experiment::LoadedModel{
        std::move(source),
        models::qwen3::bind_causal_lm(std::move(archive), "config.json"),
        r.max_metadata_bytes};
  };
}
}  // namespace
Json check_interpretation(gpu::Device& device,
                          const std::filesystem::path& target_fixture,
                          ModelComparisonPolicy policy) {
  const auto target = read(target_fixture);
  const auto av_path =
      target_fixture.parent_path().parent_path().parent_path() / "av-fixtures" /
      (target_fixture.parent_path().filename().string() + ".json");
  const auto av = read(av_path);
  const bool actual = av.at("repository") == "marco-bazzani/Qwen3-8B-nla";
  const auto storage = type(av.at("dtype"));
  const tensors::ComputationMode mode{storage, tensors::ElementType::float32};
  auto target_request = request(target, target_fixture.parent_path());
  auto av_request = request(av, av.at("root").get<std::string>());
  if (actual) {
    const auto manifest = read("scripts/validation/models-8b.json");
    const auto& entry = manifest.at("models").at(0);
    target_request.artifacts.clear();
    for (const auto& f : entry.at("files")) {
      target_request.artifacts.push_back(
          {f.at("path"), f.at("bytes"), f.at("sha256")});
    }
  }
  auto target_archive = tensors::formats::safetensors::open(target_request);
  auto av_archive = tensors::formats::safetensors::open(av_request);
  const auto target_source = target_archive->identity(),
             av_source = av_archive->identity();
  Text target_text, av_text;
  std::unique_ptr<nla::Pair> pair;
  const auto target_tokens = target.at("generation")
                                 .at("reference")
                                 .at("token_ids")
                                 .get<std::vector<std::uint32_t>>();
  const auto target_prompt_count =
      target.at("generation").at("prompt_tokens").get<std::size_t>();
  if (actual) {
    target_text = text_helpers(*target_archive);
    av_text = text_helpers(*av_archive);
    pair = nla::pairs::make_qwen3_8b(
        {target_source, av_source, av_archive->artifact("nla_meta.yaml"),
         *av_text.tokenizer, *av_text.protocol},
        {mib, mib, 4096});
  } else {
    target_text.tokenizer = std::make_unique<FixtureTokenizer>(
        source_text(target_source) + "/fixture-tokenizer");
    av_text.tokenizer = std::make_unique<FixtureTokenizer>(
        source_text(av_source) + "/fixture-tokenizer");
    auto protocol = std::make_unique<FixtureProtocol>();
    protocol->prompt = {"fixture-protocol",
                        std::string(target_text.tokenizer->identity()),
                        "fixture",
                        std::vector<std::uint32_t>(
                            target_tokens.begin(),
                            target_tokens.begin() + static_cast<std::ptrdiff_t>(
                                                        target_prompt_count)),
                        {36},
                        text::ThinkingMode::enabled};
    target_text.protocol = std::move(protocol);
    auto fixture_pair = std::make_unique<FixturePair>();
    fixture_pair->provenance = {"fixture-l2", target_source, av_source,
                                "fixture"};
    fixture_pair->site = {
        {models::ObservationBoundary::block_output, {av.at("capture_block")}},
        av.at("capture_values").size(),
        storage};
    fixture_pair->prompt = {
        "fixture-av",
        std::string(av_text.tokenizer->identity()),
        "fixture",
        av.at("prompt_ids").get<std::vector<std::uint32_t>>(),
        av.at("stop_ids").get<std::vector<std::uint32_t>>(),
        text::ThinkingMode::enabled};
    fixture_pair->injection_position = {av.at("injection_position")};
    fixture_pair->scale = av.at("scale");
    pair = std::move(fixture_pair);
  }
  check(pair->verbalizer_prompt().ids ==
            av.at("prompt_ids").get<std::vector<std::uint32_t>>(),
        "trained AV prompt IDs");
  if (actual) {
    check(pair->verbalizer_prompt().formatted_utf8 ==
              av.at("formatted_prompt").get<std::string>(),
          "trained AV prompt bytes");
  }
  const auto site = pair->target_site();
  const auto capture_values = av.at("capture_values").get<std::vector<float>>();
  const auto capture_bytes = encode(capture_values, storage);
  const nla::CaptureRow reference_row{
      site, {av.at("capture_position")}, capture_bytes};
  const auto injection = pair->prepare_injection(reference_row);
  const auto fp32 =
      nla::normalize_activation(reference_row, injection.embedding.position,
                                injection.scale, tensors::ElementType::float32);
  const auto& normalization_limit = av.at("normalization_tolerance");
  const auto normalization = compare_values(
      av.at("normalized_fp32").get<std::vector<float>>(),
      decode(fp32.embedding.bytes, tensors::ElementType::float32),
      {normalization_limit.at("max_absolute_error"),
       normalization_limit.at("max_relative_l2")});
  check(normalization.passed,
        "normalized FP32 vector disagrees with frozen reference");
  check(
      injection.embedding.bytes ==
          encode(av.at("injection_values").get<std::vector<float>>(), storage),
      "injected stored bytes disagree with frozen reference");
  // Explicit diagnostic readback checks the GPU conversion on the same frozen
  // source as the CPU helper. This is outside live target-to-AV execution.
  {
    auto queue = device.create_queue();
    auto source = device.allocate({capture_bytes.size(), 4});
    auto staging = device.allocate_staging({capture_bytes.size(), 4});
    std::memcpy(staging.bytes().data(), capture_bytes.data(),
                capture_bytes.size());
    queue.upload({&source, 0, capture_bytes.size()},
                 {&staging, 0, capture_bytes.size()});
    for (const auto output_type : {storage, tensors::ElementType::float32}) {
      inference::DeviceInjection normalized(
          device, queue, {&source, 0, capture_bytes.size()}, storage,
          output_type, site.vector_width, injection.scale);
      auto output = device.allocate_staging({normalized.output().bytes, 4});
      auto metadata = device.allocate_staging({2 * sizeof(float), 4});
      normalized.submit();
      queue.download({&output, 0, output.bytes().size()}, normalized.output());
      queue.download({&metadata, 0, sizeof(float)}, normalized.norm());
      queue.download({&metadata, sizeof(float), sizeof(unsigned)},
                     normalized.status());
      queue.drain();
      float norm{};
      unsigned status{};
      std::memcpy(&norm, metadata.bytes().data(), sizeof(norm));
      std::memcpy(&status, metadata.bytes().subspan(sizeof(norm)).data(),
                  sizeof(status));
      const auto& expected = output_type == storage ? injection : fp32;
      check(!status && norm == expected.source_l2_norm &&
                std::ranges::equal(output.bytes(), expected.embedding.bytes),
            "GPU NLA normalization differs on the frozen source row");
    }
  }
  target_archive.reset();
  av_archive.reset();
  auto sampler =
      std::make_shared<inference::sampling::Categorical>(actual ? 151936 : 37);
  experiment::RunInputs inputs{
      std::move(pair),
      provider(target_request),
      provider(av_request),
      mode,
      mode,
      std::move(target_text.tokenizer),
      std::move(av_text.tokenizer),
      std::move(target_text.protocol),
      [sampler](auto logits, auto settings, auto& rng) {
        return sampler->select(logits, settings, rng);
      },
      device.info().backend,
      256 * mib,
      inference::sampling::Execution::device};
  experiment::Run run(device, {1}, std::move(inputs),
                      {23 * gib, gib, 16, 4096, 4096, mib, {256 * mib, 4}});
  const inference::GenerationSettings target_settings{
      {inference::sampling::SelectionMode::greedy, 1, 0, 1},
      123,
      target.at("generation").at("max_new_tokens")};
  run.generate({{}, "What is 2 + 2?", text::ThinkingMode::enabled},
               target_settings);
  check(run.phase() == experiment::Phase::retained && run.record(),
        "target generation and result retained");
  check(run.record()->target().generation.tokens == target_tokens,
        "target output changed in experiment orchestration");
  const experiment::CaptureKey key{site.point, {av.at("capture_position")}};
  const auto retained = run.record()->capture(key);
  if (!retained) {
    throw std::runtime_error("target capture was not retained");
  }
  const auto retained_snapshot =
      export_capture(device, retained->device, retained->bytes,
                     site.vector_width * tensors::element_bytes(storage));
  const auto& target_captures =
      target.at("generation").at("reference").at("captures");
  const auto target_capture = std::find_if(
      target_captures.begin(), target_captures.end(), [&](const auto& c) {
        return c.at("block") == av.at("capture_block") &&
               c.at("boundary") == "block_output";
      });
  check(target_capture != target_captures.end(), "target oracle capture site");
  const auto& target_limit = target_capture->at("tolerance");
  const auto target_metric =
      compare_values(capture_values, decode(retained_snapshot, storage),
                     {target_limit.at("max_absolute_error"),
                      target_limit.at("max_relative_l2")});
  // The target keeps its original frozen numerical limits; byte identity here
  // checks retention against the engine's own completed snapshot, not a new
  // zero-error requirement for the preexisting FP32 reference comparison.
  Json numerical = Json::array();
  bool passed = accepts_model_comparison(target_metric, policy);
  std::vector<std::byte> full_av_logits;
  // The independent AV comparison owns another model outside Run's scheduler.
  // Release only idle weights before lending the device to that comparison;
  // capture leases and the retained experiment remain valid.
  const auto target_weight_bytes = run.retained_model_bytes().device_bytes;
  check(target_weight_bytes > 0, "completed target weights retained");
  if (actual) {
    run.release_models();
    check(run.retained_model_bytes().device_bytes == 0,
          "explicit idle release frees weights");
  }
  // Inspect AV math on identical frozen injection bytes before variants. All
  // these device owners die before Run loads its persistent verbalizer.
  {
    auto loaded = provider(av_request)(256 * mib);
    const auto& reference = av.at("reference");
    const auto ids =
        reference.at("token_ids").get<std::vector<std::uint32_t>>();
    inference::ForwardOptions options{
        mode,
        std::vector<models::TokenPosition>{{ids.size() - 1}},
        {},
        {64 * mib, mib},
        23 * gib};
    for (const auto& c : reference.at("captures")) {
      std::vector<models::TokenPosition> positions;
      for (const auto& p : c.at("positions")) {
        positions.push_back({p});
      }
      options.observations.push_back(
          {{c.at("boundary") == "block_input"
                ? models::ObservationBoundary::block_input
                : models::ObservationBoundary::block_output,
            {c.at("block")}},
           positions});
    }
    auto full = inference::prepare_full_forward(device, *loaded.checkpoint,
                                                {ids.size()}, options);
    const auto result = full.execute(ids, {1000}, injection.embedding);
    full_av_logits.assign(result.logits().bytes.begin(),
                          result.logits().bytes.end());
    auto metric =
        compare(reference.at("logits"), result.logits().bytes, storage, policy);
    passed = passed && metric.at("accepted").get<bool>();
    numerical.push_back(metric);
    for (std::size_t i = 0; i < result.captures().size(); ++i) {
      metric =
          compare(reference.at("captures").at(i),
                  dolittle::at(result.captures(), i).bytes, storage, policy);
      passed = passed && metric.at("accepted").get<bool>();
      numerical.push_back(metric);
    }
  }
  {
    auto loaded = provider(av_request)(256 * mib);
    const std::size_t staging_allowance = actual ? 16 * mib : 257;
    auto resident =
        inference::load_resident_model(device, std::move(loaded.checkpoint),
                                       {mode, 23 * gib, staging_allowance});
    const auto prompt = av.at("prompt_ids").get<std::vector<std::uint32_t>>();
    const auto& steps = av.at("steps");
    inference::CachedForwardOptions options{
        mode,
        {1001},
        prompt.size() + av.at("max_new_tokens").get<std::size_t>(),
        2,
        {},
        {0, 0},
        23 * gib,
        512 * mib};
    {
      if (!retained->device.has_value()) {
        throw std::runtime_error("live target capture must be device-owned");
      }
      inference::EmbeddingOverride device_override{
          injection.embedding.position,
          site.vector_width,
          storage,
          {},
          inference::DeviceNormalization{retained->device.value(), storage,
                                         injection.scale}};
      auto diagnostic = inference::prepare_cached_forward(
          *resident, options, std::move(device_override));
      expect_throws<std::invalid_argument>(
          [&] { (void)diagnostic.consume(std::span(prompt).first(1), true); });
    }
    auto cached = inference::prepare_cached_forward(*resident, options,
                                                    injection.embedding);
    for (std::size_t pos = 0; pos < prompt.size();) {
      const auto n = std::min(options.max_prefill_chunk, prompt.size() - pos);
      const auto result = cached.consume(std::span(prompt).subspan(pos, n),
                                         pos + n == prompt.size());
      pos += n;
      if (result.next_logits) {
        auto m = compare(steps.at(0).at("logits"), result.next_logits->bytes,
                         storage, policy);
        passed = passed && m.at("accepted").get<bool>();
        numerical.push_back(m);
      }
    }
    const auto emitted =
        av.at("expected_generated_ids").get<std::vector<std::uint32_t>>();
    for (std::size_t i = 0; i < emitted.size(); ++i) {
      const std::array token{emitted.at(i)};
      const auto result = cached.consume(token, true);
      const auto& expected =
          i + 1 < steps.size() ? steps.at(i + 1) : av.at("reference");
      if (!result.next_logits) {
        throw std::runtime_error("AV decode omitted logits");
      }
      if (i + 1 == emitted.size()) {
        check(result.next_logits->bytes == full_av_logits,
              "AV cached and full final logits differ on identical input");
      }
      auto m = compare(expected.at("logits"), result.next_logits->bytes,
                       storage, policy);
      passed = passed && m.at("accepted").get<bool>();
      numerical.push_back(m);
    }
    cached.finish();
  }
  const inference::GenerationSettings av_settings{
      {inference::sampling::SelectionMode::greedy, 1, 0, 1},
      42,
      av.at("max_new_tokens")};
  const auto first = run.interpret(key, av_settings),
             second = run.interpret(key, av_settings);
  check(first.value != second.value &&
            run.phase() == experiment::Phase::verbalizer_ready,
        "separate variants reuse resident AV");
  if (!actual) {
    check(run.retained_model_bytes().device_bytes > target_weight_bytes,
          "target and AV coexist when complete requests fit");
  }
  Json token_comparisons = Json::array();
  std::vector<std::uint32_t> previous_tokens;
  for (auto id : {first, second}) {
    const auto& v = run.record()->variant(id).interpretation;
    check(v.generation && v.outcome == experiment::AttemptOutcome::completed,
          "AV generation failed: " + v.failure_message);
    if (!v.generation) {
      throw std::runtime_error("AV variant omitted generation");
    }
    const auto& g = *v.generation;
    const auto expected =
        av.at("expected_generated_ids").get<std::vector<std::uint32_t>>();
    const bool matches = std::ranges::equal(
        std::span(g.tokens).subspan(g.prompt_tokens), expected);
    check(policy == ModelComparisonPolicy::report_differences || matches,
          "AV output IDs disagree with frozen reference");
    if (!previous_tokens.empty()) {
      check(g.tokens == previous_tokens,
            "AV repeated generation changed on the same backend");
    }
    previous_tokens = g.tokens;
    token_comparisons.push_back(
        {{"matches_reference", matches},
         {"actual_ids",
          std::vector<std::uint32_t>(
              g.tokens.begin() + static_cast<std::ptrdiff_t>(g.prompt_tokens),
              g.tokens.end())},
         {"expected_ids", expected}});
  }
  check(!run.record()->variant(second).interpretation.cost.load_seconds,
        "second variant reloaded AV weights");
  expect_throws<std::invalid_argument>(
      [&] { (void)run.interpret({site.point, {4096}}, av_settings); });
  std::stop_source cancelled;
  cancelled.request_stop();
  const auto cancelled_id =
      run.interpret(key, av_settings, cancelled.get_token());
  const auto& cancelled_attempt =
      run.record()->variant(cancelled_id).interpretation;
  check(cancelled_attempt.outcome == experiment::AttemptOutcome::cancelled &&
            !cancelled_attempt.generation &&
            !cancelled_attempt.cost.load_seconds,
        "pre-request cancellation retains an attempt without execution");
  const auto phase_cost = run.measurements();
  run.close();
  expect_throws<std::logic_error>(
      [&] { (void)run.interpret(key, av_settings); });
  check(run.record()->variant_count() == 3 &&
            export_capture(device, retained->device, retained->bytes,
                           retained_snapshot.size()) == retained_snapshot,
        "record survives closing model resources");
  return {
      {"passed", passed},
      {"numerical", numerical},
      {"completed_variants", 2},
      {"cancelled_variants", 1},
      {"estimated_peak_device_bytes", phase_cost.estimated_peak_device_bytes},
      {"estimated_peak_host_bytes", phase_cost.estimated_peak_host_bytes},
      {"token_comparisons", token_comparisons},
      {"normalization_max_absolute", normalization.maximum_absolute_error},
      {"target_capture",
       {{"passed", target_metric.passed},
        {"accepted", accepts_model_comparison(target_metric, policy)},
        {"max_absolute", target_metric.maximum_absolute_error},
        {"relative_l2", target_metric.relative_l2
                            ? Json(*target_metric.relative_l2)
                            : Json(nullptr)}}}};
}
Json check_steering(gpu::Device& device,
                    const std::filesystem::path& target_fixture,
                    ModelComparisonPolicy policy) {
  const auto fixture = read(target_fixture.parent_path() / "steering.json");
  const auto target = read(target_fixture);
  const auto& ar_data = fixture.at("ar");
  const auto storage = type(fixture.at("dtype"));
  const tensors::ComputationMode mode{storage, tensors::ElementType::float32};
  const auto width = ar_data.at("vectors").at(0).at("width").get<std::size_t>();
  const auto row_bytes = width * tensors::element_bytes(storage);
  Json metrics = Json::array();
  bool passed = true;
  const auto compare_row = [&](const Json& values,
                               std::span<const std::byte> actual,
                               const Json& tolerance,
                               tensors::ElementType dtype) {
    const auto decoded = decode(actual, dtype);
    const auto expected = values.get<std::vector<float>>();
    const auto metric = compare_values(
        expected, decoded,
        {tolerance.at("max_absolute_error"), tolerance.at("max_relative_l2")});
    const bool accepted = accepts_model_comparison(metric, policy);
    passed = passed && accepted;
    metrics.push_back(
        {{"passed", metric.passed},
         {"accepted", accepted},
         {"max_absolute", metric.maximum_absolute_error},
         {"relative_l2",
          metric.relative_l2 ? Json(*metric.relative_l2) : Json(nullptr)}});
  };
  inference::DeviceCapture direction;
  {
    auto ar_archive = tensors::formats::safetensors::open(
        request(ar_data, ar_data.at("root").get<std::string>()));
    auto checkpoint = models::qwen3::bind_reconstructor(
        std::move(ar_archive), ar_data.at("config").get<std::string>());
    auto ar = inference::load_resident_model(device, std::move(checkpoint),
                                             {mode, 23 * gib, 64 * mib});
    std::array<inference::DeviceCapture, 2> vectors;
    for (std::size_t i = 0; i < vectors.size(); ++i) {
      const auto ids =
          ar_data.at("prompts").at(i).get<std::vector<std::uint32_t>>();
      std::vector<std::byte> previous;
      for (const auto chunk : {std::size_t{1}, std::size_t{7}}) {
        inference::CachedForwardOptions options{
            mode, {100 + i}, ids.size(), std::min(ids.size(), chunk),
            {},   {0, 0},    23 * gib,   256 * mib};
        auto plan = inference::prepare_cached_forward(*ar, options);
        auto vector = plan.reconstruct(ids);
        const auto bytes = export_capture(device, vector, {}, row_bytes);
        compare_row(ar_data.at("vectors").at(i).at("values"), bytes,
                    ar_data.at("vectors").at(i).at("tolerance"), storage);
        if (!previous.empty()) {
          check(bytes == previous,
                "AR reconstruction changed with prefill chunk");
        }
        previous = bytes;
        vectors.at(i) = std::move(vector);
      }
    }
    auto queue = device.create_queue();
    inference::DeviceDirection difference(device, queue, vectors.at(0),
                                          vectors.at(1), storage, width);
    difference.submit();
    direction = difference.finish();
    const auto bytes =
        export_capture(device, direction, {}, width * sizeof(float));
    compare_row(fixture.at("direction"), bytes,
                fixture.at("direction_tolerance"),
                tensors::ElementType::float32);
    inference::DeviceDirection zero(device, queue, vectors.at(0), vectors.at(0),
                                    storage, width);
    zero.submit();
    expect_throws<std::invalid_argument>([&] { (void)zero.finish(); });
  }
  auto archive = tensors::formats::safetensors::open(
      request(target, target_fixture.parent_path()));
  auto resident = inference::load_resident_model(
      device, models::qwen3::bind_causal_lm(std::move(archive), "config.json"),
      {mode, 23 * gib, 64 * mib});
  const models::ObservationPoint site{models::ObservationBoundary::block_output,
                                      {fixture.at("block").get<std::size_t>()}};
  Json token_comparisons = Json::array();
  for (const auto& c : fixture.at("cases")) {
    const auto ids = c.at("prefix").get<std::vector<std::uint32_t>>();
    const auto expected = c.at("generated").get<std::vector<std::uint32_t>>();
    const auto position = c.at("position").get<std::size_t>();
    const auto capacity = ids.size() + expected.size();
    inference::CachedForwardOptions options{
        mode,     {201},
        capacity, std::min(ids.size(), std::size_t{7}),
        {site},   {capacity * row_bytes, capacity},
        23 * gib, 256 * mib};
    auto plan = inference::prepare_cached_forward(*resident, options);
    plan.prepare_intervention(
        {site, {position}, direction, width, c.at("strength")});
    std::uint32_t stop_id = 0;
    while (std::ranges::find(expected, stop_id) != expected.end()) {
      ++stop_id;
    }
    const text::PreparedPrompt prompt{
        "fixture", "fixture", {}, ids, {stop_id}, text::ThinkingMode::disabled};
    const inference::GenerationSettings settings{
        {inference::sampling::SelectionMode::greedy, 1, 0, 1},
        42,
        expected.size()};
    const auto result =
        inference::generate(plan, prompt, settings, {}, {201}, {},
                            inference::sampling::Execution::device);
    check(result.reason == inference::StopReason::token_limit,
          "steering generation failed: " + result.failure_message);
    const bool matches = std::ranges::equal(
        std::span(result.tokens).subspan(ids.size()), expected);
    check(policy == ModelComparisonPolicy::report_differences || matches,
          "steering generated IDs disagree with independent reference");
    token_comparisons.push_back(
        {{"matches_reference", matches},
         {"actual_ids",
          std::vector<std::uint32_t>(
              result.tokens.begin() + static_cast<std::ptrdiff_t>(ids.size()),
              result.tokens.end())},
         {"expected_ids", expected}});
    const auto& capture = result.captures.at(0);
    const auto bytes = export_capture(device, capture.device, capture.bytes,
                                      capacity * row_bytes);
    compare_row(c.at("after"),
                std::span(bytes).subspan(position * row_bytes, row_bytes),
                c.at("tolerance"), storage);
  }
  return {{"passed", passed},
          {"numerical", metrics},
          {"token_comparisons", token_comparisons},
          {"cases", fixture.at("cases").size()}};
}

}  // namespace dolittle::tests
