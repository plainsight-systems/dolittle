/*
 * R3 retention implementation, metal-port.md Memory and model residency:
 * validate exact source/binding identity and numerical mode; reserve the full
 * next request before retaining idle models. Conservative transient admission
 * keeps every recipe tensor live, adds KV, captures, sampling, staging and
 * native command headroom. Evict oldest idle weights until all budgets fit;
 * actual executor bounds remain enforced. All models are idle at these serial
 * phase boundaries. Unified accounting additionally includes mapped checkpoint
 * payloads because sharing with the uploaded arena is not established.
 * Optimization: stable per-role owners reuse uploaded weights when the next
 * complete request fits. No cache/sequence survives between generations.
 * Both implementation corpora queried/read: Core R.1/I.7/ES.103
 * (#rr-raii/#ri-post/#res-overflow), performance GPU.1/GPU.9:
 * guidelines/gpu/GPU.1-keep-data-on-device.md and
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "experiment/run.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "experiment/model_retention.h"
#include "gpu/device.h"
#include "gpu/memory.h"
#include "inference/decoding/device.h"
#include "inference/sampling/device.h"
#include "inference/steering/device.h"
#include "tensors/tensor.h"

namespace dolittle::experiment {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void accumulate(std::optional<double>& total, double value) {
  total = total.value_or(0) + value;
}
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::length_error("experiment size overflow");
  }
  return a + b;
}
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::length_error("experiment extent overflow");
  }
  return a * b;
}
constexpr std::size_t metadata_allowance = std::size_t{64} * 1024;
constexpr std::size_t upload_allowance = std::size_t{16} * 1024 * 1024;
bool same_source(const tensors::formats::SourceIdentity& a,
                 const tensors::formats::SourceIdentity& b) {
  return a.repository == b.repository && a.revision == b.revision &&
         a.artifact_digest == b.artifact_digest;
}
tokenizers::DecodedBytes decode_bounded(const tokenizers::Tokenizer& tokenizer,
                                        std::span<const std::uint32_t> ids,
                                        std::size_t limit) {
  tokenizers::DecodedBytes result;
  result.tokens.reserve(ids.size());
  result.bytes.reserve(limit);
  tokenizer.decode_into(ids, result);
  if (result.tokens.size() != ids.size() || result.bytes.size() > limit) {
    throw std::length_error("decoded byte budget");
  }
  return result;
}
}  // namespace
namespace {
class ProgressAdapter final : public inference::GenerationObserver {
 public:
  ProgressAdapter(RunObserver* observer, StreamRole role,
                  const tokenizers::Tokenizer& tokenizer,
                  const text::Protocol* protocol,
                  const text::PreparedPrompt& prompt, std::size_t byte_limit,
                  std::size_t max_generated, std::size_t prefill_chunk,
                  std::optional<models::ObservationSite> site = {},
                  std::optional<std::size_t> original_prompt = {})
      : observer_(observer),
        role_(role),
        tokenizer_(tokenizer),
        remaining_(byte_limit),
        site_(site) {
    if (observer) {
      decoded_.bytes.reserve(byte_limit);
      decoded_.tokens.reserve(std::max(prompt.ids.size(), std::size_t{1}));
      segments_.reserve(1);
      if (!observer->prepare_stream({prompt.ids.size(), max_generated,
                                     prefill_chunk, tokenizer.max_token_bytes(),
                                     byte_limit})) {
        observer->publication_failed();
        observer_ = nullptr;
        failed_ = true;
      }
    }
    if (observer && protocol) {
      const auto first = original_prompt.value_or(prompt.ids.size());
      if (first > prompt.ids.size()) {
        throw std::invalid_argument("replay prompt boundary");
      }
      cursor_ = protocol->make_cursor(first, prompt.thinking);
      if (!cursor_) {
        throw std::invalid_argument("protocol lacks incremental segmentation");
      }
      if (first < prompt.ids.size()) {
        (void)cursor_->append(std::span(prompt.ids).subspan(first), first);
      }
    }
  }
  bool update(const inference::GenerationUpdate& update) noexcept override {
    if (!observer_) {
      return !failed_;
    }
    try {
      tokenizer_.get().decode_into(update.appended, decoded_);
      if (decoded_.bytes.size() > remaining_ ||
          decoded_.tokens.size() != update.appended.size()) {
        throw std::length_error("decoded byte budget");
      }
      segments_.clear();
      if (cursor_ && update.first_position >= update.prompt_tokens) {
        cursor_->append_into(update.appended, update.first_position, segments_);
      }
      remaining_ -= decoded_.bytes.size();
      return observer_->progress({role_, update.first_position,
                                  update.prompt_tokens, update.consumed_tokens,
                                  std::cref(decoded_), segments_, site_});
    } catch (...) {
      observer_->publication_failed();
      return false;
    }
  }

 private:
  tokenizers::DecodedBytes decoded_;
  std::vector<text::TokenSegment> segments_;
  bool failed_ = false;
  RunObserver* observer_;
  StreamRole role_;
  std::reference_wrapper<const tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<text::SegmentCursor> cursor_;
  std::size_t remaining_;
  std::optional<models::ObservationSite> site_;
};
class Publication {
 public:
  Publication(RunObserver* observer, const std::stop_token& external)
      : observer_(observer),
        callback_(external, [this] { stopped_.request_stop(); }) {}
  void phase(WorkPhase value) {
    if (observer_ && !observer_->phase(value)) {
      observer_->publication_failed();
      observer_ = nullptr;
      stopped_.request_stop();
    }
  }
  [[nodiscard]] std::stop_token token() const { return stopped_.get_token(); }
  [[nodiscard]] RunObserver* observer() const { return observer_; }

 private:
  RunObserver* observer_;
  std::stop_source stopped_;
  std::stop_callback<std::function<void()>> callback_;
};
}  // namespace
struct Run::Impl {
  std::reference_wrapper<gpu::Device> device;
  RunId id;
  RunInputs inputs;
  RunLimits limits;
  Phase phase = Phase::empty;
  std::unique_ptr<Record> record;
  struct CachedModel {
    std::unique_ptr<inference::ResidentModel> owner;
    const models::BoundCheckpoint*
        checkpoint{};  // Borrowed from the stable owner.
    MemoryCost cost{};
    std::string identity;
    std::uint64_t used{};
  };
  std::array<CachedModel, 3> models;
  nla::ModelRole active = nla::ModelRole::target;
  std::uint64_t use_clock{};
  measurement::PhaseMeasurements measurements;
  std::uint64_t execution = 0;

  Impl(gpu::Device& d, RunId run, RunInputs in, RunLimits l)
      : device(d), id(run), inputs(std::move(in)), limits(l) {}
  inference::ExecutionId next_id() {
    if (execution == std::numeric_limits<std::uint64_t>::max()) {
      throw std::length_error("execution identity exhausted");
    }
    return {++execution};
  }
  std::size_t result_allowance(std::size_t tokens,
                               std::size_t capture_width) const {
    auto bytes = add(metadata_allowance, multiply(2, limits.max_decoded_bytes));
    bytes = add(bytes, multiply(tokens, 2 * sizeof(tokenizers::TokenBytes) +
                                            sizeof(text::TokenSegment) +
                                            2 * sizeof(std::uint32_t) +
                                            2 * sizeof(CaptureKey) +
                                            2 * sizeof(std::size_t)));
    if (inputs.selection_execution ==
        inference::sampling::Execution::host_reference) {
      bytes = add(bytes, multiply(tokens, capture_width));
    }
    bytes = add(bytes, multiply(limits.record.max_variants,
                                sizeof(std::unique_ptr<Variant>)));
    return bytes;
  }
  std::size_t resident_host() const {
    std::size_t bytes = 0;
    for (const auto& entry : models) {
      if (entry.owner) bytes = add(bytes, entry.cost.host_bytes);
    }
    return bytes;
  }
  std::size_t inactive_device() const {
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < models.size(); ++i) {
      if (models.at(i).owner && i != static_cast<std::size_t>(active)) {
        bytes = add(bytes, models.at(i).cost.device_bytes);
      }
    }
    return bytes;
  }
  struct ModelRequest {
    std::size_t capacity, pending;
    bool captures;
    std::size_t extra_device{};
  };
  inference::ResidentModel* acquire(nla::ModelRole role, ModelRequest request) {
    const auto [capacity, pending, captures, extra_device] = request;
    const auto snapshot = gpu::query_memory(device);
    const bool unified = snapshot.topology == gpu::MemoryTopology::unified;
    const auto retained_host =
        add(record ? record->retained_bytes() : 0, inputs.metadata_host_bytes);
    const auto required_host = add(retained_host, pending);
    if (required_host >= limits.max_host_bytes) {
      throw std::length_error("model admission host budget");
    }
    auto& entry = models.at(static_cast<std::size_t>(role));
    if (entry.owner && (!entry.owner->usable() ||
                        entry.identity != entry.owner->checkpoint_identity())) {
      entry = {};
    }
    LoadedModel loaded{};
    if (!entry.owner) {
      loaded = load(role, limits.max_host_bytes - required_host);
    }
    const auto* checkpoint =
        entry.owner ? entry.checkpoint : loaded.checkpoint.get();
    if (!checkpoint) {
      throw std::logic_error("resident checkpoint lease missing");
    }
    const std::string identity(checkpoint->identity());
    // Pair identity and numerical modes are immutable for this Run. A cache
    // hit uses its already-validated checkpoint; do not reopen/reverify files.
    const auto mode = role == nla::ModelRole::verbalizer
                          ? inputs.verbalizer_mode
                          : inputs.target_mode;
    const auto chunk = std::min(capacity, limits.max_prefill_chunk);
    const auto recipe = checkpoint->model().incremental_forward({chunk}, mode);
    // Conservative preparation bound: keep every non-parameter tensor live,
    // including separate prefill/decode storage. Actual executor placement may
    // use less; it still receives the remaining strict budget below.
    std::vector<bool> parameter(recipe.forward.tensors.size(), false);
    std::size_t weights = 0;
    constexpr std::size_t weight_alignment = 256;
    for (const auto& specification : checkpoint->model().parameters()) {
      weights = add(weights, (weight_alignment - weights % weight_alignment) %
                                 weight_alignment);
      weights = add(weights, tensors::required_bytes(specification.layout));
    }
    for (const auto& binding : recipe.forward.parameters) {
      parameter.at(binding.tensor.value) = true;
    }
    std::size_t scratch = 0;
    for (std::size_t i = 0; i < parameter.size(); ++i) {
      if (!parameter.at(i)) {
        scratch = add(scratch,
                      add(tensors::required_bytes(recipe.forward.tensors.at(i)),
                          weight_alignment));
      }
    }
    std::size_t cache = 0;
    for (const auto& append : recipe.cache_appends) {
      const auto& layout = recipe.forward.tensors.at(append.rotated_key.value);
      cache =
          add(cache, multiply(multiply(multiply(capacity, layout.shape.at(1)),
                                       layout.shape.at(2)),
                              2 * tensors::element_bytes(layout.element_type)));
    }
    const auto output_id = recipe.forward.terminal_output();
    if (!output_id) {
      throw std::invalid_argument("model has no terminal output");
    }
    const auto& output = recipe.forward.tensors.at(output_id->value);
    const auto output_bytes = tensors::required_bytes(output);
    const auto row = multiply(inputs.pair->target_site().vector_width,
                              tensors::element_bytes(mode.activation_storage));
    const auto capture_bytes = captures ? multiply(capacity, row) : 0;
    const auto journal =
        inference::DeviceSequence::host_bytes({capacity, capacity, chunk});
    constexpr std::size_t cuda_command_allowance = 32ULL * 1024 * 1024;
    constexpr std::size_t metal_command_allowance = 512ULL * 1024 * 1024;
    const auto commands =
        unified ? metal_command_allowance : cuda_command_allowance;
    const auto sampling =
        checkpoint->model().output_kind() == models::OutputKind::vocabulary
            ? inference::sampling::DeviceCategorical::required_bytes(
                  output.shape.back())
            : 0;
    const auto transient_device =
        add(add(add(multiply(2, scratch), cache), capture_bytes),
            add(commands, add(extra_device, sampling)));
    const auto transient_host =
        add(upload_allowance,
            add(journal, add(multiply(2, output_bytes),
                             multiply(capacity, 4 * sizeof(std::uint64_t)))));
    MemoryCost active_cost{weights, loaded.host_working_bytes, 0};
    if (unified) {
      active_cost.unified_bytes =
          add(add(weights, checkpoint->parameter_bytes()),
              loaded.host_working_bytes);
    }
    if (entry.owner) active_cost = entry.cost;
    std::vector<ResidentEntry> resident;
    std::size_t owned_device = retained_device_bytes();
    for (std::size_t i = 0; i < models.size(); ++i) {
      if (models.at(i).owner) {
        resident.push_back({static_cast<nla::ModelRole>(i), models.at(i).cost,
                            models.at(i).used, false});
        owned_device = add(owned_device, models.at(i).cost.device_bytes);
      }
    }
    auto device_limit =
        std::min(limits.max_device_bytes, snapshot.working_set_limit_bytes);
    if (snapshot.available_device_bytes) {
      device_limit = std::min(
          device_limit, add(*snapshot.available_device_bytes, owned_device));
    }
    const MemoryCost retained{
        retained_device_bytes(), required_host,
        unified ? add(retained_device_bytes(), required_host) : 0};
    const MemoryCost transient{
        transient_device, transient_host,
        unified ? add(transient_device, transient_host) : 0};
    const auto decision = plan_model_retention(
        {device_limit, limits.max_host_bytes,
         unified ? std::optional(snapshot.working_set_limit_bytes)
                 : std::nullopt},
        {role, active_cost, retained, transient}, resident);
    if (!decision) {
      throw std::length_error(
          "model and complete request do not fit memory budget");
    }
    for (const auto victim : decision->evict) {
      models.at(static_cast<std::size_t>(victim)) = {};
    }
    active = role;
    if (!entry.owner) {
      const auto host = available_host(pending);
      if (loaded.host_working_bytes >= host) {
        throw std::length_error("model upload host budget");
      }
      entry.owner = inference::load_resident_model(
          device, std::move(loaded.checkpoint),
          {mode, available_device(),
           std::min(upload_allowance, host - loaded.host_working_bytes)});
      if (entry.owner->resident_device_bytes() != weights) {
        throw std::logic_error("resident weight accounting mismatch");
      }
      entry.checkpoint = checkpoint;
      entry.cost = active_cost;
      entry.identity = identity;
    }
    if (use_clock == std::numeric_limits<std::uint64_t>::max()) {
      throw std::length_error("model use counter exhausted");
    }
    entry.used = ++use_clock;
    return entry.owner.get();
  }
  std::size_t available_host(std::size_t pending) const {
    const auto retained = record ? record->retained_bytes() : 0;
    const auto used = add(add(retained, pending),
                          add(inputs.metadata_host_bytes, resident_host()));
    if (used >= limits.max_host_bytes) {
      throw std::length_error("experiment host budget");
    }
    return limits.max_host_bytes - used;
  }
  LoadedModel load(nla::ModelRole role, std::size_t allowance) {
    auto result =
        (role == nla::ModelRole::target ? inputs.target
                                        : (role == nla::ModelRole::reconstructor
                                               ? inputs.reconstructor
                                               : inputs.verbalizer))(allowance);
    if (!result.checkpoint || result.host_working_bytes > allowance ||
        !same_source(result.source, result.checkpoint->source_identity())) {
      throw std::invalid_argument(
          "provider exceeded allowance or returned inconsistent source");
    }
    inputs.pair->validate_model(role, result.source, *result.checkpoint,
                                role == nla::ModelRole::verbalizer
                                    ? inputs.verbalizer_mode
                                    : inputs.target_mode);
    return result;
  }
  std::size_t retained_device_bytes() const {
    std::size_t bytes = 0;
    if (record) {
      for (const auto& capture : record->target().generation.captures) {
        if (capture.device) {
          bytes = add(bytes, capture.device->bytes);
        }
      }
    }
    return bytes;
  }
  std::size_t available_device() const {
    const auto used = add(retained_device_bytes(), inactive_device());
    if (used >= limits.max_device_bytes) {
      throw std::length_error("retained device budget");
    }
    return limits.max_device_bytes - used;
  }
  inference::CachedForwardOptions options(std::size_t capacity,
                                          inference::ExecutionId execution_id,
                                          bool target, std::size_t host) const {
    const auto site = inputs.pair->target_site();
    const auto row =
        multiply(site.vector_width, tensors::element_bytes(site.storage_type));
    return {target ? inputs.target_mode : inputs.verbalizer_mode,
            {execution_id.value},
            capacity,
            std::min(capacity, limits.max_prefill_chunk),
            target ? std::vector<models::ObservationPoint>{site.point}
                   : std::vector<models::ObservationPoint>{},
            {target ? multiply(capacity, row) : 0, target ? capacity : 0},
            available_device(),
            host};
  }
};
Run::Run(gpu::Device& device, RunId id, RunInputs inputs, RunLimits limits)
    : impl_(std::make_unique<Impl>(device, id, std::move(inputs), limits)) {
  const auto& in = impl_->inputs;
  if (!in.pair || !in.target || !in.verbalizer || !in.target_tokenizer ||
      !in.verbalizer_tokenizer || !in.target_protocol ||
      (in.selection_execution ==
           inference::sampling::Execution::host_reference &&
       !in.selector) ||
      in.backend_identity.empty() || !limits.max_prefill_chunk ||
      !limits.max_target_tokens || !limits.max_verbalizer_tokens ||
      !limits.max_decoded_bytes || !limits.max_device_bytes ||
      !limits.record.max_variants ||
      add(limits.record.max_retained_bytes, in.metadata_host_bytes) >=
          limits.max_host_bytes) {
    throw std::invalid_argument("invalid experiment configuration");
  }
  if (!in.target_tokenizer->identity().starts_with(
          in.pair->identity().target.repository + "/" +
          in.pair->identity().target.revision + "/" +
          in.pair->identity().target.artifact_digest + "/") ||
      !in.verbalizer_tokenizer->identity().starts_with(
          in.pair->identity().verbalizer.repository + "/" +
          in.pair->identity().verbalizer.revision + "/" +
          in.pair->identity().verbalizer.artifact_digest + "/") ||
      in.pair->verbalizer_prompt().tokenizer_identity !=
          in.verbalizer_tokenizer->identity()) {
    throw std::invalid_argument("experiment tokenizer provenance mismatch");
  }
}
Run::~Run() = default;
Phase Run::phase() const noexcept { return impl_->phase; }
const Record* Run::record() const noexcept { return impl_->record.get(); }
const measurement::PhaseMeasurements& Run::measurements() const noexcept {
  return impl_->measurements;
}
gpu::MemoryFootprint Run::retained_model_bytes() const {
  gpu::MemoryFootprint result;
  for (const auto& entry : impl_->models) {
    if (entry.owner) {
      result.device_bytes = add(result.device_bytes, entry.cost.device_bytes);
      result.host_bytes = add(result.host_bytes, entry.cost.host_bytes);
      result.unified_bytes =
          add(result.unified_bytes, entry.cost.unified_bytes);
    }
  }
  return result;
}
void Run::release_models() {
  for (auto& entry : impl_->models) entry = {};
  if (impl_->phase == Phase::verbalizer_ready) impl_->phase = Phase::retained;
}
void Run::close() {
  for (auto& entry : impl_->models) entry = {};
  impl_->phase = Phase::closed;
}
void Run::generate_impl(const text::Prompt& prompt,
                        inference::GenerationSettings settings,
                        RunObserver* observer,
                        const std::stop_token& external_stop) {
  Publication publication(observer, external_stop);
  const auto stop = publication.token();
  auto& p = *impl_;
  if (p.phase != Phase::empty) {
    throw std::logic_error("target generation already started or run closed");
  }
  inference::sampling::validate_settings(settings.sampling);
  const auto prepared =
      p.inputs.target_protocol->prepare(prompt, *p.inputs.target_tokenizer);
  if (prepared.formatted_utf8.size() > p.limits.max_decoded_bytes) {
    throw std::length_error("formatted target prompt byte budget");
  }
  const auto capacity = add(prepared.ids.size(), settings.max_new_tokens);
  if (!capacity || capacity > p.limits.max_target_tokens) {
    throw std::length_error("target token budget");
  }
  const auto site = p.inputs.pair->target_site();
  const auto retained = p.result_allowance(
      capacity,
      multiply(site.vector_width, tensors::element_bytes(site.storage_type)));
  if (retained > p.limits.record.max_retained_bytes) {
    throw std::length_error("target retained budget");
  }
  auto started = Clock::now();
  publication.phase(WorkPhase::loading_target);
  auto* resident =
      p.acquire(nla::ModelRole::target, {capacity, retained, true});
  const auto host = p.available_host(retained);
  const std::size_t working =
      0;  // Cached model host ownership is already reserved.
  p.measurements.target_load_seconds = elapsed(started);
  const auto execution = p.next_id();
  // Preflight failures above remain retryable. Once sequence execution
  // begins, unexpected failure closes the run rather than silently repeating
  // a target.
  try {
    auto plan = inference::prepare_cached_forward(
        *resident, p.options(capacity, execution, true, host - working));
    if (p.inputs.selection_execution ==
        inference::sampling::Execution::device) {
      plan.prepare_sampling();
    }
    p.measurements.estimated_peak_device_bytes =
        plan.estimated_peak_device_bytes();
    p.measurements.estimated_peak_host_bytes =
        add(add(p.inputs.metadata_host_bytes, retained),
            add(working, std::max(plan.estimated_peak_host_bytes(),
                                  std::min(upload_allowance, host - working))));
    started = Clock::now();
    publication.phase(WorkPhase::generating_target);
    ProgressAdapter adapter(
        publication.observer(), StreamRole::target, *p.inputs.target_tokenizer,
        p.inputs.target_protocol.get(), prepared, p.limits.max_decoded_bytes,
        settings.max_new_tokens, plan.max_prefill_chunk(), site);
    auto result = inference::generate(plan, prepared, settings,
                                      p.inputs.selector, execution, adapter,
                                      stop, p.inputs.selection_execution);
    p.measurements.target_generate_seconds = elapsed(started);
    p.measurements.estimated_peak_device_bytes =
        plan.estimated_peak_device_bytes();
    p.measurements.estimated_peak_host_bytes =
        add(add(p.inputs.metadata_host_bytes, retained),
            add(working, std::max(plan.estimated_peak_host_bytes(),
                                  std::min(upload_allowance, host - working))));
    started = Clock::now();
    auto decoded = decode_bounded(*p.inputs.target_tokenizer, result.tokens,
                                  p.limits.max_decoded_bytes);
    auto segments = p.inputs.target_protocol->segment(
        std::span(result.tokens).subspan(result.prompt_tokens),
        result.prompt_tokens, prompt.thinking);
    p.measurements.decode_seconds = elapsed(started);
    auto record = std::make_unique<Record>(
        TargetRecord{p.id, p.inputs.pair->identity(), site,
                     prepared.formatted_utf8, std::move(result),
                     std::move(decoded), std::move(segments),
                     p.inputs.backend_identity},
        p.limits.record);
    p.record = std::move(record);
    publication.phase(WorkPhase::releasing_target);
    started = Clock::now();  // Include sequence teardown and resident
                             // release in this phase.
  } catch (...) {
    p.phase = Phase::closed;
    throw;
  }

  p.measurements.target_release_seconds = elapsed(started);
  p.phase = p.record->target().generation.reason ==
                    inference::StopReason::execution_failure
                ? Phase::closed
                : Phase::retained;
  publication.phase(WorkPhase::target_retained);
}
VariantId Run::interpret_impl(CaptureKey key,
                              inference::GenerationSettings settings,
                              RunObserver* observer,
                              const std::stop_token& external_stop) {
  Publication publication(observer, external_stop);
  const auto stop = publication.token();
  auto& p = *impl_;
  if (p.phase != Phase::retained && p.phase != Phase::verbalizer_ready) {
    throw std::logic_error("run cannot interpret");
  }
  inference::sampling::validate_settings(settings.sampling);
  const auto row = p.record->capture(key);
  if (!row) {
    throw std::invalid_argument("capture unavailable");
  }
  const auto& prompt = p.inputs.pair->verbalizer_prompt();
  const auto capacity = add(prompt.ids.size(), settings.max_new_tokens);
  if (!capacity || capacity > p.limits.max_verbalizer_tokens) {
    throw std::length_error("AV token budget");
  }
  const auto pending = p.result_allowance(capacity, 0);
  if (p.record->variant_count() >= p.limits.record.max_variants ||
      pending >
          p.limits.record.max_retained_bytes - p.record->retained_bytes()) {
    throw std::length_error("AV retained budget");
  }
  const auto conversion_bytes =
      row->device
          ? 0
          : multiply(row->site.vector_width,
                     sizeof(float) +
                         2 * tensors::element_bytes(
                                 p.inputs.verbalizer_mode.activation_storage));
  std::size_t host = 0;
  Interpretation attempt{};
  attempt.capture = key;
  attempt.requested_settings = settings;
  if (stop.stop_requested()) {
    attempt.outcome = AttemptOutcome::cancelled;
    return p.record->append(std::move(attempt));
  }
  try {
    auto start = Clock::now();
    auto injection = p.inputs.pair->prepare_injection(*row);
    if (!row->device) {
      attempt.cost.conversion_seconds = elapsed(start);
    }
    const InjectionProvenance provenance{
        injection.embedding.position, injection.source_l2_norm, injection.scale,
        injection.embedding.storage_type, p.inputs.pair->identity().convention};
    if (!row->device) {
      attempt.injection = provenance;
    }
    attempt.stage = AttemptStage::loading;
    start = Clock::now();
    const bool reused = p.models.at(1).owner && p.models.at(1).owner->usable();
    publication.phase(WorkPhase::loading_verbalizer);
    (void)p.acquire(nla::ModelRole::verbalizer,
                    {capacity, add(pending, conversion_bytes), false});
    host = p.available_host(add(pending, conversion_bytes));
    if (!reused) {
      attempt.cost.load_seconds = elapsed(start);
      accumulate(p.measurements.verbalizer_load_seconds,
                 *attempt.cost.load_seconds);
    }
    p.phase = Phase::verbalizer_ready;
    attempt.stage = AttemptStage::preparation;
    const auto execution = p.next_id();
    start = Clock::now();
    {
      auto plan = inference::prepare_cached_forward(
          *p.models.at(1).owner, p.options(capacity, execution, false, host),
          std::move(injection.embedding));
      if (p.inputs.selection_execution ==
          inference::sampling::Execution::device) {
        plan.prepare_sampling();
      }
      attempt.cost.prepare_seconds = elapsed(start);
      p.measurements.estimated_peak_device_bytes =
          std::max(p.measurements.estimated_peak_device_bytes,
                   plan.estimated_peak_device_bytes());
      const auto estimated_host = add(
          add(p.record->retained_bytes(), add(pending, conversion_bytes)),
          add(add(p.inputs.metadata_host_bytes, p.models.at(1).cost.host_bytes),
              std::max(plan.estimated_peak_host_bytes(),
                       std::min(upload_allowance, host))));
      p.measurements.estimated_peak_host_bytes =
          std::max(p.measurements.estimated_peak_host_bytes, estimated_host);
      attempt.stage = AttemptStage::generation;
      start = Clock::now();
      publication.phase(WorkPhase::generating_verbalizer);
      ProgressAdapter adapter(publication.observer(), StreamRole::verbalizer,
                              *p.inputs.verbalizer_tokenizer, nullptr, prompt,
                              p.limits.max_decoded_bytes,
                              settings.max_new_tokens,
                              plan.max_prefill_chunk());
      attempt.generation = inference::generate(
          plan, prompt, settings, p.inputs.selector, execution, adapter, stop,
          p.inputs.selection_execution);
      p.measurements.estimated_peak_device_bytes = std::max(
          p.measurements.estimated_peak_device_bytes,
          add(p.retained_device_bytes(), plan.estimated_peak_device_bytes()));
      p.measurements.estimated_peak_host_bytes = std::max(
          p.measurements.estimated_peak_host_bytes,
          add(add(p.record->retained_bytes(), add(pending, conversion_bytes)),
              add(add(p.inputs.metadata_host_bytes,
                      p.models.at(1).cost.host_bytes),
                  plan.estimated_peak_host_bytes())));
      if (attempt.generation->injection_norm) {
        attempt.injection = provenance;
        attempt.injection->source_l2_norm = *attempt.generation->injection_norm;
      }
      attempt.cost.generate_seconds = elapsed(start);
      accumulate(p.measurements.verbalizer_generate_seconds,
                 *attempt.cost.generate_seconds);
      start = Clock::now();
    }
    attempt.cost.release_seconds = elapsed(start);
    attempt.stage = AttemptStage::decoding;
    start = Clock::now();
    const auto& result = *attempt.generation;
    attempt.decoded =
        decode_bounded(*p.inputs.verbalizer_tokenizer,
                       std::span(result.tokens).subspan(result.prompt_tokens),
                       p.limits.max_decoded_bytes);
    attempt.explanation = p.inputs.pair->explanation(attempt.decoded.bytes);
    attempt.cost.decode_seconds = elapsed(start);
    accumulate(p.measurements.decode_seconds, *attempt.cost.decode_seconds);
    attempt.outcome = result.reason == inference::StopReason::execution_failure
                          ? AttemptOutcome::failed
                          : (result.reason == inference::StopReason::user_stop
                                 ? AttemptOutcome::cancelled
                                 : AttemptOutcome::completed);
    attempt.failure_message = result.failure_message;
    attempt.stage = AttemptStage::complete;
  } catch (const std::bad_alloc&) {
    if (p.models.at(1).owner && !p.models.at(1).owner->usable()) {
      close();
    }
    throw;
  } catch (const std::exception& e) {
    attempt.outcome = AttemptOutcome::failed;
    attempt.failure_message = e.what();
  }
  if (p.models.at(1).owner && !p.models.at(1).owner->usable()) {
    p.models.at(1).owner.reset();
    p.models.at(1).cost.host_bytes = 0;
    p.phase = Phase::closed;
  }
  const auto id = p.record->append(std::move(attempt));
  publication.phase(p.phase == Phase::closed ? WorkPhase::closed
                                             : WorkPhase::verbalizer_ready);
  return id;
}
void Run::generate(const text::Prompt& prompt,
                   inference::GenerationSettings settings,
                   const std::stop_token& stop) {
  generate_impl(prompt, settings, nullptr, stop);
}
void Run::generate(const text::Prompt& prompt,
                   inference::GenerationSettings settings,
                   RunObserver& observer, const std::stop_token& stop) {
  generate_impl(prompt, settings, &observer, stop);
}
VariantId Run::interpret(CaptureKey key, inference::GenerationSettings settings,
                         const std::stop_token& stop) {
  return interpret_impl(key, settings, nullptr, stop);
}
VariantId Run::interpret(CaptureKey key, inference::GenerationSettings settings,
                         RunObserver& observer, const std::stop_token& stop) {
  return interpret_impl(key, settings, &observer, stop);
}
TrialId Run::steer(SteeringEdit edit, RunObserver& observer,
                   const std::stop_token& external_stop) {
  auto& p = *impl_;
  if (!p.record || p.phase == Phase::closed || !p.inputs.reconstructor ||
      !p.inputs.reconstructor_tokenizer ||
      !p.inputs.pair->identity().reconstructor ||
      p.inputs.selection_execution != inference::sampling::Execution::device) {
    throw std::invalid_argument("steering unavailable");
  }
  const auto& sample = p.record->variant(edit.variant).interpretation;
  if (sample.outcome != AttemptOutcome::completed ||
      sample.explanation.format != nla::ExplanationFormat::complete ||
      !sample.explanation.content || !p.record->capture(sample.capture) ||
      !std::isfinite(edit.strength) || edit.strength < 0 || edit.strength > 2 ||
      edit.explanation.empty() ||
      edit.explanation.size() > nla::max_explanation_bytes) {
    throw std::invalid_argument(
        "steering needs a complete sample and valid edit");
  }
  const auto& target = p.record->target();
  const auto& g = target.generation;
  const auto prefix = replay_prefix(target, sample.capture);
  const auto capacity = add(prefix, edit.settings.max_new_tokens);
  inference::sampling::validate_settings(edit.settings.sampling);
  if (!edit.settings.max_new_tokens || capacity > p.limits.max_target_tokens ||
      p.record->trial_count() >= p.limits.record.max_trials) {
    throw std::length_error("steering trial/context limit");
  }
  SteeringTrial trial{};
  trial.variant = edit.variant;
  trial.capture = sample.capture;
  const auto range = *sample.explanation.content;
  trial.original =
      sample.decoded.bytes.substr(range.begin, range.end - range.begin);
  trial.edited = std::move(edit.explanation);
  trial.strength = edit.strength;
  trial.settings = edit.settings;
  trial.prefix_tokens = prefix;
  trial.original_prompt_tokens = g.prompt_tokens;
  trial.reconstructor = p.inputs.pair->identity().reconstructor;
  if (trial.strength > 0 && trial.original == trial.edited) {
    throw std::invalid_argument("edit the explanation before steering");
  }
  trial.original_ar_tokens = p.inputs.pair->reconstruction_prompt(
      trial.original, *p.inputs.reconstructor_tokenizer);
  trial.edited_ar_tokens = p.inputs.pair->reconstruction_prompt(
      trial.edited, *p.inputs.reconstructor_tokenizer);
  const auto pending =
      add(multiply(2, p.result_allowance(capacity, 0)),
          add(trial.original.capacity(),
              add(trial.edited.capacity(),
                  multiply(add(trial.original_ar_tokens.capacity(),
                               trial.edited_ar_tokens.capacity()),
                           sizeof(std::uint32_t)))));
  if (pending >
      p.limits.record.max_retained_bytes - p.record->retained_bytes()) {
    throw std::length_error("steering retained budget");
  }
  std::size_t host = 0;
  observer.prepare_comparison({prefix, edit.settings.max_new_tokens,
                               std::min(capacity, p.limits.max_prefill_chunk),
                               p.inputs.target_tokenizer->max_token_bytes(),
                               p.limits.max_decoded_bytes});
  if (!observer.comparison_started(trial)) {
    observer.publication_failed();
    throw std::runtime_error("comparison publication failed");
  }
  Publication publication(&observer, external_stop);
  const auto stop = publication.token();
  p.phase = Phase::retained;
  try {
    inference::DeviceCapture direction{};
    const auto check_stop = [&] {
      if (stop.stop_requested()) {
        throw std::runtime_error("Comparison stopped.");
      }
    };
    check_stop();
    if (trial.strength > 0) {
      publication.phase(WorkPhase::loading_reconstructor);
      const auto width = p.inputs.pair->target_site().vector_width;
      const auto direction_allowance =
          add(multiply(width, 2 * sizeof(float)),
              inference::DeviceDirection::required_bytes(width));
      auto* ar = p.acquire(nla::ModelRole::reconstructor,
                           {std::max(trial.original_ar_tokens.size(),
                                     trial.edited_ar_tokens.size()),
                            pending, false, direction_allowance});
      host = p.available_host(pending);
      check_stop();
      publication.phase(WorkPhase::reconstructing);
      const auto reconstruct = [&](std::span<const std::uint32_t> ids,
                                   std::size_t retained) {
        const auto execution = p.next_id();
        auto options = p.options(ids.size(), execution, false, host);
        options.max_device_bytes = p.available_device() - retained;
        auto plan = inference::prepare_cached_forward(*ar, options);
        return plan.reconstruct(ids, stop);
      };
      auto original = reconstruct(trial.original_ar_tokens, 0);
      check_stop();
      auto edited =
          reconstruct(trial.edited_ar_tokens, original.owner->size_bytes());
      check_stop();

      const auto vectors =
          add(original.owner->size_bytes(), edited.owner->size_bytes());
      const auto extra = inference::DeviceDirection::required_bytes(width);
      if (add(add(ar->resident_device_bytes(), vectors), extra) >
          p.available_device()) {
        throw std::length_error("direction device budget");
      }
      auto queue = p.device.get().create_queue();
      inference::DeviceDirection difference(
          p.device, queue, std::move(original), std::move(edited),
          p.inputs.target_mode.activation_storage, width);
      difference.submit();
      direction = difference.finish();
    }
    check_stop();
    publication.phase(WorkPhase::loading_target);
    const auto direction_bytes =
        direction.owner ? direction.owner->size_bytes() : 0;
    auto* resident = p.acquire(nla::ModelRole::target,
                               {capacity, pending, false, direction_bytes});
    host = p.available_host(pending);
    const auto device_budget = p.available_device() - direction_bytes;
    text::PreparedPrompt prompt{
        g.protocol_identity,
        g.tokenizer_identity,
        {},
        {g.tokens.begin(),
         g.tokens.begin() + static_cast<std::ptrdiff_t>(prefix)},
        g.stop_ids,
        g.thinking};
    const auto branch = [&](BranchResult& result, StreamRole role) {
      check_stop();
      const auto execution = p.next_id();
      auto options = p.options(capacity, execution, false, host);
      options.max_device_bytes = device_budget;
      auto plan = inference::prepare_cached_forward(*resident, options);
      if (role == StreamRole::steered && trial.strength > 0) {
        plan.prepare_intervention(
            {trial.capture.site, trial.capture.position, direction,
             p.inputs.pair->target_site().vector_width, trial.strength});
      }
      publication.phase(role == StreamRole::baseline
                            ? WorkPhase::generating_baseline
                            : WorkPhase::generating_steered);
      ProgressAdapter adapter(
          publication.observer(), role, *p.inputs.target_tokenizer,
          p.inputs.target_protocol.get(), prompt, p.limits.max_decoded_bytes,
          edit.settings.max_new_tokens, plan.max_prefill_chunk(), {},
          g.prompt_tokens);
      result.generation = inference::generate(
          plan, prompt, edit.settings, p.inputs.selector, execution, adapter,
          stop, inference::sampling::Execution::device);
      const auto& output = *result.generation;
      result.decoded = decode_bounded(*p.inputs.target_tokenizer,
                                      std::span(output.tokens).subspan(prefix),
                                      p.limits.max_decoded_bytes);
      auto segments = p.inputs.target_protocol->segment(
          std::span(output.tokens).subspan(g.prompt_tokens), g.prompt_tokens,
          g.thinking);
      result.segments.assign(segments.begin() + static_cast<std::ptrdiff_t>(
                                                    prefix - g.prompt_tokens),
                             segments.end());
      if (output.reason == inference::StopReason::execution_failure) {
        throw std::runtime_error(output.failure_message.empty()
                                     ? "Branch execution failed."
                                     : output.failure_message);
      }
      check_stop();
    };
    branch(trial.baseline, StreamRole::baseline);
    branch(trial.steered, StreamRole::steered);
    trial.outcome = AttemptOutcome::completed;
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception& error) {
    trial.outcome = stop.stop_requested() ? AttemptOutcome::cancelled
                                          : AttemptOutcome::failed;
    trial.message = error.what();
  }
  const auto id = p.record->append(std::move(trial));
  publication.phase(WorkPhase::target_retained);
  return id;
}

}  // namespace dolittle::experiment
