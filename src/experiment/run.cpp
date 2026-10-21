/*
 * Steering scheduling, R3 (steering.md): resolve exact retained explanation and
 * replay IDs, admit both results, then release AV. Load AR once, forward both
 * descriptions with fresh KV, and normalize their difference on GPU. Retain
 * only the direction while switching to target. Reuse target weights for two
 * fresh-KV sequences with matched seed/settings, applying one residual update
 * in the steered branch. Classification replays the fixed generated prefix
 * before observing new IDs. Exceptions/cancellation retain a partial trial.
 * All vector operations and per-token decisions stay on GPU; CPU phase changes
 * follow completion. Budgets include original captures and surviving vectors.
 * Implementation corpora queried/read 2026-10-20: Core I.7 (#ri-post), explicit
 * results/lifetimes; GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md),
 resident
 * intermediates; GDSA.2, unchanged fixed transformer arithmetic.
 * Axis R3: sequential target/AV phases with durable experiment records.
 * Contract: Run owns pairing, providers, text helpers, records and optional
 * resident AV weights; it borrows Device, which must outlive capture leases.
 * Providers load only verified sources within their host allowance. Operations
 * are serial. Target executes once; later interpretations use fresh sequence
 * state and may reuse AV weights. Rejected preflight is retryable; unexpected
 * target failure closes the run. close releases model resources, while the
 * immutable record and its capture leases remain readable until destruction.
 *
 * Algorithm: logical-overview.md experiment phases; execution-boundary.md
 H0-H4.
 * 1. Prepare/tokenize target text, validate pairing and budgets, load target
 *    weights, then prepare and execute the complete GPU sequence. The host
 *    observer detokenizes/classifies metadata without scheduling GPU progress.
 * 2. Build the retained record from completed device capture leases and exact
 *    IDs. Release target sequence/cache/weights; retained captures remain
 valid.
 * 3. Resolve a selected consumed row, validate the trained pair convention and
 *    construct a device normalization descriptor. Admit the AV model/sequence
 *    alongside retained captures. Reuse AV weights on subsequent requests.
 * 4. GPU normalization/status gates AV prefill; marker embedding replacement
 *    precedes block0. Run the same GPU generation path, retain reported source
 *    norm and exact AV IDs, decode them and extract explanation tags.
 * 5. Append a completed/cancelled/failed interpretation with honest phase costs
 *    and provenance. A load failure can be retried; device failure prevents
 *    pretending execution succeeded. Cancellation is an external stop request.
 * Host-reference selection remains an explicitly composed diagnostic path with
 * host captures/conversion. It is not a live fallback.
 * Cost: peak device=max(target phase, retained captures+AV phase), including
 KV,
 * scratch, sampler, constants, graph state and prepared journal allocations.
 * Host estimates include metadata, verified loading and diagnostic staging.
 * GPU normalization is included in generation duration; conversion_seconds
 * applies only to CPU reference conversion. Estimates exclude driver overhead.
 * Optimization: one model weight set resident at a time, capture arenas survive
 * the switch, and repeated AV requests reuse weights without reloading target.
 *
 * Corpus consultation, header design and implementation, 2026-10-20: both local
 * servers queried separately in each phase and the cited rules read in full.
 * cpp-guidelines I.1 (#ri-explicit), R.1 (#rr-raii): explicit residency,
 * completion, ownership and failure contracts. cpp-perf-guidelines GPU.1,
 * GPU.6, GPU.7, GPU.9: keep tensor intermediates/decisions on device, batch
 * dispatch, decouple observation and separate persistent from transient arenas.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Related GPU.6/7/9 entries describe dispatch, staging and storage.
 */
#include "experiment/run.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "gpu/device.h"
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
  std::unique_ptr<inference::ResidentModel> verbalizer;
  measurement::PhaseMeasurements measurements;
  std::uint64_t execution = 0;
  std::size_t verbalizer_host_bytes = 0;
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
  std::size_t available_host(std::size_t pending) const {
    const auto retained = record ? record->retained_bytes() : 0;
    const auto used =
        add(add(retained, pending),
            add(inputs.metadata_host_bytes, verbalizer_host_bytes));
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
                                role == nla::ModelRole::target
                                    ? inputs.target_mode
                                    : inputs.verbalizer_mode);
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
    const auto used = retained_device_bytes();
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
            target ? limits.max_device_bytes : available_device(),
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
void Run::close() {
  impl_->verbalizer.reset();
  impl_->verbalizer_host_bytes = 0;
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
  const auto host = p.available_host(retained);
  auto started = Clock::now();
  publication.phase(WorkPhase::loading_target);
  auto loaded = p.load(nla::ModelRole::target, host);
  const auto working = loaded.host_working_bytes;
  if (working >= host) {
    throw std::length_error("no host allowance for target execution");
  }
  auto resident = inference::load_resident_model(
      p.device, std::move(loaded.checkpoint),
      {p.inputs.target_mode, p.limits.max_device_bytes,
       std::min(upload_allowance, host - working)});
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
  resident.reset();
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
  auto host = p.available_host(add(pending, conversion_bytes));
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
    if (!p.verbalizer) {
      start = Clock::now();
      publication.phase(WorkPhase::loading_verbalizer);
      auto loaded = p.load(nla::ModelRole::verbalizer, host);
      if (loaded.host_working_bytes >= host) {
        throw std::length_error("no host allowance for AV upload");
      }
      p.verbalizer = inference::load_resident_model(
          p.device, std::move(loaded.checkpoint),
          {p.inputs.verbalizer_mode, p.available_device(),
           std::min(upload_allowance, host - loaded.host_working_bytes)});
      attempt.cost.load_seconds = elapsed(start);
      accumulate(p.measurements.verbalizer_load_seconds,
                 *attempt.cost.load_seconds);
      p.verbalizer_host_bytes = loaded.host_working_bytes;
      host -= loaded.host_working_bytes;
      p.phase = Phase::verbalizer_ready;
    }
    attempt.stage = AttemptStage::preparation;
    const auto execution = p.next_id();
    start = Clock::now();
    {
      auto plan = inference::prepare_cached_forward(
          *p.verbalizer, p.options(capacity, execution, false, host),
          std::move(injection.embedding));
      if (p.inputs.selection_execution ==
          inference::sampling::Execution::device) {
        plan.prepare_sampling();
      }
      attempt.cost.prepare_seconds = elapsed(start);
      p.measurements.estimated_peak_device_bytes =
          std::max(p.measurements.estimated_peak_device_bytes,
                   plan.estimated_peak_device_bytes());
      const auto estimated_host =
          add(add(p.record->retained_bytes(), add(pending, conversion_bytes)),
              add(add(p.inputs.metadata_host_bytes, p.verbalizer_host_bytes),
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
              add(add(p.inputs.metadata_host_bytes, p.verbalizer_host_bytes),
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
    if (p.verbalizer && !p.verbalizer->usable()) {
      close();
    }
    throw;
  } catch (const std::exception& e) {
    attempt.outcome = AttemptOutcome::failed;
    attempt.failure_message = e.what();
  }
  if (p.verbalizer && !p.verbalizer->usable()) {
    p.verbalizer.reset();
    p.verbalizer_host_bytes = 0;
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
  auto host = p.available_host(pending);
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
  p.verbalizer.reset();
  p.verbalizer_host_bytes = 0;
  p.phase = Phase::retained;
  host = p.available_host(pending);
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
      auto loaded = p.load(nla::ModelRole::reconstructor, host);
      if (loaded.host_working_bytes >= host) {
        throw std::length_error("AR host budget");
      }
      auto ar = inference::load_resident_model(
          p.device, std::move(loaded.checkpoint),
          {p.inputs.target_mode, p.available_device(),
           std::min(upload_allowance, host - loaded.host_working_bytes)});
      check_stop();
      publication.phase(WorkPhase::reconstructing);
      const auto reconstruct = [&](std::span<const std::uint32_t> ids,
                                   std::size_t retained) {
        const auto execution = p.next_id();
        auto options = p.options(ids.size(), execution, false,
                                 host - loaded.host_working_bytes);
        options.max_device_bytes = p.available_device() - retained;
        auto plan = inference::prepare_cached_forward(*ar, options);
        return plan.reconstruct(ids, stop);
      };
      auto original = reconstruct(trial.original_ar_tokens, 0);
      check_stop();
      auto edited =
          reconstruct(trial.edited_ar_tokens, original.owner->size_bytes());
      check_stop();
      const auto width = p.inputs.pair->target_site().vector_width;
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
    auto loaded = p.load(nla::ModelRole::target, host);
    if (loaded.host_working_bytes >= host) {
      throw std::length_error("branch host budget");
    }
    const auto direction_bytes =
        direction.owner ? direction.owner->size_bytes() : 0;
    const auto device_budget = p.available_device() - direction_bytes;
    auto resident = inference::load_resident_model(
        p.device, std::move(loaded.checkpoint),
        {p.inputs.target_mode, device_budget,
         std::min(upload_allowance, host - loaded.host_working_bytes)});
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
      auto options = p.options(capacity, execution, false,
                               host - loaded.host_working_bytes);
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
