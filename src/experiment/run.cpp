/*
 * Sequential target/AV orchestration with retained records.
 * Axis R3: admission, one active experiment, model residency and cancellation.
 * Contract: Run owns its pair, model providers, text helpers and retained
 * record; it borrows one device that must outlive it. Public calls are
 * synchronous and serialized, not reentrant or concurrently polled. The device
 * is exclusively leased to this Run by composition; multiple Run objects are
 * not GPU scheduling. Providers open verified host checkpoints on demand, never
 * upload or cache GPU weights. LoadedModel source identity must describe the
 * returned checkpoint's verified backing archive; validate it against Pair
 * before device preparation. A pair's source metadata may be prepared
 * separately without retaining all model payloads. The host budget includes the
 * active loader's reported working allowance; mapped file address space and OS
 * page cache are reported separately.
 *
 * Algorithm: docs/architecture/logical-overview.md, "1 Establish the run",
 * "2 Generate and capture", "3 Retain the experiment and change phases", and
 * "4 Interpret selected positions". Let P be prompt length, G target cap,
 * Q the fixed AV prompt length and A an interpretation's output cap.
 * 1. Before target upload, format/encode P tokens, validate settings and admit
 *    C=P+G capacity (including final emitted-token consumption), all capture
 *    bytes/indexes, decoded text bounds and transient/readback budgets. Load
 * and validate target provenance and the pair's capture site.
 * Overflow/unsupported inputs fail before generation; never shorten a request
 * to fit silently.
 * 2. Load one resident target weight owner, prepare one sequence plan, and call
 *    generate with the pair's dynamic observation site. Retain only completed
 *    captures on cancellation/failure; decode/classify and move into Record.
 * 3. Finish/drain all dependent work; destroy sequence KV/scratch/staging, then
 *    resident target weights and host checkpoint. Publish retained state only
 *    after this release. An allocation failure cannot erase an already
 * committed record; before one can be materialized, the call may throw without
 * a result.
 * 4. interpret(key,settings) validates the retained row and reserves an
 * attempt's record/temporary space. Normalize only that row through Pair. On
 * the first request, open/validate/upload AV after target release; keep its
 * weights resident between requests. No target rerun or target+AV GPU overlap
 * occurs.
 * 5. Admit Q+A capacity and a fresh cache/sequence ID. Install the owned
 * injection during preparation. Generate using the exact AV prompt, selected
 * sampler and seed; consume the final emitted token before sealing as in target
 * mode. No target observation hooks are requested on the AV. Decode output,
 * extract tags, append a variant, and release sequence resources while
 * retaining AV.
 * 6. A canceled request records partial output and returns to AV-ready after
 *    cleanup. A load failure returns to retained so a later request may retry.
 *    A device failure destroys the poisoned resident owner and closes this Run;
 *    in-place device recovery is outside this slice. Record data remains
 * available, with no automatic target rerun. close releases AV resources but
 * keeps the record and forbids further generation/interpretation. Destructor
 * drains before release. Invalid calls (second target generation, unknown
 * capture, closed Run, rejected admission) throw before an attempt is accepted.
 * Recoverable execution errors append a failed attempt. bad_alloc may prevent
 * recording an error and propagates; existing record entries remain intact. A
 * stop requested before interpret starts does no GPU work and creates a
 * canceled attempt if its record slot is admitted. Preflight rejection leaves
 * empty state retryable. Once target execution starts, failure to materialize
 * any record closes the Run; it cannot silently rerun. Execution/sequence IDs
 * are monotonic within Run, separate from RunId and VariantId. Reject counter
 * overflow; never reuse an ID after a failed attempt.
 *
 * Baseline: one target generation per Run; one row per interpretation. Callers
 * explore neighboring rows by successive calls, obtaining separate variants.
 * Explicit settings/seed are always recorded; initial exploration uses
 * temperature 1, while deterministic numerical fixtures use greedy selection.
 * No confidence score, AR, HTTP server or browser behavior belongs to this
 * slice.
 *
 * Cost: peak device arena bytes = max(target weights+target sequence,
 * AV weights+AV sequence), not their sum. Per-model KV is 2*L*C*Hkv*d*b;
 * L=36,Hkv=8,d=128,b=2 gives 147456*C bytes. Account separately for scratch,
 * weight upload staging, injection staging, readback and driver headroom.
 * Host peak includes retained captures 8192*(P+G), record/index/text
 * allowances, one pending variant, active loader, tokenizers, normalized row
 * and staging. Host estimates cover admitted payload/staging and supplied
 * loader/text metadata; recipe/kernel metadata, allocator overhead and mapped
 * address space are excluded. These are allocation estimates, not measured
 * process/device peaks. Optimization: retain AV weights once; fresh contiguous
 * KV per variant keeps isolation simple. Allocation pooling remains deferred,
 * as in target generation. Timings separate loading, target generation,
 * release/switch, AV generation and decoding; GPU completion bounds compute
 * time. No performance claim is made. Corpus consultation, implementation,
 * 2026-10-19 (full rules read): cpp-guidelines R.1 (#rr-raii): sequence dies
 * before resident owner/device. cpp-guidelines I.7 (#ri-post): retained state
 * means target GPU resources released. cpp-perf-guidelines GPU.9: separate
 * persistent weights from transient sequence arenas; source
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines GPU.1: upload each model per phase and one row per
 * attempt; source guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "experiment/run.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <stdexcept>

#include "gpu/device.h"
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
  for (auto id : ids) {
    const std::array one{id};
    auto part = tokenizer.decode(one);
    if (part.tokens.size() != 1 ||
        part.bytes.size() > limit - result.bytes.size()) {
      throw std::length_error("decoded byte budget");
    }
    const auto begin = result.bytes.size();
    result.bytes += part.bytes;
    result.tokens.push_back(
        {id, begin, result.bytes.size(), part.tokens.at(0).mapped});
  }
  return result;
}
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
    bytes = add(bytes, multiply(tokens, capture_width));
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
                                        : inputs.verbalizer)(allowance);
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
            limits.max_device_bytes,
            host};
  }
};
Run::Run(gpu::Device& device, RunId id, RunInputs inputs, RunLimits limits)
    : impl_(std::make_unique<Impl>(device, id, std::move(inputs), limits)) {
  const auto& in = impl_->inputs;
  if (!in.pair || !in.target || !in.verbalizer || !in.target_tokenizer ||
      !in.verbalizer_tokenizer || !in.target_protocol || !in.selector ||
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
void Run::generate(const text::Prompt& prompt,
                   inference::GenerationSettings settings,
                   const std::stop_token& stop) {
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
  // Preflight failures above remain retryable. Once sequence execution begins,
  // unexpected failure closes the run rather than silently repeating a target.
  try {
    auto plan = inference::prepare_cached_forward(
        *resident, p.options(capacity, execution, true, host - working));
    p.measurements.estimated_peak_device_bytes =
        plan.estimated_peak_device_bytes();
    p.measurements.estimated_peak_host_bytes =
        add(add(p.inputs.metadata_host_bytes, retained),
            add(working, std::max(plan.estimated_peak_host_bytes(),
                                  std::min(upload_allowance, host - working))));
    started = Clock::now();
    auto result = inference::generate(plan, prepared, settings,
                                      p.inputs.selector, execution, stop);
    p.measurements.target_generate_seconds = elapsed(started);
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
    started = Clock::now();  // Include sequence teardown and resident release
                             // in this phase.
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
}
VariantId Run::interpret(CaptureKey key, inference::GenerationSettings settings,
                         const std::stop_token& stop) {
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
  const auto conversion_bytes = multiply(
      row->site.vector_width,
      sizeof(float) + 2 * tensors::element_bytes(
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
    attempt.cost.conversion_seconds = elapsed(start);
    attempt.injection = InjectionProvenance{
        injection.embedding.position, injection.source_l2_norm, injection.scale,
        injection.embedding.storage_type, p.inputs.pair->identity().convention};
    attempt.stage = AttemptStage::loading;
    if (!p.verbalizer) {
      start = Clock::now();
      auto loaded = p.load(nla::ModelRole::verbalizer, host);
      if (loaded.host_working_bytes >= host) {
        throw std::length_error("no host allowance for AV upload");
      }
      p.verbalizer = inference::load_resident_model(
          p.device, std::move(loaded.checkpoint),
          {p.inputs.verbalizer_mode, p.limits.max_device_bytes,
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
      attempt.generation = inference::generate(
          plan, prompt, settings, p.inputs.selector, execution, stop);
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
  return p.record->append(std::move(attempt));
}
}  // namespace dolittle::experiment
