/*
 * Steering retention, R4: validate source sample bytes, exact replay prefix,
 * independent branch results and matched settings before appending an immutable
 * bounded trial. Existing captures/variants remain untouched. Byte accounting
 * includes descriptions, AR IDs, branch token/byte/segment capacities and
 * source identities. replay_prefix keeps the full original prompt or prefix
 * through p and rejects generated stop IDs. O(prefix+branch tokens) host
 * validation, no device work or raw-vector copies. Overlay: steering.md, Exact
 * replay. Implementation corpora queried/read 2026-10-20: Core I.7 (#ri-post),
 * explicit retained invariants; GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md), keep numerical values with
 * device owners. Axis R4: immutable target capture index and bounded
 * interpretation history. Contract: a record owns target IDs, decoded byte
 * mapping, classifications, capture leases and provenance. It admits bounded
 * variants and validates every count/extent/site before publishing. Capture
 * lookup returns only a completed consumed-token row. Device must outlive
 * device leases, including borrowed row copies held beyond the record. Record
 * contains no execution queue.
 *
 * Algorithm: logical-overview.md retain/interpret; execution-boundary.md H1/H4.
 * Validate generation provenance and emitted/consumed counts; graceful partial
 * prefill may leave a prompt suffix but never a pending generated token.
 * Validate each site and payload, index (site, absolute consumed position) to
 * its row offset, sort and reject duplicates. Host reference lookup returns a
 * span; live lookup returns an immutable subregion sharing its device owner.
 * Append a variant only after validating its source key, settings, normalized
 * injection provenance when available, generation outcome and text mapping.
 * Failed GPU normalization may have no valid injection provenance. Raw target
 * data and previous variants remain unchanged by rejected appends.
 * Cost: O(captured positions) index, O(log positions) lookup, bounded variant
 * storage. retained_bytes accounts host payload/capacity; Run separately admits
 * the retained device arenas when loading AV weights. Optimization: no live
 * activation copies or CPU normalization during lookup/model switching.
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
#include "experiment/record.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

#include "bounds.h"
#include "tensors/tensor.h"

namespace dolittle::experiment {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::length_error("record size overflow");
  }
  return a + b;
}
std::size_t multiply(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::length_error("record extent overflow");
  }
  return a * b;
}
template <class T>
std::size_t allocation(const std::vector<T>& v) {
  return multiply(v.capacity(), sizeof(T));
}
std::size_t text_bytes(const std::string& s) { return add(s.capacity(), 1); }
auto key(CaptureKey k) {
  return std::tuple{k.site.block.value, k.site.boundary, k.position.value};
}
bool same_site(models::ObservationSite a, models::ObservationSite b) {
  return key({a.point, {0}}) == key({b.point, {0}}) &&
         a.vector_width == b.vector_width && a.storage_type == b.storage_type;
}
std::size_t source_bytes(const tensors::formats::SourceIdentity& s) {
  return add(add(text_bytes(s.repository), text_bytes(s.revision)),
             text_bytes(s.artifact_digest));
}
std::size_t decoded_bytes(const tokenizers::DecodedBytes& d) {
  return add(text_bytes(d.bytes), allocation(d.tokens));
}
void validate_decoded(const tokenizers::DecodedBytes& decoded,
                      std::span<const std::uint32_t> ids) {
  if (decoded.tokens.size() != ids.size()) {
    throw std::invalid_argument("decoded token count mismatch");
  }
  std::size_t end = 0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto& range = decoded.tokens.at(i);
    if (range.id != dolittle::at(ids, i) || range.begin != end ||
        range.end < range.begin || range.end > decoded.bytes.size() ||
        (!range.mapped && range.begin != range.end)) {
      throw std::invalid_argument("invalid token byte mapping");
    }
    end = range.end;
  }
  if (end != decoded.bytes.size()) {
    throw std::invalid_argument("unmapped decoded bytes");
  }
}
void validate_generation(const inference::GenerationResult& g) {
  if (!g.prompt_tokens || g.prompt_tokens > g.tokens.size() ||
      g.consumed_tokens > g.tokens.size() ||
      g.tokens.size() - g.prompt_tokens > g.settings.max_new_tokens ||
      g.checkpoint_identity.empty() || g.tokenizer_identity.empty() ||
      g.protocol_identity.empty() || g.stop_ids.empty()) {
    throw std::invalid_argument("invalid generation provenance/counts");
  }
  inference::sampling::validate_settings(g.settings.sampling);
  if (g.reason != inference::StopReason::execution_failure &&
      (g.reason != inference::StopReason::user_stop ||
       g.tokens.size() != g.prompt_tokens) &&
      g.consumed_tokens != g.tokens.size()) {
    throw std::invalid_argument("successful generation has unconsumed IDs");
  }
  if (g.consumed_tokens >= g.prompt_tokens &&
      g.tokens.size() - g.consumed_tokens > 1) {
    throw std::invalid_argument("more than one pending generated ID");
  }
  switch (g.reason) {
    case inference::StopReason::stop_token:
      if (g.tokens.size() == g.prompt_tokens ||
          std::find(g.stop_ids.begin(), g.stop_ids.end(), g.tokens.back()) ==
              g.stop_ids.end()) {
        throw std::invalid_argument("stop result lacks stop ID");
      }
      break;
    case inference::StopReason::token_limit:
      if (g.tokens.size() - g.prompt_tokens != g.settings.max_new_tokens) {
        throw std::invalid_argument("token limit count mismatch");
      }
      break;
    case inference::StopReason::user_stop:
    case inference::StopReason::execution_failure:
      break;
    default:
      throw std::invalid_argument("unknown stop reason");
  }
}
std::size_t generation_bytes(const inference::GenerationResult& g) {
  auto n = add(add(allocation(g.tokens), allocation(g.stop_ids)),
               allocation(g.captures));
  for (const auto& c : g.captures) {
    n = add(n, add(allocation(c.positions), allocation(c.bytes)));
  }
  for (const auto* s : {&g.checkpoint_identity, &g.tokenizer_identity,
                        &g.protocol_identity, &g.failure_message}) {
    n = add(n, text_bytes(*s));
  }
  return n;
}
bool same_settings(const inference::GenerationSettings& a,
                   const inference::GenerationSettings& b) {
  return a.seed == b.seed && a.max_new_tokens == b.max_new_tokens &&
         a.sampling.mode == b.sampling.mode &&
         a.sampling.temperature == b.sampling.temperature &&
         a.sampling.top_k == b.sampling.top_k &&
         a.sampling.top_p == b.sampling.top_p;
}
}  // namespace
struct Record::Impl {
  struct Row {
    CaptureKey capture;
    std::size_t chunk, offset, bytes;
  };
  TargetRecord target;
  RecordLimits limits;
  std::vector<Row> rows;
  std::vector<std::unique_ptr<Variant>> variants;
  std::vector<std::unique_ptr<Trial>> trials;
  std::size_t bytes{};
  Impl(TargetRecord t, RecordLimits l) : target(std::move(t)), limits(l) {}
};
Record::Record(TargetRecord target, RecordLimits limits)
    : impl_(std::make_unique<Impl>(std::move(target), limits)) {
  auto& p = *impl_;
  const auto& t = p.target;
  const auto& g = t.generation;
  validate_generation(g);
  validate_decoded(t.decoded, g.tokens);
  if (t.pair.convention.empty() || t.backend_identity.empty() ||
      !t.capture_site.vector_width) {
    throw std::invalid_argument("missing experiment provenance");
  }
  if (t.segments.size() != g.tokens.size() - g.prompt_tokens) {
    throw std::invalid_argument("segment count mismatch");
  }
  for (std::size_t i = 0; i < t.segments.size(); ++i) {
    if (t.segments.at(i).position != g.prompt_tokens + i) {
      throw std::invalid_argument("segment position mismatch");
    }
  }
  auto n = add(sizeof(Impl), generation_bytes(g));
  n = add(n, decoded_bytes(t.decoded));
  n = add(n, allocation(t.segments));
  for (const auto* s : {&t.formatted_prompt, &t.backend_identity,
                        &t.pair.convention, &t.pair.metadata_sha256}) {
    n = add(n, text_bytes(*s));
  }
  n = add(n, add(source_bytes(t.pair.target), source_bytes(t.pair.verbalizer)));
  if (t.pair.reconstructor) {
    n = add(n, source_bytes(*t.pair.reconstructor));
  }
  n = add(n, text_bytes(t.pair.reconstruction_metadata_sha256));
  std::size_t count = 0;
  for (const auto& c : g.captures) {
    count = add(count, c.positions.size());
  }
  n = add(n, multiply(count, sizeof(Impl::Row)));
  n = add(n, multiply(limits.max_variants, sizeof(std::unique_ptr<Variant>)));
  if (n > limits.max_retained_bytes) {
    throw std::length_error("target record exceeds retained budget");
  }
  p.rows.reserve(count);
  p.variants.reserve(limits.max_variants);
  if (limits.max_trials >
      limits.max_retained_bytes / sizeof(std::unique_ptr<Trial>)) {
    throw std::length_error("trial slot budget");
  }
  p.trials.reserve(limits.max_trials);
  n = add(n, allocation(p.trials));
  n = add(n, multiply(p.rows.capacity() - count, sizeof(Impl::Row)));
  n = add(n, multiply(p.variants.capacity() - limits.max_variants,
                      sizeof(std::unique_ptr<Variant>)));
  if (n > limits.max_retained_bytes) {
    throw std::length_error("record capacity exceeds retained budget");
  }
  for (std::size_t i = 0; i < g.captures.size(); ++i) {
    const auto& c = g.captures.at(i);
    const auto row_bytes = multiply(
        c.site.vector_width, tensors::element_bytes(c.site.storage_type));
    if (!same_site(c.site, t.capture_site) ||
        (c.device
             ? (!c.bytes.empty() || !c.device->owner ||
                c.device->bytes < multiply(c.positions.size(), row_bytes))
             : c.bytes.size() != multiply(c.positions.size(), row_bytes))) {
      throw std::invalid_argument("capture shape/site mismatch");
    }
    for (std::size_t j = 0; j < c.positions.size(); ++j) {
      const auto position = c.positions.at(j);
      if (position.value >= g.consumed_tokens) {
        throw std::invalid_argument("capture was not consumed");
      }
      p.rows.push_back(
          {{c.site.point, position}, i, multiply(j, row_bytes), row_bytes});
    }
  }
  std::sort(p.rows.begin(), p.rows.end(), [](const auto& a, const auto& b) {
    return key(a.capture) < key(b.capture);
  });
  for (std::size_t i = 1; i < p.rows.size(); ++i) {
    if (key(p.rows.at(i - 1).capture) == key(p.rows.at(i).capture)) {
      throw std::invalid_argument("duplicate capture row");
    }
  }
  p.bytes = n;
}
Record::~Record() = default;
const TargetRecord& Record::target() const noexcept { return impl_->target; }
std::size_t Record::variant_count() const noexcept {
  return impl_->variants.size();
}
std::size_t Record::retained_bytes() const noexcept { return impl_->bytes; }
const Variant& Record::variant(VariantId id) const {
  if (!id.value || id.value > impl_->variants.size()) {
    throw std::invalid_argument("unknown variant ID");
  }
  return *impl_->variants.at(static_cast<std::size_t>(id.value - 1));
}
std::optional<nla::CaptureRow> Record::capture(CaptureKey wanted) const {
  const auto& p = *impl_;
  const auto it = std::lower_bound(
      p.rows.begin(), p.rows.end(), key(wanted),
      [](const auto& r, const auto& k) { return key(r.capture) < k; });
  if (it == p.rows.end() || key(it->capture) != key(wanted)) {
    return std::nullopt;
  }
  const auto& c = p.target.generation.captures.at(it->chunk);
  if (c.device) {
    auto view = *c.device;
    view.offset += it->offset;
    view.bytes = it->bytes;
    return nla::CaptureRow{c.site, wanted.position, {}, std::move(view)};
  }
  return nla::CaptureRow{c.site, wanted.position,
                         std::span(c.bytes).subspan(it->offset, it->bytes)};
}
VariantId Record::append(Interpretation attempt) {
  auto& p = *impl_;
  if (!capture(attempt.capture)) {
    throw std::invalid_argument("interpretation has no retained capture");
  }
  if (attempt.outcome != AttemptOutcome::completed &&
      attempt.outcome != AttemptOutcome::cancelled &&
      attempt.outcome != AttemptOutcome::failed) {
    throw std::invalid_argument("unknown interpretation outcome");
  }
  if (attempt.explanation.format != nla::ExplanationFormat::complete &&
      attempt.explanation.format != nla::ExplanationFormat::missing &&
      attempt.explanation.format != nla::ExplanationFormat::malformed) {
    throw std::invalid_argument("unknown explanation format");
  }
  inference::sampling::validate_settings(attempt.requested_settings.sampling);
  if (p.variants.size() >= p.limits.max_variants ||
      p.variants.size() == std::numeric_limits<std::uint64_t>::max()) {
    throw std::length_error("variant count budget");
  }
  auto bytes = add(sizeof(Variant), add(decoded_bytes(attempt.decoded),
                                        text_bytes(attempt.failure_message)));
  if (attempt.injection) {
    const auto& i = *attempt.injection;
    if (!std::isfinite(i.source_l2_norm) || i.source_l2_norm <= 0 ||
        !std::isfinite(i.scale) || i.scale <= 0 ||
        i.conversion_identity != p.target.pair.convention) {
      throw std::invalid_argument("invalid injection provenance");
    }
    bytes = add(bytes, text_bytes(i.conversion_identity));
  }
  if (attempt.generation) {
    const auto& g = *attempt.generation;
    validate_generation(g);
    const bool decoding_failed = attempt.outcome == AttemptOutcome::failed &&
                                 attempt.stage == AttemptStage::decoding;
    if (!decoding_failed) {
      validate_decoded(attempt.decoded,
                       std::span(g.tokens).subspan(g.prompt_tokens));
    }
    if ((!attempt.injection &&
         g.reason != inference::StopReason::execution_failure) ||
        (attempt.injection &&
         attempt.injection->prompt_position.value >= g.prompt_tokens) ||
        !g.captures.empty() ||
        !same_settings(attempt.requested_settings, g.settings)) {
      throw std::invalid_argument("AV result provenance mismatch");
    }
    const auto expected = g.reason == inference::StopReason::execution_failure
                              ? AttemptOutcome::failed
                              : (g.reason == inference::StopReason::user_stop
                                     ? AttemptOutcome::cancelled
                                     : AttemptOutcome::completed);
    if (attempt.outcome != expected && !decoding_failed) {
      throw std::invalid_argument("attempt outcome contradicts generation");
    }
    bytes = add(bytes, generation_bytes(g));
  } else if (attempt.outcome == AttemptOutcome::completed ||
             !attempt.decoded.bytes.empty() ||
             !attempt.decoded.tokens.empty() || attempt.explanation.content) {
    throw std::invalid_argument("attempt without generation has output");
  }
  if (attempt.explanation.format == nla::ExplanationFormat::complete) {
    if (!attempt.explanation.content ||
        attempt.explanation.content->begin > attempt.explanation.content->end ||
        attempt.explanation.content->end > attempt.decoded.bytes.size()) {
      throw std::invalid_argument("explanation range outside output");
    }
  } else if (attempt.explanation.content) {
    throw std::invalid_argument("incomplete tags have a range");
  }
  const auto total = add(p.bytes, bytes);
  if (total > p.limits.max_retained_bytes) {
    throw std::length_error("variant retained byte budget");
  }
  const VariantId id{p.variants.size() + 1};
  auto stored = std::make_unique<Variant>(Variant{id, std::move(attempt)});
  p.variants.push_back(std::move(stored));
  p.bytes = total;
  return id;
}
std::size_t replay_prefix(const TargetRecord& target, CaptureKey capture) {
  const auto& g = target.generation;
  if (key({target.capture_site.point, capture.position}) != key(capture) ||
      !g.prompt_tokens || g.prompt_tokens > g.tokens.size() ||
      capture.position.value >= g.consumed_tokens ||
      g.consumed_tokens > g.tokens.size()) {
    throw std::invalid_argument("invalid steering capture position");
  }
  const auto end = std::max(g.prompt_tokens, capture.position.value + 1);
  for (const auto id :
       std::span(g.tokens).subspan(g.prompt_tokens, end - g.prompt_tokens)) {
    if (std::find(g.stop_ids.begin(), g.stop_ids.end(), id) !=
        g.stop_ids.end()) {
      throw std::invalid_argument(
          "cannot branch beyond a generated stop token");
    }
  }
  return end;
}
std::size_t Record::trial_count() const noexcept {
  return impl_->trials.size();
}
const Trial& Record::trial(TrialId id) const {
  if (!id.value || id.value > impl_->trials.size()) {
    throw std::invalid_argument("unknown steering trial");
  }
  return *impl_->trials.at(static_cast<std::size_t>(id.value - 1));
}
TrialId Record::append(SteeringTrial attempt) {
  auto& p = *impl_;
  const auto& sample = variant(attempt.variant).interpretation;
  if (sample.outcome != AttemptOutcome::completed ||
      sample.explanation.format != nla::ExplanationFormat::complete ||
      !sample.explanation.content ||
      key(sample.capture) != key(attempt.capture) ||
      !capture(attempt.capture) ||
      attempt.prefix_tokens != replay_prefix(p.target, attempt.capture) ||
      attempt.original_prompt_tokens != p.target.generation.prompt_tokens ||
      !std::isfinite(attempt.strength) || attempt.strength < 0 ||
      attempt.strength > 2 || attempt.edited.empty() ||
      attempt.edited.size() > nla::max_explanation_bytes) {
    throw std::invalid_argument("invalid steering provenance");
  }
  if (!attempt.reconstructor || !p.target.pair.reconstructor ||
      attempt.reconstructor->repository !=
          p.target.pair.reconstructor->repository ||
      attempt.reconstructor->revision !=
          p.target.pair.reconstructor->revision ||
      attempt.reconstructor->artifact_digest !=
          p.target.pair.reconstructor->artifact_digest ||
      attempt.original_ar_tokens.empty() || attempt.edited_ar_tokens.empty() ||
      !attempt.settings.max_new_tokens) {
    throw std::invalid_argument(
        "missing or mismatched reconstruction provenance");
  }
  const auto range = *sample.explanation.content;
  if (attempt.original !=
          sample.decoded.bytes.substr(range.begin, range.end - range.begin) ||
      (attempt.strength > 0 && attempt.original == attempt.edited) ||
      attempt.original_ar_tokens.size() > nla::max_reconstruction_tokens ||
      attempt.edited_ar_tokens.size() > nla::max_reconstruction_tokens ||
      (attempt.outcome != AttemptOutcome::completed &&
       attempt.outcome != AttemptOutcome::cancelled &&
       attempt.outcome != AttemptOutcome::failed)) {
    throw std::invalid_argument("invalid steering description or outcome");
  }
  inference::sampling::validate_settings(attempt.settings.sampling);
  if (p.trials.size() >= p.limits.max_trials) {
    throw std::length_error("steering trial limit");
  }
  auto bytes = add(sizeof(Trial), add(text_bytes(attempt.original),
                                      text_bytes(attempt.edited)));
  bytes = add(bytes, add(allocation(attempt.original_ar_tokens),
                         allocation(attempt.edited_ar_tokens)));
  bytes = add(bytes, add(text_bytes(attempt.numerical_identity),
                         text_bytes(attempt.message)));
  if (attempt.reconstructor) {
    bytes = add(bytes, source_bytes(*attempt.reconstructor));
  }
  for (const auto* branch : {&attempt.baseline, &attempt.steered}) {
    bytes = add(bytes, add(decoded_bytes(branch->decoded),
                           allocation(branch->segments)));
    if (!branch->generation) {
      if (attempt.outcome == AttemptOutcome::completed ||
          !branch->decoded.tokens.empty() || !branch->segments.empty()) {
        throw std::invalid_argument("missing steering branch result");
      }
      continue;
    }
    const auto& g = *branch->generation;
    validate_generation(g);
    if (g.prompt_tokens != attempt.prefix_tokens || !g.captures.empty() ||
        !same_settings(g.settings, attempt.settings) ||
        g.checkpoint_identity != p.target.generation.checkpoint_identity ||
        g.tokenizer_identity != p.target.generation.tokenizer_identity ||
        g.protocol_identity != p.target.generation.protocol_identity ||
        g.thinking != p.target.generation.thinking ||
        g.stop_ids != p.target.generation.stop_ids ||
        g.numerical_mode.activation_storage !=
            p.target.generation.numerical_mode.activation_storage ||
        g.numerical_mode.reduction_accumulation !=
            p.target.generation.numerical_mode.reduction_accumulation ||
        !std::equal(
            g.tokens.begin(),
            g.tokens.begin() + static_cast<std::ptrdiff_t>(g.prompt_tokens),
            p.target.generation.tokens.begin())) {
      throw std::invalid_argument(
          "steering branch changed fixed input or settings");
    }
    if (attempt.outcome == AttemptOutcome::completed &&
        (g.reason == inference::StopReason::user_stop ||
         g.reason == inference::StopReason::execution_failure)) {
      throw std::invalid_argument("incomplete steering branch");
    }
    if (attempt.outcome != AttemptOutcome::failed ||
        !branch->decoded.tokens.empty()) {
      validate_decoded(branch->decoded,
                       std::span(g.tokens).subspan(g.prompt_tokens));
      if (branch->segments.size() != g.tokens.size() - g.prompt_tokens) {
        throw std::invalid_argument("steering segment count");
      }
      for (std::size_t i = 0; i < branch->segments.size(); ++i) {
        if (branch->segments.at(i).position != g.prompt_tokens + i) {
          throw std::invalid_argument("steering segment position");
        }
      }
    }
    bytes = add(bytes, generation_bytes(g));
  }
  const auto total = add(p.bytes, bytes);
  if (total > p.limits.max_retained_bytes) {
    throw std::length_error("steering retained budget");
  }
  const TrialId id{p.trials.size() + 1};
  auto stored = std::make_unique<Trial>(Trial{id, std::move(attempt)});
  p.trials.push_back(std::move(stored));
  p.bytes = total;
  return id;
}

}  // namespace dolittle::experiment
