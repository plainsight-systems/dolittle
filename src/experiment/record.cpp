/*
 * Owned in-memory experiment records and atomic variant insertion.
 * Axis R4: owned target transcript/captures and append-only interpretation variants.
 * Contract: one Record owns one completed or interrupted target result, its exact
 * formatted prompt, decoded byte mapping and segmentation, plus bounded variants.
 * RunId identifies the experiment; CaptureKey identifies a consumed target token
 * at an advertised site; VariantId identifies an attempt within that run. None is
 * an AV token position. No device, plan, tokenizer or checkpoint lifetime is needed
 * after construction. This is in-memory retention, not persistence to disk.
 *
 * Algorithm: docs/architecture/logical-overview.md, "Capture alignment",
 * "3 Retain the experiment and change phases", "4 Interpret selected positions":
 * 1. Move the generation's exact tokens, completed raw capture chunks and
 *    provenance into the record. Validate prompt/consumed/emitted lengths and
 *    capture row extents. Each (site,position) is unique, position < consumed,
 *    and the site's storage/width agrees with the pair-validated capture_site
 *    supplied by Run. Preserve original bytes. Record checks consistency; Pair
 *    and Run establish checkpoint/semantic compatibility before creating it.
 * 2. Index rows by site and absolute consumed position, retaining chunk owners;
 *    do not concatenate/copy the full capture payload. Absent rows remain absent.
 *    An emitted but unconsumed token after failure has no fabricated vector.
 * 3. Receive the byte mapping and segmentation already prepared by Run's target
 *    tokenizer/protocol. Validate ID correspondence, byte extents and absolute
 *    segment positions. Record does not tokenize or classify. Exact IDs remain
 *    authoritative, including byte splits, unmapped IDs and unclassified segments.
 * 4. For each selected row, append a separately owned AV attempt with settings,
 *    seed, injection provenance, exact output IDs/bytes, stop/failure and tag status.
 *    Repeated attempts never overwrite an earlier description or rerun the target.
 *    Failed loading/conversion attempts can have no generated result. Completed
 *    tags do not erase cancellation, token-limit or execution-failure status.
 *    completed means the loop ended by stop ID or token cap, not that tags were
 *    complete. A generation result must agree with outcome, requested settings,
 *    injection provenance and decoded generated IDs; without one, outcome cannot
 *    be completed and there can be no decoded output or explanation range.
 * 5. Validate and admit the entire candidate before assigning its monotonic ID
 *    and publishing it. A rejected append leaves the record unchanged. A failed
 *    attempt that was admitted remains a record, not a successful interpretation.
 *
 * Ownership/failure: views borrow this stable noncopyable/nonmovable owner.
 * Capture views survive variant insertion; variant references are stable until
 * Record destruction (individually owned entries). Callers serialize all access.
 * Invalid provenance/alignment -> invalid_argument; byte/count overflow or budget
 * exhaustion -> length_error; bad_alloc propagates with no partial publication.
 * Limits include owned vector/string capacities and index/entry storage; allocator
 * and process overhead are excluded explicitly, never represented as a RAM cap.
 *
 * Cost: O(N + output bytes + R log R) construction for R indexed capture rows; payload
 * N*D*b for one site. Selected BF16 pair costs 8192*N capture bytes, plus metadata,
 * decoded text, pending candidate and generation staging admitted by Run.
 * Optimization: move payload owners and index their rows; no whole-trace copy on
 * model switch or variant append. Growth is bounded by count and byte budgets.
 * Corpus consultation, implementation, 2026-10-19 (full rules read):
 * cpp-guidelines R.1 (#rr-raii): records, not GPU plans, own retained payloads.
 * cpp-guidelines I.7 (#ri-post): atomic append and unavailable-row semantics.
 * cpp-perf-guidelines GPU.1: retain the existing completed readback; no replay
 * or extra device transfer to rediscover a selected capture.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "experiment/record.h"
#include "bounds.h"
#include "tensors/tensor.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

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
template <class T> std::size_t allocation(const std::vector<T> &v) {
    return multiply(v.capacity(), sizeof(T));
}
std::size_t text_bytes(const std::string &s) {
    return add(s.capacity(), 1);
}
auto key(CaptureKey k) {
    return std::tuple{k.site.block.value, k.site.boundary, k.position.value};
}
bool same_site(models::ObservationSite a, models::ObservationSite b) {
    return key({a.point, {0}}) == key({b.point, {0}}) && a.vector_width == b.vector_width &&
           a.storage_type == b.storage_type;
}
std::size_t source_bytes(const tensors::formats::SourceIdentity &s) {
    return add(add(text_bytes(s.repository), text_bytes(s.revision)), text_bytes(s.artifact_digest));
}
std::size_t decoded_bytes(const tokenizers::DecodedBytes &d) {
    return add(text_bytes(d.bytes), allocation(d.tokens));
}
void validate_decoded(const tokenizers::DecodedBytes &decoded, std::span<const std::uint32_t> ids) {
    if (decoded.tokens.size() != ids.size()) {
        throw std::invalid_argument("decoded token count mismatch");
    }
    std::size_t end = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto &range = decoded.tokens.at(i);
        if (range.id != dolittle::at(ids, i) || range.begin != end || range.end < range.begin ||
            range.end > decoded.bytes.size() || (!range.mapped && range.begin != range.end)) {
            throw std::invalid_argument("invalid token byte mapping");
        }
        end = range.end;
    }
    if (end != decoded.bytes.size()) {
        throw std::invalid_argument("unmapped decoded bytes");
    }
}
void validate_generation(const inference::GenerationResult &g) {
    if (!g.prompt_tokens || g.prompt_tokens > g.tokens.size() || g.consumed_tokens > g.tokens.size() ||
        g.tokens.size() - g.prompt_tokens > g.settings.max_new_tokens || g.checkpoint_identity.empty() ||
        g.tokenizer_identity.empty() || g.protocol_identity.empty() || g.stop_ids.empty()) {
        throw std::invalid_argument("invalid generation provenance/counts");
    }
    inference::sampling::validate_settings(g.settings.sampling);
    if (g.reason != inference::StopReason::execution_failure && g.consumed_tokens != g.tokens.size()) {
        throw std::invalid_argument("successful generation has unconsumed IDs");
    }
    if (g.consumed_tokens >= g.prompt_tokens && g.tokens.size() - g.consumed_tokens > 1) {
        throw std::invalid_argument("more than one pending generated ID");
    }
    switch (g.reason) {
    case inference::StopReason::stop_token:
        if (g.tokens.size() == g.prompt_tokens ||
            std::find(g.stop_ids.begin(), g.stop_ids.end(), g.tokens.back()) == g.stop_ids.end()) {
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
std::size_t generation_bytes(const inference::GenerationResult &g) {
    auto n = add(add(allocation(g.tokens), allocation(g.stop_ids)), allocation(g.captures));
    for (const auto &c : g.captures) {
        n = add(n, add(allocation(c.positions), allocation(c.bytes)));
    }
    for (const auto *s :
         {&g.checkpoint_identity, &g.tokenizer_identity, &g.protocol_identity, &g.failure_message}) {
        n = add(n, text_bytes(*s));
    }
    return n;
}
bool same_settings(const inference::GenerationSettings &a, const inference::GenerationSettings &b) {
    return a.seed == b.seed && a.max_new_tokens == b.max_new_tokens && a.sampling.mode == b.sampling.mode &&
           a.sampling.temperature == b.sampling.temperature && a.sampling.top_k == b.sampling.top_k &&
           a.sampling.top_p == b.sampling.top_p;
}
} // namespace
struct Record::Impl {
    struct Row {
        CaptureKey capture;
        std::size_t chunk, offset, bytes;
    };
    TargetRecord target;
    RecordLimits limits;
    std::vector<Row> rows;
    std::vector<std::unique_ptr<Variant>> variants;
    std::size_t bytes{};
    Impl(TargetRecord t, RecordLimits l) : target(std::move(t)), limits(l) {
    }
};
Record::Record(TargetRecord target, RecordLimits limits)
    : impl_(std::make_unique<Impl>(std::move(target), limits)) {
    auto &p = *impl_;
    const auto &t = p.target;
    const auto &g = t.generation;
    validate_generation(g);
    validate_decoded(t.decoded, g.tokens);
    if (t.pair.convention.empty() || t.backend_identity.empty() || !t.capture_site.vector_width) {
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
    for (const auto *s :
         {&t.formatted_prompt, &t.backend_identity, &t.pair.convention, &t.pair.metadata_sha256}) {
        n = add(n, text_bytes(*s));
    }
    n = add(n, add(source_bytes(t.pair.target), source_bytes(t.pair.verbalizer)));
    std::size_t count = 0;
    for (const auto &c : g.captures) {
        count = add(count, c.positions.size());
    }
    n = add(n, multiply(count, sizeof(Impl::Row)));
    n = add(n, multiply(limits.max_variants, sizeof(std::unique_ptr<Variant>)));
    if (n > limits.max_retained_bytes) {
        throw std::length_error("target record exceeds retained budget");
    }
    p.rows.reserve(count);
    p.variants.reserve(limits.max_variants);
    n = add(n, multiply(p.rows.capacity() - count, sizeof(Impl::Row)));
    n = add(n, multiply(p.variants.capacity() - limits.max_variants, sizeof(std::unique_ptr<Variant>)));
    if (n > limits.max_retained_bytes) {
        throw std::length_error("record capacity exceeds retained budget");
    }
    for (std::size_t i = 0; i < g.captures.size(); ++i) {
        const auto &c = g.captures.at(i);
        const auto row_bytes = multiply(c.site.vector_width, tensors::element_bytes(c.site.storage_type));
        if (!same_site(c.site, t.capture_site) || c.bytes.size() != multiply(c.positions.size(), row_bytes)) {
            throw std::invalid_argument("capture shape/site mismatch");
        }
        for (std::size_t j = 0; j < c.positions.size(); ++j) {
            const auto position = c.positions.at(j);
            if (position.value >= g.consumed_tokens) {
                throw std::invalid_argument("capture was not consumed");
            }
            p.rows.push_back({{c.site.point, position}, i, multiply(j, row_bytes), row_bytes});
        }
    }
    std::sort(p.rows.begin(), p.rows.end(),
              [](const auto &a, const auto &b) { return key(a.capture) < key(b.capture); });
    for (std::size_t i = 1; i < p.rows.size(); ++i) {
        if (key(p.rows.at(i - 1).capture) == key(p.rows.at(i).capture)) {
            throw std::invalid_argument("duplicate capture row");
        }
    }
    p.bytes = n;
}
Record::~Record() = default;
const TargetRecord &Record::target() const noexcept {
    return impl_->target;
}
std::size_t Record::variant_count() const noexcept {
    return impl_->variants.size();
}
std::size_t Record::retained_bytes() const noexcept {
    return impl_->bytes;
}
const Variant &Record::variant(VariantId id) const {
    if (!id.value || id.value > impl_->variants.size()) {
        throw std::invalid_argument("unknown variant ID");
    }
    return *impl_->variants.at(static_cast<std::size_t>(id.value - 1));
}
std::optional<nla::CaptureRow> Record::capture(CaptureKey wanted) const {
    const auto &p = *impl_;
    const auto it = std::lower_bound(p.rows.begin(), p.rows.end(), key(wanted),
                                     [](const auto &r, const auto &k) { return key(r.capture) < k; });
    if (it == p.rows.end() || key(it->capture) != key(wanted)) {
        return std::nullopt;
    }
    const auto &c = p.target.generation.captures.at(it->chunk);
    return nla::CaptureRow{c.site, wanted.position, std::span(c.bytes).subspan(it->offset, it->bytes)};
}
VariantId Record::append(Interpretation attempt) {
    auto &p = *impl_;
    if (!capture(attempt.capture)) {
        throw std::invalid_argument("interpretation has no retained capture");
    }
    if (attempt.outcome != AttemptOutcome::completed && attempt.outcome != AttemptOutcome::cancelled &&
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
    auto bytes =
        add(sizeof(Variant), add(decoded_bytes(attempt.decoded), text_bytes(attempt.failure_message)));
    if (attempt.injection) {
        const auto &i = *attempt.injection;
        if (!std::isfinite(i.source_l2_norm) || i.source_l2_norm <= 0 || !std::isfinite(i.scale) ||
            i.scale <= 0 || i.conversion_identity != p.target.pair.convention) {
            throw std::invalid_argument("invalid injection provenance");
        }
        bytes = add(bytes, text_bytes(i.conversion_identity));
    }
    if (attempt.generation) {
        const auto &g = *attempt.generation;
        validate_generation(g);
        const bool decoding_failed =
            attempt.outcome == AttemptOutcome::failed && attempt.stage == AttemptStage::decoding;
        if (!decoding_failed) {
            validate_decoded(attempt.decoded, std::span(g.tokens).subspan(g.prompt_tokens));
        }
        if (!attempt.injection || attempt.injection->prompt_position.value >= g.prompt_tokens ||
            !g.captures.empty() || !same_settings(attempt.requested_settings, g.settings)) {
            throw std::invalid_argument("AV result provenance mismatch");
        }
        const auto expected =
            g.reason == inference::StopReason::execution_failure
                ? AttemptOutcome::failed
                : (g.reason == inference::StopReason::user_stop ? AttemptOutcome::cancelled
                                                                : AttemptOutcome::completed);
        if (attempt.outcome != expected && !decoding_failed) {
            throw std::invalid_argument("attempt outcome contradicts generation");
        }
        bytes = add(bytes, generation_bytes(g));
    } else if (attempt.outcome == AttemptOutcome::completed || !attempt.decoded.bytes.empty() ||
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
} // namespace dolittle::experiment
