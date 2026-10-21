/*
 * Steering extension: resolve a completed variant, publish its edit and replay
 * boundary, preflight both bounded streams, then execute AR and matched
 * branches on the worker. Events distinguish baseline/steered and retain trial
 * completion. Core I.1 (#ri-explicit): typed command/stream identities;
 * Performance MEM.9 (guidelines/memory/MEM.9-embedded-allocate-at-init.md):
 * reserve before execution. Both corpora queried/read for this implementation,
 * 2026-10-20.
 *
 * Live experiment session: IMPLEMENTATION, 2026-10-19.
 * Axes R3/R4: command admission, single-worker Run ownership and bounded
 * publication. HTTP framing and presentation layout belong to their own axes.
 * Contract: one Session owns one persistent joined worker, one device and at
 * most one current Run. Factories execute on that worker; Run/device teardown
 * happens there too. Request/read/stop calls may come from HTTP threads but
 * never expose Run, Record, capture bytes or device views. All public results
 * own their small values or share immutable presentation events. Session is
 * noncopyable/nonmovable. Destruction stops admission, requests cooperative
 * cancellation outside locks and joins before destroying referenced state.
 *
 * Algorithm: docs/architecture/live-application.md, "Algorithm and ownership"
 * and "Transport contract":
 * 1. Validate request identity, expected RunId, settings and bounded job
 * counts. Under one admission lock, deduplicate canonical typed payloads and
 * reserve command/receipt capacity atomically. Same ID/data returns the same
 * receipt; changed data conflicts. Reject before enqueue on capacity
 * exhaustion.
 * 2. The worker takes one command, releases the queue lock, and creates or uses
 *    Run. A new run replaces retained state only on explicit matching prior ID
 *    and no active/queued work. Assign monotonic IDs, rejecting overflow.
 * 3. Adapt RunObserver deltas into owned presentation events. Append once with
 *    a monotonic event sequence; decoded bytes remain exact. No raw vectors
 *    enter the journal. HTTP copies bounded shared event owners under a short
 *    lock, then encodes/sends outside it. Never call factories, Run, callbacks,
 *    stop callbacks or I/O while holding the admission/publication mutex.
 * 4. On target completion, inspect Record on the worker and publish actual
 *    available capture ranges only after target release. A pending selection
 *    is resolved then. Interpret each job with fresh settings/seed and KV;
 *    preserve request/job identity separately from the eventual VariantId.
 * 5. stop matches run AND active request, signals that operation's stop source,
 *    and cancels its remaining jobs. A stale stop cannot affect the next job
 *    batch. Cancellation during target generation also rejects waiting AV jobs
 *    only if no usable retained result exists; otherwise they remain queued.
 * 6. Keep the journal/receipts until explicit replacement. read(after) returns
 *    only a bounded contiguous page, advancing through events actually present.
 *    A stale run/future cursor fails explicitly. Slow clients neither block
 *    inference on sockets nor pin unbounded pages. An event allocation/budget
 *    failure uses reserved terminal event slots and requests graceful stop. A
 *    preallocated status records publication failure even if no event can be
 *    materialized. Never report a successful transcript with silently lost IDs.
 *
 * Bounds: queue commands/jobs/receipts, event count/payload, per-token decoded
 * bytes, page events/encoded bytes and active reader pins are explicit. Add
 * O(N+V*A+B+E) presentation memory to Run's budgets, plus W*R response scratch.
 * Charge vector/string capacities and shared owners; allocator overhead is
 * separately acknowledged. Check byte/count arithmetic before publication; each
 * temporary delta is bounded by the decoded-output and progress-chunk limits.
 * No journal eviction while a run is current; capacity rejects new requests.
 * Optimization: move delta payloads, share const event chunks, serialize only
 * requested pages, and keep one worker rather than one thread per request.
 * TokenDelta positions are absolute within its labeled stream; token byte
 * ranges index delta.bytes. Prompt tokens use unclassified segments. AV tags
 * are finalized from Record, never inferred from a partial browser string.
 *
 * Page events and their owners are borrowed for the page lifetime; callers
 * cannot retain copied event owners beyond that lease. Only page destruction
 * releases the accounted reader slot. Moved-from pages may only be destroyed
 * or assigned. Lease accounting survives Session through shared immutable
 * publication state.
 * Invalid arguments -> invalid_argument; bounded exhaustion -> length_error;
 * bad_alloc propagates before admission, or becomes explicit worker failure
 * afterward when reserved status storage permits. Expected request conflicts
 * use ReceiptCode. An uncaught worker failure marks unavailable and is retained
 * as status; it must not terminate the process or auto-replay a target.
 * Implementation corpus queried 2026-10-19 (full rules read): cpp-guidelines
 * ES.103 (#res-overflow), checked publication sums/products; I.7 (#ri-post),
 * explicit failed-publication status and cursor postconditions. CP.23
 * (#rconc-join), join before ownership ends; CP.41 (#rconc-create), persistent
 * worker; CP.22 (#rconc-unknown), callbacks outside locks; F.27
 * (#rf-shared_ptr), shared ownership only for immutable events crossing
 * reader/worker lifetimes. cpp-perf-guidelines COPY.7, move payloads and borrow
 * views instead of copying full histories;
 * guidelines/copy-move/COPY.7-eliminate-hidden-copies.md. GPU.1, publication
 * consumes completed host results without extra readback;
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Steady-state design, 2026-10-20:
 * Prepare stable event slots and token/text/segment capacities before each
 * stream. Publication fills the next unused slot and shares its existing
 * owner by aliasing; no slot is reused while readers may reference it.
 * Check count/byte budgets before publication, with explicit exhaustion and
 * no heap fallback. HTTP response allocations remain on reader threads.
 * Overlay: transformer-execution.md, "Device selection and allocation-free
 * generation". Core I.7 (#ri-post): borrowing, bounds and failure contracts.
 * Performance GPU.1: device-resident decisions; MEM.9: prepare storage before
 * steady state and check the invariant. Sources:
 * guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/memory/MEM.9-embedded-allocate-at-init.md.
 * Both corpora queried/read for header design and implementation, 2026-10-20.
 */
#include "experiment/session.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

#include "gpu/device.h"

namespace dolittle::experiment {
namespace {
constexpr std::size_t json_escape_expansion = 6;
using Command =
    std::variant<StartRequest, InterpretRequest, SteerRequest, StopRequest>;
constexpr std::size_t event_overhead = 4096;
constexpr std::size_t token_wire_bound = 256;
constexpr std::size_t progress_chunk = 64;
constexpr std::size_t max_request_id_bytes = 128;
constexpr std::size_t max_failure_message_bytes = 1024;
constexpr std::size_t event_reserve = 32;
bool same(inference::GenerationSettings a, inference::GenerationSettings b) {
  return a.seed == b.seed && a.max_new_tokens == b.max_new_tokens &&
         a.sampling.mode == b.sampling.mode &&
         a.sampling.temperature == b.sampling.temperature &&
         a.sampling.top_k == b.sampling.top_k &&
         a.sampling.top_p == b.sampling.top_p;
}
bool same(CaptureKey a, CaptureKey b) {
  return a.position.value == b.position.value &&
         a.site.block.value == b.site.block.value &&
         a.site.boundary == b.site.boundary;
}
bool same(const StartRequest& a, const StartRequest& b) {
  return a.replace.has_value() == b.replace.has_value() &&
         (!a.replace || a.replace->value == b.replace->value) &&
         a.prompt.system == b.prompt.system && a.prompt.user == b.prompt.user &&
         a.prompt.thinking == b.prompt.thinking && same(a.settings, b.settings);
}
bool same(const InterpretRequest& a, const InterpretRequest& b) {
  if (a.run.value != b.run.value || a.jobs.size() != b.jobs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.jobs.size(); ++i) {
    if (!same(a.jobs.at(i).capture, b.jobs.at(i).capture) ||
        !same(a.jobs.at(i).settings, b.jobs.at(i).settings)) {
      return false;
    }
  }
  return true;
}
bool same(const SteerRequest& a, const SteerRequest& b) {
  return a.run.value == b.run.value &&
         a.edit.variant.value == b.edit.variant.value &&
         a.edit.explanation == b.edit.explanation &&
         a.edit.strength == b.edit.strength &&
         same(a.edit.settings, b.edit.settings);
}
bool same(const StopRequest& a, const StopRequest& b) {
  return a.run.value == b.run.value &&
         a.active_request.value == b.active_request.value;
}
bool same(const Command& lhs, const Command& rhs) {
  return std::visit(
      [&](const auto& value) {
        const auto* other = std::get_if<std::decay_t<decltype(value)>>(&rhs);
        return other && same(value, *other);
      },
      lhs);
}
const RequestId& identity(const Command& command) {
  return std::visit(
      [](const auto& value) -> const RequestId& { return value.request; },
      command);
}
std::size_t jobs(const Command& command) {
  const auto* batch = std::get_if<InterpretRequest>(&command);
  return batch ? batch->jobs.size() : 1;
}
void validate(const Command& command, const RunLimits& limits,
              std::size_t max_jobs) {
  const auto& id = identity(command).value;
  if (id.empty() || id.size() > max_request_id_bytes ||
      !std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '-' || c == '_';
      })) {
    throw std::invalid_argument("invalid request ID");
  }
  std::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, StartRequest>) {
          inference::sampling::validate_settings(value.settings.sampling);
          if (value.prompt.user.size() > limits.max_decoded_bytes ||
              (value.prompt.system &&
               value.prompt.system->size() >
                   limits.max_decoded_bytes - value.prompt.user.size()) ||
              value.settings.max_new_tokens > limits.max_target_tokens) {
            throw std::length_error("target request limit");
          }
        } else if constexpr (std::is_same_v<T, InterpretRequest>) {
          if (value.jobs.empty() || value.jobs.size() > max_jobs) {
            throw std::length_error("interpretation batch limit");
          }
          for (const auto& job : value.jobs) {
            inference::sampling::validate_settings(job.settings.sampling);
            if (job.settings.max_new_tokens > limits.max_verbalizer_tokens) {
              throw std::length_error("interpretation token limit");
            }
          }
        } else if constexpr (std::is_same_v<T, SteerRequest>) {
          inference::sampling::validate_settings(value.edit.settings.sampling);
          if (!std::isfinite(value.edit.strength) || value.edit.strength < 0 ||
              value.edit.strength > 2 || value.edit.explanation.empty() ||
              value.edit.explanation.size() > nla::max_explanation_bytes ||
              !value.edit.settings.max_new_tokens ||
              value.edit.settings.max_new_tokens > limits.max_target_tokens) {
            throw std::invalid_argument("invalid steering edit/settings");
          }
        } else {
          if (value.active_request.value.empty() ||
              value.active_request.value.size() > max_request_id_bytes) {
            throw std::invalid_argument("invalid active request ID");
          }
        }
      },
      command);
}
std::size_t add(std::size_t lhs, std::size_t rhs) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::length_error("publication size overflow");
  }
  return lhs + rhs;
}
std::size_t multiply(std::size_t lhs, std::size_t rhs) {
  if (lhs && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::length_error("publication extent overflow");
  }
  return lhs * rhs;
}
std::size_t text_bound(const std::string& value) {
  return multiply(add(value.capacity(), 1), json_escape_expansion);
}
std::size_t source_bound(const tensors::formats::SourceIdentity& value) {
  return add(add(text_bound(value.repository), text_bound(value.revision)),
             text_bound(value.artifact_digest));
}
std::size_t event_bound(const EventBody& body) {
  return std::visit(
      [](const auto& value) -> std::size_t {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, TokenDelta>) {
          return add(event_overhead,
                     add(text_bound(value.decoded.bytes),
                         multiply(add(value.decoded.tokens.capacity(),
                                      value.segments.capacity()),
                                  token_wire_bound)));
        } else if constexpr (std::is_same_v<T, TargetAvailable>) {
          auto bytes = add(event_overhead, multiply(value.captures.capacity(),
                                                    token_wire_bound));
          bytes = add(bytes, add(source_bound(value.pair.target),
                                 source_bound(value.pair.verbalizer)));
          bytes = add(bytes, add(text_bound(value.pair.convention),
                                 text_bound(value.pair.metadata_sha256)));
          return add(bytes, text_bound(value.backend_identity));
        } else if constexpr (std::is_same_v<T, AttemptAvailable> ||
                             std::is_same_v<T, RequestFinished> ||
                             std::is_same_v<T, SteeringAvailable>) {
          return add(event_overhead, text_bound(value.message));
        } else if constexpr (std::is_same_v<T, SteeringStarted>) {
          return add(event_overhead,
                     add(text_bound(value.original), text_bound(value.edited)));
        } else {
          return event_overhead;
        }
      },
      body);
}
struct ReaderBudget {
  std::atomic<std::size_t> count{};
};
struct Work {
  Command command;
  RunId run;
};
}  // namespace
struct EventPage::Impl {
  std::shared_ptr<ReaderBudget> readers;
  RunId run{};
  SessionStatus status{};
  std::vector<std::shared_ptr<const PublishedEvent>> events;
  EventCursor next{};
  bool more{};
  explicit Impl(std::shared_ptr<ReaderBudget> budget)
      : readers(std::move(budget)) {}
  ~Impl() { --readers->count; }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
};
EventPage::EventPage(std::unique_ptr<Impl> p) : impl_(std::move(p)) {}
EventPage::~EventPage() = default;
EventPage::EventPage(EventPage&&) noexcept = default;
EventPage& EventPage::operator=(EventPage&&) noexcept = default;
RunId EventPage::run() const noexcept { return impl_->run; }
const SessionStatus& EventPage::status() const noexcept {
  return impl_->status;
}
std::span<const std::shared_ptr<const PublishedEvent>> EventPage::events()
    const noexcept {
  return impl_->events;
}
EventCursor EventPage::next() const noexcept { return impl_->next; }
bool EventPage::more() const noexcept { return impl_->more; }
EventPageView EventPage::view() const noexcept {
  return {impl_->run, std::cref(impl_->status), impl_->events, impl_->next,
          impl_->more};
}
struct Session::Impl final : RunObserver {
  SessionSetup setup;
  // This mutex guards status, commands, receipts, publication and stop source.
  mutable std::mutex mutex;
  std::condition_variable available;
  SessionStatus state{SessionPhase::idle, {}, {}, 0, {0}, false};
  std::deque<std::shared_ptr<const Work>> queue;
  std::map<std::string, std::shared_ptr<const Work>> receipts;
  std::vector<std::shared_ptr<const PublishedEvent>> journal;
  std::vector<std::size_t> event_sizes;
  std::shared_ptr<ReaderBudget> readers = std::make_shared<ReaderBudget>();
  std::size_t publication_bytes{};
  struct StreamEvents {
    std::vector<PublishedEvent> slots;
  };
  std::shared_ptr<StreamEvents> stream_events;
  std::size_t next_stream_event{}, reserved_stream_bytes{};
  std::uint64_t next_run{};
  bool closing{};
  bool worker_failed{};
  std::stop_source active_stop;
  // Only the worker touches these fields; it releases Run before device.
  std::unique_ptr<gpu::Device> device;
  std::unique_ptr<Run> run;
  RequestId request;
  std::size_t job_index{};
  bool observer_failed{};
  std::jthread worker;
  explicit Impl(SessionSetup config) : setup(std::move(config)) {
    const auto& l = setup.session;
    if (!setup.open_device || !setup.make_inputs || !l.max_commands ||
        !l.max_jobs || !l.max_receipts || l.max_events < 2 ||
        l.max_jobs > l.max_events - 2 ||
        l.max_publication_bytes < event_overhead ||
        l.max_page_bytes < event_overhead || !l.max_page_events ||
        !l.max_readers) {
      throw std::invalid_argument("invalid session limits");
    }
    journal.reserve(l.max_events);
    event_sizes.reserve(l.max_events);
    worker = std::jthread([this] { work(); });
  }
  ~Impl() override { shutdown(); }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  void shutdown() {
    auto stop = [this] {
      std::lock_guard lock(mutex);
      closing = true;
      return active_stop;
    }();
    stop.request_stop();
    available.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }
  void publication_failed() noexcept override {
    observer_failed = true;
    auto stop = [this] {
      std::lock_guard lock(mutex);
      state.publication_failed = true;
      return active_stop;
    }();
    stop.request_stop();
  }
  bool emit(EventBody body, bool terminal = false) noexcept {
    try {
      auto event = std::make_shared<PublishedEvent>(
          PublishedEvent{{0}, request, std::move(body)});
      return publish(std::move(event), terminal);
    } catch (...) {
      publication_failed();
      return false;
    }
  }
  bool publish(std::shared_ptr<PublishedEvent> event,
               bool terminal = false) noexcept {
    try {
      const auto bytes =
          add(event_bound(event->body), text_bound(event->request.value));
      const auto& l = setup.session;
      bool accepted = false;
      {
        std::lock_guard lock(mutex);
        const auto reserve = terminal ? 0 : l.max_jobs + 2;
        if (journal.size() < l.max_events - reserve &&
            bytes <= l.max_publication_bytes - publication_bytes -
                         reserved_stream_bytes &&
            bytes <= l.max_page_bytes - event_overhead &&
            (!state.publication_failed || terminal)) {
          event->sequence = {static_cast<std::uint64_t>(journal.size() + 1)};
          journal.push_back(std::move(event));
          event_sizes.push_back(bytes);
          publication_bytes += bytes;
          state.latest = {static_cast<std::uint64_t>(journal.size())};
          accepted = true;
        }
      }
      if (!accepted) {
        publication_failed();
      }
      return accepted;
    } catch (...) {
      publication_failed();
      return false;
    }
  }
  bool phase(WorkPhase phase_value) noexcept override {
    {
      std::lock_guard lock(mutex);
      switch (phase_value) {
        case WorkPhase::loading_target:
          state.phase = SessionPhase::loading_target;
          break;
        case WorkPhase::generating_target:
          state.phase = SessionPhase::generating_target;
          break;
        case WorkPhase::releasing_target:
          state.phase = SessionPhase::releasing_target;
          break;
        case WorkPhase::target_retained:
          state.phase = SessionPhase::retained;
          break;
        case WorkPhase::loading_verbalizer:
          state.phase = SessionPhase::loading_verbalizer;
          break;
        case WorkPhase::generating_verbalizer:
          state.phase = SessionPhase::interpreting;
          break;
        case WorkPhase::verbalizer_ready:
          state.phase = SessionPhase::retained;
          break;
        case WorkPhase::loading_reconstructor:
          state.phase = SessionPhase::loading_reconstructor;
          break;
        case WorkPhase::reconstructing:
          state.phase = SessionPhase::reconstructing;
          break;
        case WorkPhase::generating_baseline:
          state.phase = SessionPhase::generating_baseline;
          break;
        case WorkPhase::generating_steered:
          state.phase = SessionPhase::generating_steered;
          break;
        case WorkPhase::closed:
          state.phase = SessionPhase::unavailable;
          break;
      }
    }
    return emit(phase_value);
  }
  bool comparison_started(const SteeringTrial& trial) noexcept override {
    try {
      return emit(SteeringStarted{trial.variant, trial.capture, trial.original,
                                  trial.edited, trial.strength, trial.settings,
                                  trial.prefix_tokens,
                                  trial.original_prompt_tokens});
    } catch (...) {
      publication_failed();
      return false;
    }
  }
  void prepare_comparison(StreamBounds bounds) override {
    // Only this worker publishes. Preflight both streams before loading AR;
    // per-stream reservations below then consume this same remaining capacity.
    const auto prompt_events =
        (bounds.prompt_tokens + progress_chunk - 1) / progress_chunk;
    const auto count = add(
        add(prompt_events, (bounds.prompt_tokens + bounds.prefill_chunk - 1) /
                               bounds.prefill_chunk),
        add(multiply(bounds.generated_tokens, 2), 1));
    const auto token_bytes =
        bounds.token_bytes ? bounds.token_bytes : bounds.decoded_bytes;
    const auto event_count = add(multiply(2, count), event_reserve);
    auto pool =
        multiply(count, sizeof(PublishedEvent) + request.value.size() + 1);
    pool =
        add(pool, multiply(add(multiply(prompt_events, progress_chunk),
                               count - prompt_events),
                           add(token_bytes, sizeof(tokenizers::TokenBytes) +
                                                sizeof(text::TokenSegment))));
    auto bytes = add(
        multiply(2, pool),
        multiply(event_count, add(event_overhead, text_bound(request.value))));
    bytes = add(
        bytes,
        multiply(
            2, add(multiply(bounds.decoded_bytes, json_escape_expansion),
                   multiply(add(bounds.prompt_tokens, bounds.generated_tokens),
                            2 * token_wire_bound))));
    bytes = add(
        bytes, multiply(2 * nla::max_explanation_bytes, json_escape_expansion));
    std::lock_guard lock(mutex);
    if (event_count > setup.session.max_events - journal.size() ||
        bytes > setup.session.max_publication_bytes - reserved_stream_bytes -
                    publication_bytes) {
      throw std::length_error(
          "Comparison exceeds remaining session capacity; start a new run or "
          "reduce the continuation budget.");
    }
  }
  bool prepare_stream(StreamBounds bounds) noexcept override {
    try {
      const auto prompt_events =
          (bounds.prompt_tokens + progress_chunk - 1) / progress_chunk;
      const auto count = add(
          add(prompt_events, (bounds.prompt_tokens + bounds.prefill_chunk - 1) /
                                 bounds.prefill_chunk),
          add(multiply(bounds.generated_tokens, 2), 1));
      const auto token_bytes =
          bounds.token_bytes ? bounds.token_bytes : bounds.decoded_bytes;
      std::size_t bytes =
          multiply(count, sizeof(PublishedEvent) + request.value.size() + 1);
      bytes = add(bytes,
                  multiply(prompt_events,
                           add(std::min(bounds.decoded_bytes,
                                        multiply(progress_chunk, token_bytes)),
                               multiply(progress_chunk,
                                        sizeof(tokenizers::TokenBytes) +
                                            sizeof(text::TokenSegment)))));
      bytes = add(bytes,
                  multiply(count - prompt_events,
                           add(token_bytes, sizeof(tokenizers::TokenBytes) +
                                                sizeof(text::TokenSegment))));
      if (count > setup.session.max_events ||
          bytes > setup.session.max_publication_bytes - reserved_stream_bytes -
                      publication_bytes) {
        throw std::length_error(
            "stream publication reservation exceeds budget");
      }
      auto pool = std::make_shared<StreamEvents>();
      pool->slots.resize(count);
      for (std::size_t i = 0; i < count; ++i) {
        auto& event = pool->slots.at(i);
        event.request = request;
        event.body = TokenDelta{};
        auto& delta = std::get<TokenDelta>(event.body);
        const auto tokens = i < prompt_events ? progress_chunk : 1;
        delta.decoded.bytes.reserve(
            std::min(bounds.decoded_bytes, multiply(tokens, token_bytes)));
        delta.decoded.tokens.reserve(tokens);
        delta.segments.reserve(tokens);
      }
      reserved_stream_bytes += bytes;
      next_stream_event = 0;
      stream_events = std::move(pool);
      return true;
    } catch (...) {
      publication_failed();
      return false;
    }
  }
  bool progress(const RunProgress& update) noexcept override {
    try {
      const auto& data = update.delta.get();
      for (std::size_t first = 0; first < data.tokens.size() || first == 0;) {
        const auto count = std::min(progress_chunk, data.tokens.size() - first);
        const auto begin = count ? data.tokens.at(first).begin : 0;
        const auto end = count ? data.tokens.at(first + count - 1).end : 0;
        if (!stream_events ||
            next_stream_event == stream_events->slots.size()) {
          publication_failed();
          return false;
        }
        auto& slot = stream_events->slots.at(next_stream_event++);
        auto& delta = std::get<TokenDelta>(slot.body);
        delta.stream = {update.role, job_index};
        delta.first_position = update.first_position + first;
        delta.prompt_tokens = update.prompt_tokens;
        delta.consumed_tokens = update.consumed_tokens;
        delta.capture_site = update.capture_site;
        if (end - begin > delta.decoded.bytes.capacity() ||
            count > delta.decoded.tokens.capacity() ||
            update.segments.size() > delta.segments.capacity()) {
          publication_failed();
          return false;
        }
        delta.decoded.bytes.assign(data.bytes, begin, end - begin);
        for (std::size_t i = first; i < first + count; ++i) {
          auto token = data.tokens.at(i);
          token.begin -= begin;
          token.end -= begin;
          delta.decoded.tokens.push_back(token);
        }
        for (const auto& segment : update.segments) {
          if (segment.position >= delta.first_position &&
              segment.position < delta.first_position + count) {
            delta.segments.push_back(segment);
          }
        }
        if (!publish(std::shared_ptr<PublishedEvent>(stream_events, &slot))) {
          return false;
        }
        if (!count) {
          break;
        }
        first += count;
      }
      return true;
    } catch (...) {
      publication_failed();
      return false;
    }
  }
  void execute(const StartRequest& value, RunId id,
               const std::stop_token& stop) {
    run.reset();
    if (stop.stop_requested() || !phase(WorkPhase::loading_target)) {
      return;
    }
    if (!device) {
      device = setup.open_device();
    }
    if (!device) {
      throw std::runtime_error("device unavailable");
    }
    auto inputs = setup.make_inputs();
    if (stop.stop_requested()) {
      return;
    }
    run = std::make_unique<Run>(*device, id, std::move(inputs), setup.run);
    run->generate(value.prompt, value.settings, *this, stop);
    if (const auto* record = run->record()) {
      const auto& target = record->target();
      if (!target.generation.failure_message.empty()) {
        std::cerr << "Target failure: " << target.generation.failure_message
                  << '\n';
      }
      TargetAvailable ready{target.pair,
                            target.capture_site,
                            {},
                            target.generation.settings,
                            target.generation.reason,
                            target.backend_identity};
      for (std::size_t position = 0;
           position < target.generation.consumed_tokens; ++position) {
        if (record->capture({target.capture_site.point, {position}})) {
          if (!ready.captures.empty() &&
              ready.captures.back().end == position) {
            ++ready.captures.back().end;
          } else {
            ready.captures.push_back({position, position + 1});
          }
        }
      }
      (void)emit(std::move(ready), true);
    }
  }
  void execute(const InterpretRequest& value, RunId,
               const std::stop_token& stop) {
    for (job_index = 0; job_index < value.jobs.size(); ++job_index) {
      const auto& job = value.jobs.at(job_index);
      AttemptAvailable done{job_index,
                            job.capture,
                            {},
                            job.settings,
                            AttemptOutcome::failed,
                            {},
                            {},
                            "Interpretation was not admitted."};
      if (stop.stop_requested()) {
        done.outcome = AttemptOutcome::cancelled;
        done.message = "Request stopped.";
      } else if (run && run->phase() != Phase::closed && run->record()) {
        try {
          const auto id =
              run->interpret(job.capture, job.settings, *this, stop);
          const auto& result = run->record()->variant(id).interpretation;
          if (!result.failure_message.empty()) {
            std::cerr << "Interpretation failure: " << result.failure_message
                      << '\n';
          }
          done.variant = id;
          done.outcome = result.outcome;
          done.explanation = result.explanation;
          if (result.generation) {
            done.stop = result.generation->reason;
          }
          done.message =
              result.outcome == AttemptOutcome::failed
                  ? "Interpretation failed; inspect the local service log."
                  : "";
        } catch (const std::exception& error) {
          std::cerr << "Interpretation admission: " << error.what() << '\n';
          done.message = "Interpretation rejected by run admission.";
        }
      }
      (void)emit(std::move(done), true);
    }
  }
  void execute(const SteerRequest& value, RunId, const std::stop_token& stop) {
    if (!run || !run->record() || run->phase() == Phase::closed) {
      throw std::invalid_argument("No retained run for steering.");
    }
    const auto& record = *run->record();
    const auto id = run->steer(value.edit, *this, stop);
    const auto& trial = record.trial(id).steering;
    (void)emit(
        SteeringAvailable{id, trial.outcome,
                          trial.baseline.generation
                              ? std::optional(trial.baseline.generation->reason)
                              : std::nullopt,
                          trial.steered.generation
                              ? std::optional(trial.steered.generation->reason)
                              : std::nullopt,
                          trial.message},
        true);
  }
  void work() noexcept {
    try {
      for (;;) {
        std::shared_ptr<const Work> item;
        std::stop_token stop;
        {
          std::unique_lock lock(mutex);
          available.wait(lock, [this] { return closing || !queue.empty(); });
          if (closing) {
            break;
          }
          item = queue.front();
          queue.pop_front();
          state.queued_jobs -= jobs(item->command);
          active_stop = std::stop_source();
          stop = active_stop.get_token();
          state.active_request = identity(item->command);
        }
        request = identity(item->command);
        job_index = 0;
        observer_failed = false;
        auto outcome = AttemptOutcome::completed;
        std::string message;
        try {
          std::visit(
              [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (!std::is_same_v<T, StopRequest>) {
                  execute(value, item->run, stop);
                }
              },
              item->command);
          if (observer_failed || !run || run->phase() == Phase::closed) {
            outcome = AttemptOutcome::failed;
          }
          if (stop.stop_requested() && !observer_failed) {
            outcome = AttemptOutcome::cancelled;
          }
        } catch (const std::exception& error) {
          std::cerr << "Request failure: " << error.what() << '\n';
          outcome = AttemptOutcome::failed;
          message =
              std::string(error.what()).substr(0, max_failure_message_bytes);
        }
        (void)emit(RequestFinished{outcome, std::move(message)}, true);
        {
          std::lock_guard lock(mutex);
          state.active_request.reset();
          state.phase = run && run->record() && run->phase() != Phase::closed
                            ? SessionPhase::retained
                            : SessionPhase::unavailable;
        }
      }
    } catch (...) {
      std::lock_guard lock(mutex);
      worker_failed = true;
      state.phase = SessionPhase::unavailable;
      state.publication_failed = true;
    }
    run.reset();
    device.reset();
    {
      std::lock_guard lock(mutex);
      queue.clear();
      state.queued_jobs = 0;
      state.active_request.reset();
      if (closing) {
        state.phase = SessionPhase::closed;
      }
    }
  }
  Receipt submit(Command command) {
    validate(command, setup.run, setup.session.max_jobs);
    const auto request_id = identity(command);
    auto item = std::make_shared<Work>(Work{std::move(command), {0}});
    std::optional<std::stop_source> stop;
    Receipt result{ReceiptCode::accepted, request_id, {}};
    {
      std::lock_guard lock(mutex);
      if (closing || worker_failed) {
        return {ReceiptCode::unavailable, request_id, state.run};
      }
      if (const auto found = receipts.find(request_id.value);
          found != receipts.end()) {
        return {same(found->second->command, item->command)
                    ? ReceiptCode::replayed
                    : ReceiptCode::conflict,
                request_id, found->second->run};
      }
      const auto* start = std::get_if<StartRequest>(&item->command);
      if (start) {
        if (state.active_request || !queue.empty()) {
          return {ReceiptCode::busy, request_id, state.run};
        }
        if (start->replace.has_value() != state.run.has_value() ||
            (start->replace && start->replace->value != state.run->value)) {
          return {ReceiptCode::stale, request_id, state.run};
        }
        if (next_run == std::numeric_limits<std::uint64_t>::max()) {
          return {ReceiptCode::capacity, request_id, state.run};
        }
        item->run = {next_run + 1};
        // Build replacement containers before clearing the old retained view.
        std::map<std::string, std::shared_ptr<const Work>> ledger;
        ledger.emplace(request_id.value, item);
        std::deque<std::shared_ptr<const Work>> pending;
        pending.push_back(item);
        receipts.swap(ledger);
        queue.swap(pending);
        ++next_run;
        journal.clear();
        stream_events.reset();
        reserved_stream_bytes = 0;
        event_sizes.clear();
        publication_bytes = 0;
        state = {SessionPhase::queued, item->run, {}, 1, {0}, false};
      } else {
        const auto id = std::visit(
            [](const auto& value) -> RunId {
              using T = std::decay_t<decltype(value)>;
              if constexpr (std::is_same_v<T, StartRequest>) {
                return {};
              } else {
                return value.run;
              }
            },
            item->command);
        if (!state.run || id.value != state.run->value) {
          return {ReceiptCode::stale, request_id, state.run};
        }
        item->run = id;
        if (receipts.size() >= setup.session.max_receipts) {
          return {ReceiptCode::capacity, request_id, state.run};
        }
        if (const auto* cancel = std::get_if<StopRequest>(&item->command)) {
          if (!state.active_request ||
              cancel->active_request.value != state.active_request->value) {
            return {ReceiptCode::stale, request_id, state.run};
          }
          receipts.emplace(request_id.value, item);
          stop.emplace(active_stop);
          state.phase = SessionPhase::stopping;
        } else {
          if (state.phase == SessionPhase::unavailable ||
              state.publication_failed) {
            return {ReceiptCode::unavailable, request_id, state.run};
          }
          if (queue.size() >= setup.session.max_commands ||
              jobs(item->command) >
                  setup.session.max_jobs - state.queued_jobs) {
            return {ReceiptCode::capacity, request_id, state.run};
          }
          receipts.emplace(request_id.value, item);
          try {
            queue.push_back(item);
          } catch (...) {
            receipts.erase(request_id.value);
            throw;
          }
          state.queued_jobs += jobs(item->command);
        }
      }
      result.run = item->run;
    }
    if (stop) {
      stop->request_stop();
    }
    available.notify_one();
    return result;
  }
};
Session::Session(SessionSetup setup)
    : impl_(std::make_unique<Impl>(std::move(setup))) {}
Session::~Session() = default;
Receipt Session::submit(StartRequest request) {
  return impl_->submit(std::move(request));
}
Receipt Session::submit(InterpretRequest request) {
  return impl_->submit(std::move(request));
}
Receipt Session::submit(SteerRequest request) {
  return impl_->submit(std::move(request));
}
Receipt Session::stop(StopRequest request) {
  return impl_->submit(std::move(request));
}
SessionStatus Session::status() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->state;
}
void Session::close() { impl_->shutdown(); }
EventPage Session::read(RunId id, EventCursor after) const {
  auto& p = *impl_;
  std::lock_guard lock(p.mutex);
  if (!p.state.run || id.value != p.state.run->value ||
      after.value > p.journal.size()) {
    throw std::invalid_argument("stale run or cursor");
  }
  if (p.readers->count >= p.setup.session.max_readers) {
    throw std::length_error("reader capacity");
  }
  auto page = std::make_unique<EventPage::Impl>(p.readers);
  ++p.readers->count;
  page->run = id;
  page->status = p.state;
  page->next = after;
  std::size_t bytes = event_overhead;
  for (std::size_t i = static_cast<std::size_t>(after.value);
       i < p.journal.size() &&
       page->events.size() < p.setup.session.max_page_events;
       ++i) {
    if (p.event_sizes.at(i) > p.setup.session.max_page_bytes - bytes) {
      break;
    }
    bytes += p.event_sizes.at(i);
    page->events.push_back(p.journal.at(i));
    page->next = p.journal.at(i)->sequence;
  }
  page->more = page->next.value < p.journal.size();
  return EventPage(std::move(page));
}
}  // namespace dolittle::experiment
