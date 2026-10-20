/*
 * Live experiment session: IMPLEMENTED CONTRACT, 2026-10-19.
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
 * Header corpus (both servers queried, full rules read): cpp-guidelines CP.23
 * (#rconc-join), join before ownership ends; CP.41 (#rconc-create), persistent
 * worker; CP.22 (#rconc-unknown), callbacks outside locks; F.27
 * (#rf-shared_ptr), shared ownership only for immutable events crossing
 * reader/worker lifetimes. cpp-perf-guidelines COPY.7, move payloads and borrow
 * views instead of copying full histories;
 * guidelines/copy-move/COPY.7-eliminate-hidden-copies.md. GPU.1, publication
 * consumes completed host results without extra readback;
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>

#include "experiment/run.h"

namespace dolittle::experiment {
struct RequestId {
  std::string value;
};
struct EventCursor {
  std::uint64_t value;
};
struct StartRequest {
  RequestId request;
  std::optional<RunId> replace;  // Required if a current run exists.
  text::Prompt prompt;
  inference::GenerationSettings settings{};
};
struct InterpretationJob {
  CaptureKey capture{};
  inference::GenerationSettings settings{};
};
struct InterpretRequest {
  RequestId request;
  RunId run{};
  std::vector<InterpretationJob> jobs;
};
struct StopRequest {
  RequestId request;
  RunId run{};
  RequestId active_request;
};
enum class ReceiptCode : std::uint8_t {
  accepted,
  replayed,
  busy,
  stale,
  conflict,
  capacity,
  unavailable
};
struct Receipt {
  ReceiptCode code{ReceiptCode::unavailable};
  RequestId request;
  std::optional<RunId> run;
};
enum class SessionPhase : std::uint8_t {
  idle,
  queued,
  loading_target,
  generating_target,
  releasing_target,
  retained,
  loading_verbalizer,
  interpreting,
  stopping,
  unavailable,
  closed
};
struct SessionStatus {
  SessionPhase phase{SessionPhase::idle};
  std::optional<RunId> run;
  std::optional<RequestId> active_request;
  std::size_t queued_jobs{};
  EventCursor latest{};
  bool publication_failed{};
};
struct StreamKey {
  StreamRole role;
  std::size_t
      job_index;  // Zero for target; request ID scopes interpretation jobs.
};
struct TokenDelta {
  StreamKey stream;
  std::size_t first_position;
  std::size_t prompt_tokens;
  std::size_t consumed_tokens;
  tokenizers::DecodedBytes decoded;
  std::vector<text::TokenSegment> segments;
  std::optional<models::ObservationSite> capture_site{};
};
struct PositionRange {
  std::size_t begin;
  std::size_t end;  // Exclusive.
};
struct TargetAvailable {
  nla::PairIdentity pair;
  models::ObservationSite site;
  std::vector<PositionRange> captures;
  inference::GenerationSettings settings{};
  inference::StopReason stop;
  std::string backend_identity;
};
struct AttemptAvailable {
  std::size_t job_index;
  CaptureKey capture{};
  std::optional<VariantId>
      variant;  // Rejection before Run may produce no variant.
  inference::GenerationSettings settings{};
  AttemptOutcome outcome;
  std::optional<inference::StopReason> stop;
  nla::ExplanationRange
      explanation;  // Relative to generated AV raw bytes only.
  std::string message;
};
struct RequestFinished {
  AttemptOutcome outcome;
  std::string message;
};
using EventBody = std::variant<WorkPhase, TokenDelta, TargetAvailable,
                               AttemptAvailable, RequestFinished>;
struct PublishedEvent {
  EventCursor sequence;
  RequestId request;
  EventBody body;
};
// Borrowed projection also supports pure wire tests with in-memory events.
struct EventPageView {
  RunId run{};
  std::reference_wrapper<const SessionStatus> status;
  std::span<const std::shared_ptr<const PublishedEvent>> events;
  EventCursor next;
  bool more;
};
// Move-only page pins an accounted reader lease until destruction. Session
// replacement cannot accumulate unbounded old journals through outstanding
// reads.
class EventPage {
 public:
  ~EventPage();
  EventPage(const EventPage&) = delete;
  EventPage& operator=(const EventPage&) = delete;
  EventPage(EventPage&&) noexcept;
  EventPage& operator=(EventPage&&) noexcept;
  [[nodiscard]] EventPageView view() const noexcept;
  [[nodiscard]] RunId run() const noexcept;
  [[nodiscard]] const SessionStatus& status() const noexcept;
  [[nodiscard]] std::span<const std::shared_ptr<const PublishedEvent>> events()
      const noexcept;
  [[nodiscard]] EventCursor next() const noexcept;
  [[nodiscard]] bool more() const noexcept;

 private:
  friend class Session;
  struct Impl;
  explicit EventPage(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
struct SessionLimits {
  std::size_t max_commands;
  std::size_t max_jobs;
  std::size_t max_receipts;
  std::size_t max_events;
  std::size_t max_publication_bytes;
  std::size_t max_page_events;
  std::size_t max_page_bytes;
  std::size_t max_readers;
};
struct SessionSetup {
  std::function<std::unique_ptr<gpu::Device>()> open_device;
  std::function<RunInputs()> make_inputs;
  RunLimits run;
  SessionLimits session;
};
class Session {
 public:
  explicit Session(SessionSetup);
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) = delete;
  Session& operator=(Session&&) = delete;
  [[nodiscard]] Receipt submit(StartRequest);
  [[nodiscard]] Receipt submit(InterpretRequest);
  [[nodiscard]] Receipt stop(StopRequest);
  [[nodiscard]] SessionStatus status() const;
  [[nodiscard]] EventPage read(RunId, EventCursor after) const;
  void close();  // Idempotent; joins. No concurrent destruction by callers.

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::experiment
