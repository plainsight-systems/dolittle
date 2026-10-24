/*
 * SSH access contract (A1): allowed_host accepts configured authorities or
 * exact loopback names with an optional valid TCP port. allowed_origin accepts
 * configured origins, absent Origin, or http:// plus the accepted loopback
 * Host. Both borrow inputs for the call and return a boolean without
 * allocation. The HTTP caller rejects duplicate Host/Origin headers and still
 * authenticates API requests. See live-application.md, Transport contract.
 * Algorithm: compare configured values; otherwise split a recognized loopback
 * name from its port, validate decimal range 1..65535, then compare Origin's
 * HTTP authority with Host. No DNS lookup or network access occurs.
 * Cost: linear in configured entries and header bytes, constant extra storage.
 * Design corpus: Core I.1 (#ri-explicit), explicit predicates; SL.str.2
 * (#rstr-view), borrowed character views; performance COPY.7, no temporary
 * strings (guidelines/copy-move/COPY.7-eliminate-hidden-copies.md).
 *
 * Steering transport contract: POST /api/v1/runs/{run}/steering carries a
 * variant, edited text, strength and matched generation settings. The session
 * resolves source capture/prefix; clients cannot provide activation vectors.
 * Branch events retain request identity for reconnect and scoped cancellation.
 * Steering corpus consultation, header design and implementation 2026-10-20:
 * Core I.1 (#ri-explicit), typed contracts; performance MEM.9, bounded setup
 * before execution (guidelines/memory/MEM.9-embedded-allocate-at-init.md).
 * Both local corpora queried and full rules read.
 *
 * Local live-application HTTP boundary: IMPLEMENTED CONTRACT, 2026-10-19.
 * Axis A1: assets, wire encoding, request delivery and reconnect pages.
 * Contract: borrow a live Session for Server's full lifetime. Own HTTP
 * listener, fixed asset contents, access-token configuration and bounded worker
 * pool. Server never calls Run, reads Record or accesses GPU memory.
 * Application routes are /api/v1; assets come from a fixed packaged allowlist.
 * Browser text is data, never executable markup. No model download or arbitrary
 * file path is accepted from HTTP. Bind/payload/timeouts/origin limits are
 * explicit.
 *
 * Algorithm: docs/architecture/live-application.md, "Transport contract":
 * 1. Authenticate API request and validate Host/Origin, method/path/content
 * type and body byte cap before parsing. Parse strict duplicate-free JSON into
 *    typed commands. Reject unknown fields, non-finite numeric settings and
 *    invalid enum/ID ranges. Decimal strings preserve 64-bit values in JS.
 * 2. Require server epoch on mutations, then submit to Session. Return its
 *    receipt; accepted means queued. Stable client IDs make retries idempotent.
 *    No socket handler waits for generation or holds the Session mutex.
 * 3. For event reads, validate epoch/run/cursor and borrow one bounded page.
 *    Serialize outside Session locks; encode token bytes as base64 and retain
 *    exact positions/IDs. Page cursor advances only through included events.
 *    Account for JSON escaping/base64 expansion before choosing page bounds.
 * 4. Send the bounded response using fixed workers and socket timeouts. Dropped
 *    connections release page owners and do not stop the experiment. Reconnect
 *    reads the existing journal rather than regenerating output. Exceeding
 *    connection/queue capacity rejects rather than spawning unbounded workers.
 * 5. Stop accepting before shutdown, terminate/drain HTTP workers, then allow
 *    composition to close Session. No HTTP worker can outlive borrowed Session.
 *
 * Cost: O(request bytes + returned page bytes) per call, W*R response budget
 * plus bounded requests and pinned event references. Optimization: serialize
 * deltas, never raw activation vectors or the entire growing trace. Fixed
 * assets may be cached once. Known-maximum asset/JSON sizes are checked before
 * allocation; compression is disabled initially to keep memory costs explicit.
 * Pure decode_command/encode_page touch only in-memory data for contract units;
 * Server binding/thread/socket behavior belongs to integration verification.
 * Dependencies: cpp-httplib 0.62.0 (MIT), source archive SHA256 pinned in
 * CMake; existing nlohmann/json for JSON. Implementation corpus queried
 * 2026-10-19: cpp-guidelines CP.22 (#rconc-unknown),
 * encode/send outside Session locks; CP.23 (#rconc-join), join network workers
 * before destruction; CP.41 (#rconc-create), bounded reusable pool.
 * cpp-perf-guidelines COPY.7, borrow request bytes and const event owners,
 * avoid per-poll history copies; source
 * guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 */
#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <string_view>

#include "experiment/session.h"

namespace dolittle::server {
[[nodiscard]] bool allowed_host(std::string_view host,
                                std::span<const std::string> configured);
[[nodiscard]] bool allowed_origin(std::string_view origin,
                                  std::string_view host,
                                  std::span<const std::string> configured);
using Command =
    std::variant<experiment::StartRequest, experiment::InterpretRequest,
                 experiment::SteerRequest, experiment::StopRequest>;
struct WireLimits {
  std::size_t max_request_bytes;
  std::size_t max_response_bytes;
  std::size_t max_jobs;
  std::size_t max_id_bytes;
};
// Epoch is checked by the HTTP boundary; path and body identities must agree.
struct CommandInput {
  std::string_view method;
  std::string_view path;
  std::string_view body;
};
[[nodiscard]] Command decode_command(CommandInput, WireLimits);
[[nodiscard]] std::string encode_page(experiment::EventPageView,
                                      std::string_view epoch, WireLimits);
struct HttpOptions {
  std::string bind_address;
  std::uint16_t port;
  std::vector<std::string> allowed_hosts;
  std::vector<std::string> allowed_origins;
  std::string access_token;
  std::string epoch;
  std::filesystem::path asset_root;
  std::size_t workers;
  std::size_t max_connections;
  std::size_t socket_timeout_ms;
  WireLimits wire;
  experiment::RunLimits run_limits;
  std::string backend_identity;
  std::string pair_identity;
  bool steering_available = false;
};
[[nodiscard]] std::string encode_status(const experiment::SessionStatus&,
                                        const HttpOptions&);
[[nodiscard]] std::string encode_receipt(const experiment::Receipt&,
                                         std::size_t max_bytes);
class Server {
 public:
  Server(experiment::Session&, HttpOptions);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;
  Server(Server&&) = delete;
  Server& operator=(Server&&) = delete;
  void serve(
      const std::stop_token&);  // Blocks; joins workers before returning.

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::server
