/*
 * Local live-application HTTP boundary: IMPLEMENTATION, 2026-10-19.
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
 * Dependency proposal: source-pinned cpp-httplib (MIT) for HTTP transport,
 * nlohmann/json already pinned for JSON. Select revision/hash before
 * implementation. Implementation corpus queried 2026-10-19 (full rules read):
 * cpp-guidelines CP.22 (#rconc-unknown), encode/send outside Session locks;
 * CP.23 (#rconc-join), join network workers before destruction; CP.41
 * (#rconc-create), bounded reusable pool. cpp-perf-guidelines COPY.7, borrow
 * request bytes and const event owners, avoid per-poll history copies; source
 * guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 */
#include "server/http.h"

#include <httplib.h>
#include <openssl/crypto.h>

#include <charconv>
#include <fstream>

namespace dolittle::server {
namespace {
constexpr auto minimum_token_bytes = 32;
constexpr auto http_ok = 200;
constexpr auto http_accepted = 202;
constexpr auto http_bad_request = 400;
constexpr auto http_unauthorized = 401;
constexpr auto http_forbidden = 403;
constexpr auto http_conflict = 409;
constexpr auto http_capacity = 429;
constexpr auto http_unavailable = 503;
constexpr std::size_t max_asset_bytes = std::size_t{4} * 1024 * 1024;
void error(httplib::Response& response, int code, std::string_view message) {
  response.status = code;
  response.set_content(std::string(message), "text/plain; charset=utf-8");
}
std::uint64_t decimal(std::string_view text) {
  std::uint64_t value{};
  const auto parsed = std::from_chars(std::to_address(text.begin()),
                                      std::to_address(text.end()), value);
  if (text.empty() || (text.size() > 1 && text.front() == '0') ||
      parsed.ec != std::errc{} || parsed.ptr != std::to_address(text.end())) {
    throw std::invalid_argument("invalid cursor");
  }
  return value;
}
bool allowed(std::string_view value, const std::vector<std::string>& list) {
  return std::find(list.begin(), list.end(), value) != list.end();
}
}  // namespace
struct Server::Impl {
  std::reference_wrapper<experiment::Session> session;
  HttpOptions options;
  httplib::Server http;
  Impl(experiment::Session& service, HttpOptions config)
      : session(service), options(std::move(config)) {
    if (options.access_token.size() < minimum_token_bytes ||
        options.epoch.empty() || !options.workers || !options.max_connections ||
        !options.socket_timeout_ms || options.allowed_hosts.empty() ||
        options.allowed_origins.empty()) {
      throw std::invalid_argument("invalid HTTP configuration");
    }
    http.new_task_queue = [this] {
      return std::make_unique<httplib::ThreadPool>(
                 options.workers, options.workers, options.max_connections)
          .release();
    };  // Library takes ownership of its TaskQueue factory result.
    http.set_payload_max_length(options.wire.max_request_bytes);
    http.set_read_timeout(std::chrono::milliseconds(options.socket_timeout_ms));
    http.set_write_timeout(
        std::chrono::milliseconds(options.socket_timeout_ms));
    http.set_keep_alive_max_count(1);
    http.set_pre_routing_handler([this](const httplib::Request& request,
                                        httplib::Response& response) {
      response.set_header(
          "Content-Security-Policy",
          "default-src 'self'; script-src 'self'; style-src 'self'; font-src "
          "'self'; connect-src 'self'; img-src 'self'; object-src 'none'; "
          "base-uri 'none'; frame-ancestors 'none'");
      response.set_header("X-Content-Type-Options", "nosniff");
      response.set_header("Referrer-Policy", "no-referrer");
      if (request.get_header_value_count("Host") != 1 ||
          !allowed(request.get_header_value("Host"), options.allowed_hosts)) {
        error(response, http_forbidden, "Host is not allowed.");
        return httplib::Server::HandlerResponse::Handled;
      }
      const auto origin = request.get_header_value("Origin");
      if (!origin.empty() && !allowed(origin, options.allowed_origins)) {
        error(response, http_forbidden, "Origin is not allowed.");
        return httplib::Server::HandlerResponse::Handled;
      }
      if (request.path.starts_with("/api/")) {
        response.set_header("Cache-Control", "no-store");
        const auto token = request.get_header_value("Authorization");
        const auto expected = "Bearer " + options.access_token;
        if (token.size() != expected.size() ||
            CRYPTO_memcmp(token.data(), expected.data(), expected.size()) !=
                0) {
          error(response, http_unauthorized,
                "Open the launcher access link to connect.");
          return httplib::Server::HandlerResponse::Handled;
        }
        if (request.method == "POST" &&
            request.get_header_value("X-Dolittle-Epoch") != options.epoch) {
          error(response, http_conflict,
                "Service restarted. Reconnect before sending this request.");
          return httplib::Server::HandlerResponse::Handled;
        }
      }
      return httplib::Server::HandlerResponse::Unhandled;
    });
    http.set_exception_handler([](const httplib::Request&,
                                  httplib::Response& response,
                                  const std::exception_ptr& exception) {
      try {
        std::rethrow_exception(exception);
      } catch (const std::length_error&) {
        error(response, http_capacity, "Request exceeds service capacity.");
      } catch (const std::invalid_argument&) {
        error(response, http_bad_request, "Invalid request.");
      } catch (...) {
        error(response, http_unavailable, "Service unavailable.");
      }
    });
    http.Get("/api/v1/status", [this](const httplib::Request&,
                                      httplib::Response& response) {
      response.set_content(encode_status(session.get().status(), options),
                           "application/json");
    });
    http.Get(
        R"(/api/v1/runs/([0-9]+)/events)",
        [this](const httplib::Request& request, httplib::Response& response) {
          if (request.get_header_value("X-Dolittle-Epoch") != options.epoch) {
            error(response, http_conflict, "Service epoch changed.");
            return;
          }
          if (!request.has_param("after")) {
            throw std::invalid_argument("missing cursor");
          }
          try {
            if (request.matches.size() != 2) {
              throw std::invalid_argument("missing run ID");
            }
            const auto page = session.get().read(
                {decimal(std::next(request.matches.begin())->str())},
                {decimal(request.get_param_value("after"))});
            response.set_content(
                encode_page(page.view(), options.epoch, options.wire),
                "application/json");
          } catch (const std::invalid_argument&) {
            error(response, http_conflict, "Run or cursor is stale.");
          }
        });
    http.Post(
        R"(/api/v1/runs(?:/[0-9]+/(?:interpretations|stop))?)",
        [this](const httplib::Request& request, httplib::Response& response) {
          if (request.get_header_value("Content-Type") != "application/json") {
            error(response, http_bad_request, "Expected application/json.");
            return;
          }
          auto command = decode_command(
              {request.method, request.path, request.body}, options.wire);
          const auto receipt = std::visit(
              [this](auto value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, experiment::StopRequest>) {
                  return session.get().stop(std::move(value));
                } else {
                  return session.get().submit(std::move(value));
                }
              },
              std::move(command));
          switch (receipt.code) {
            case experiment::ReceiptCode::accepted:
              response.status = http_accepted;
              break;
            case experiment::ReceiptCode::replayed:
              response.status = http_ok;
              break;
            case experiment::ReceiptCode::capacity:
              response.status = http_capacity;
              break;
            case experiment::ReceiptCode::unavailable:
              response.status = http_unavailable;
              break;
            default:
              response.status = http_conflict;
              break;
          }
          response.set_content(
              encode_receipt(receipt, options.wire.max_response_bytes),
              "application/json");
        });
    struct Asset {
      std::string_view path;
      std::string_view mime;
    };
    constexpr std::array assets = {
        Asset{"index.html", "text/html; charset=utf-8"},
        Asset{"app.css", "text/css"},
        Asset{"client.js", "text/javascript"},
        Asset{"controller.js", "text/javascript"},
        Asset{"view.js", "text/javascript"},
        Asset{"fonts/plex-serif.woff2", "font/woff2"},
        Asset{"fonts/plex-mono.woff2", "font/woff2"}};
    const auto root = std::filesystem::canonical(options.asset_root);
    for (const auto& asset : assets) {
      const auto path = std::filesystem::canonical(root / asset.path);
      if (path.lexically_relative(root).empty() ||
          path.lexically_relative(root).native().starts_with("..") ||
          std::filesystem::file_size(path) > max_asset_bytes) {
        throw std::invalid_argument("invalid packaged asset");
      }
      std::ifstream input(path, std::ios::binary);
      std::string bytes(
          static_cast<std::size_t>(std::filesystem::file_size(path)), '\0');
      if (!input.read(bytes.data(),
                      static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("cannot read packaged asset");
      }
      const auto route = asset.path == "index.html"
                             ? std::string("/")
                             : "/" + std::string(asset.path);
      http.Get(route,
               [body = std::move(bytes), mime = std::string(asset.mime)](
                   const httplib::Request&, httplib::Response& response) {
                 response.set_content(body, mime);
               });
    }
  }
};
Server::Server(experiment::Session& session, HttpOptions options)
    : impl_(std::make_unique<Impl>(session, std::move(options))) {}
Server::~Server() = default;
void Server::serve(const std::stop_token& stop) {
  auto& p = *impl_;
  if (stop.stop_requested()) return;
  if (!p.http.bind_to_port(p.options.bind_address, p.options.port)) {
    throw std::runtime_error("cannot bind HTTP listener");
  }
  std::stop_callback callback(stop, [&p] { p.http.stop(); });
  if (!stop.stop_requested() && !p.http.listen_after_bind()) {
    throw std::runtime_error("HTTP listener failed");
  }
}
}  // namespace dolittle::server
