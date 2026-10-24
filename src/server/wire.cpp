/*
 * A1 SSH address validation: borrow Host/Origin and configured lists, compare
 * exact configured entries, or recognize localhost/127.0.0.1/[::1] followed by
 * no port or a decimal port in 1..65535. A loopback Origin must be http:// plus
 * that request's Host. Reject malformed suffixes without DNS or allocation.
 * Algorithm: live-application.md, Transport contract. Cost is linear in header
 * bytes and configured entries, with constant extra storage. Implementation
 * corpus: Core I.1 (#ri-explicit), explicit policy inputs; SL.str.2
 * (#rstr-view), borrowed views; performance COPY.7, avoid temporary header
 * strings (guidelines/copy-move/COPY.7-eliminate-hidden-copies.md). Both
 * corpora queried for design and implementation; rules retrieved in full.
 *
 * Steering wire algorithm: validate bounded edit/strength/settings, decode
 * variant IDs without numeric loss, and preserve original/edited text plus
 * baseline/steered stream identities in reconnect pages. Serialize completed
 * metadata only; no vectors cross HTTP. Cost remains linear in page bytes.
 * Steering corpus consultation, header design and implementation 2026-10-20:
 * Core I.1 (#ri-explicit), typed contracts; performance MEM.9, bounded setup
 * before execution (guidelines/memory/MEM.9-embedded-allocate-at-init.md).
 * Both local corpora queried and full rules read.
 *
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
#include <array>
#include <charconv>
#include <cmath>
#include <nlohmann/json.hpp>

#include "server/http.h"
#include "tensors/formats/json_metadata.h"

namespace dolittle::server {
namespace {
bool loopback_authority(std::string_view host) {
  constexpr std::array<std::string_view, 3> names{"localhost", "127.0.0.1",
                                                  "[::1]"};
  constexpr std::size_t max_port_digits = 5;
  for (const auto name : names) {
    if (host == name) return true;
    if (!host.starts_with(name)) continue;
    auto suffix = host.substr(name.size());
    if (!suffix.starts_with(':')) continue;
    suffix.remove_prefix(1);
    if (suffix.empty() || suffix.size() > max_port_digits) return false;
    std::uint16_t port{};
    const auto parsed = std::from_chars(std::to_address(suffix.begin()),
                                        std::to_address(suffix.end()), port);
    return parsed.ec == std::errc{} &&
           parsed.ptr == std::to_address(suffix.end()) && port != 0;
  }
  return false;
}
constexpr auto bits_per_byte = 8;
constexpr auto decimal_u64_digits = 20;
constexpr auto metadata_expansion = 32;
constexpr auto max_enum_bytes = 32;
using Json = nlohmann::json;
constexpr std::size_t bits_per_sextet = 6;
constexpr std::size_t bytes_per_group = 3;
constexpr std::uint32_t sextet_mask = 63;
void fields(const Json& value, std::initializer_list<std::string_view> required,
            std::initializer_list<std::string_view> optional = {}) {
  if (!value.is_object()) throw std::invalid_argument("expected object");
  for (auto key : required) {
    if (!value.contains(key)) throw std::invalid_argument("missing field");
  }
  for (const auto& [key, unused] : value.items()) {
    (void)unused;
    if (std::find(required.begin(), required.end(), key) == required.end() &&
        std::find(optional.begin(), optional.end(), key) == optional.end()) {
      throw std::invalid_argument("unknown field");
    }
  }
}
std::string string(const Json& value, std::size_t cap) {
  if (!value.is_string() || value.get_ref<const std::string&>().size() > cap) {
    throw std::invalid_argument("invalid string");
  }
  return value.get<std::string>();
}
std::uint64_t decimal(std::string_view value) {
  std::uint64_t result{};
  if (value.empty() || (value.size() > 1 && value.front() == '0')) {
    throw std::invalid_argument("invalid decimal ID");
  }
  const auto parsed = std::from_chars(std::to_address(value.begin()),
                                      std::to_address(value.end()), result);
  if (parsed.ec != std::errc{} || parsed.ptr != std::to_address(value.end())) {
    throw std::invalid_argument("invalid decimal ID");
  }
  return result;
}
std::size_t natural(const Json& value) {
  if (!value.is_number_unsigned()) {
    throw std::invalid_argument("expected unsigned integer");
  }
  const auto n = value.get<std::uint64_t>();
  if (n > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("integer out of range");
  }
  return static_cast<std::size_t>(n);
}
inference::GenerationSettings settings(const Json& value) {
  fields(value,
         {"mode", "temperature", "top_k", "top_p", "seed", "max_new_tokens"});
  const auto mode = string(value.at("mode"), max_enum_bytes);
  if (mode != "greedy" && mode != "categorical") {
    throw std::invalid_argument("unknown sampling mode");
  }
  if (!value.at("temperature").is_number() || !value.at("top_p").is_number()) {
    throw std::invalid_argument("invalid sampling scalar");
  }
  inference::GenerationSettings result{
      {mode == "greedy" ? inference::sampling::SelectionMode::greedy
                        : inference::sampling::SelectionMode::categorical,
       value.at("temperature").get<double>(), natural(value.at("top_k")),
       value.at("top_p").get<double>()},
      decimal(string(value.at("seed"), decimal_u64_digits)),
      natural(value.at("max_new_tokens"))};
  inference::sampling::validate_settings(result.sampling);
  return result;
}
Json settings(inference::GenerationSettings value) {
  return {
      {"mode", value.sampling.mode == inference::sampling::SelectionMode::greedy
                   ? "greedy"
                   : "categorical"},
      {"temperature", value.sampling.temperature},
      {"top_k", value.sampling.top_k},
      {"top_p", value.sampling.top_p},
      {"seed", std::to_string(value.seed)},
      {"max_new_tokens", value.max_new_tokens}};
}
Json key(experiment::CaptureKey value) {
  return {{"position", value.position.value},
          {"block", value.site.block.value},
          {"boundary",
           value.site.boundary == models::ObservationBoundary::block_output
               ? "block_output"
               : "block_input"}};
}
experiment::CaptureKey key(const Json& value) {
  fields(value, {"position", "block", "boundary"});
  const auto boundary = string(value.at("boundary"), metadata_expansion);
  if (boundary != "block_output" && boundary != "block_input") {
    throw std::invalid_argument("unknown capture boundary");
  }
  return {
      {boundary == "block_output" ? models::ObservationBoundary::block_output
                                  : models::ObservationBoundary::block_input,
       {natural(value.at("block"))}},
      {natural(value.at("position"))}};
}
std::string base64(std::string_view bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string output;
  output.reserve(((bytes.size() + 2) / bytes_per_group) * 4);
  for (std::size_t i = 0; i < bytes.size(); i += bytes_per_group) {
    const auto size = std::min(bytes_per_group, bytes.size() - i);
    std::uint32_t value = static_cast<unsigned char>(bytes.at(i));
    value = (value << bits_per_byte) |
            (size > 1 ? static_cast<unsigned char>(bytes.at(i + 1)) : 0U);
    value = (value << bits_per_byte) |
            (size > 2 ? static_cast<unsigned char>(bytes.at(i + 2)) : 0U);
    output += alphabet.at((value >> (bits_per_sextet * 3)) & sextet_mask);
    output += alphabet.at((value >> (bits_per_sextet * 2)) & sextet_mask);
    output +=
        size > 1 ? alphabet.at((value >> bits_per_sextet) & sextet_mask) : '=';
    output += size > 2 ? alphabet.at(value & sextet_mask) : '=';
  }
  return output;
}
std::string_view phase(experiment::SessionPhase value) {
  constexpr std::array names = {"idle",
                                "queued",
                                "loading_target",
                                "generating_target",
                                "releasing_target",
                                "retained",
                                "loading_verbalizer",
                                "interpreting",
                                "loading_reconstructor",
                                "reconstructing",
                                "generating_baseline",
                                "generating_steered",
                                "stopping",
                                "unavailable",
                                "closed"};
  return names.at(static_cast<std::size_t>(value));
}
std::string_view outcome(experiment::AttemptOutcome value) {
  constexpr std::array names = {"completed", "cancelled", "failed"};
  return names.at(static_cast<std::size_t>(value));
}
std::string_view reason(inference::StopReason value) {
  constexpr std::array names = {"stop_token", "token_limit", "user_stop",
                                "execution_failure"};
  return names.at(static_cast<std::size_t>(value));
}
Json status(const experiment::SessionStatus& value) {
  return {{"phase", phase(value.phase)},
          {"run",
           value.run ? Json(std::to_string(value.run->value)) : Json(nullptr)},
          {"active_request", value.active_request
                                 ? Json(value.active_request->value)
                                 : Json(nullptr)},
          {"queued_jobs", value.queued_jobs},
          {"latest", std::to_string(value.latest.value)},
          {"publication_failed", value.publication_failed}};
}
Json event(const experiment::PublishedEvent& event_value) {
  auto result = std::visit(
      [](const auto& value) -> Json {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, experiment::WorkPhase>) {
          constexpr std::array names = {
              "loading_target",     "generating_target",
              "releasing_target",   "target_retained",
              "loading_verbalizer", "generating_verbalizer",
              "verbalizer_ready",   "loading_reconstructor",
              "reconstructing",     "generating_baseline",
              "generating_steered", "closed"};
          return {{"type", "phase"},
                  {"phase", names.at(static_cast<std::size_t>(value))}};
        } else if constexpr (std::is_same_v<T, experiment::TokenDelta>) {
          Json tokens = Json::array();
          for (std::size_t i = 0; i < value.decoded.tokens.size(); ++i) {
            const auto& token = value.decoded.tokens.at(i);
            auto kind = text::SegmentKind::unclassified;
            if (!value.segments.empty()) kind = value.segments.at(i).kind;
            constexpr std::array names = {"unclassified", "reasoning", "answer",
                                          "control"};
            tokens.push_back(
                {{"id", token.id},
                 {"begin", token.begin},
                 {"end", token.end},
                 {"mapped", token.mapped},
                 {"kind", names.at(static_cast<std::size_t>(kind))}});
          }
          return {{"type", "tokens"},
                  {"role",
                   std::array{"target", "verbalizer", "baseline", "steered"}.at(
                       static_cast<std::size_t>(value.stream.role))},
                  {"job", value.stream.job_index},
                  {"first", value.first_position},
                  {"prompt_tokens", value.prompt_tokens},
                  {"consumed_tokens", value.consumed_tokens},
                  {"site", value.capture_site
                               ? key({value.capture_site->point, {0}})
                               : Json(nullptr)},
                  {"bytes", base64(value.decoded.bytes)},
                  {"tokens", std::move(tokens)}};
        } else if constexpr (std::is_same_v<T, experiment::TargetAvailable>) {
          Json ranges = Json::array();
          for (auto range : value.captures) {
            ranges.push_back({range.begin, range.end});
          }
          return {{"type", "target_retained"},
                  {"site", key({value.site.point, {0}})},
                  {"capture_ranges", std::move(ranges)},
                  {"settings", settings(value.settings)},
                  {"stop", reason(value.stop)},
                  {"backend", value.backend_identity},
                  {"pair",
                   {{"convention", value.pair.convention},
                    {"target", value.pair.target.repository},
                    {"target_revision", value.pair.target.revision},
                    {"verbalizer", value.pair.verbalizer.repository},
                    {"verbalizer_revision", value.pair.verbalizer.revision}}}};
        } else if constexpr (std::is_same_v<T, experiment::AttemptAvailable>) {
          constexpr std::array tags = {"complete", "missing", "malformed"};
          return {
              {"type", "interpretation"},
              {"job", value.job_index},
              {"capture", key(value.capture)},
              {"variant", value.variant
                              ? Json(std::to_string(value.variant->value))
                              : Json(nullptr)},
              {"settings", settings(value.settings)},
              {"outcome", outcome(value.outcome)},
              {"stop", value.stop ? Json(reason(*value.stop)) : Json(nullptr)},
              {"format",
               tags.at(static_cast<std::size_t>(value.explanation.format))},
              {"content", value.explanation.content
                              ? Json::array({value.explanation.content->begin,
                                             value.explanation.content->end})
                              : Json(nullptr)},
              {"message", value.message}};
        } else if constexpr (std::is_same_v<T, experiment::SteeringStarted>) {
          return {{"type", "steering_started"},
                  {"variant", std::to_string(value.variant.value)},
                  {"capture", key(value.capture)},
                  {"original", value.original},
                  {"edited", value.edited},
                  {"strength", value.strength},
                  {"settings", settings(value.settings)},
                  {"prefix_tokens", value.prefix_tokens},
                  {"original_prompt_tokens", value.original_prompt_tokens}};
        } else if constexpr (std::is_same_v<T, experiment::SteeringAvailable>) {
          return {{"type", "steering_complete"},
                  {"trial", std::to_string(value.trial.value)},
                  {"outcome", outcome(value.outcome)},
                  {"message", value.message},
                  {"baseline_stop", value.baseline_stop
                                        ? Json(reason(*value.baseline_stop))
                                        : Json(nullptr)},
                  {"steered_stop", value.steered_stop
                                       ? Json(reason(*value.steered_stop))
                                       : Json(nullptr)}};
        } else {
          return {{"type", "request_finished"},
                  {"outcome", outcome(value.outcome)},
                  {"message", value.message}};
        }
      },
      event_value.body);
  result.emplace("sequence", std::to_string(event_value.sequence.value));
  result.emplace("request", event_value.request.value);
  return result;
}
std::string bounded(const Json& value, std::size_t cap) {
  auto bytes = value.dump();
  if (bytes.size() > cap) throw std::length_error("response byte limit");
  return bytes;
}
}  // namespace
bool allowed_host(std::string_view host,
                  std::span<const std::string> configured) {
  return std::find(configured.begin(), configured.end(), host) !=
             configured.end() ||
         loopback_authority(host);
}
bool allowed_origin(std::string_view origin, std::string_view host,
                    std::span<const std::string> configured) {
  constexpr std::string_view scheme = "http://";
  return origin.empty() ||
         std::find(configured.begin(), configured.end(), origin) !=
             configured.end() ||
         (loopback_authority(host) && origin.starts_with(scheme) &&
          origin.substr(scheme.size()) == host);
}
Command decode_command(CommandInput input, WireLimits limits) {
  const auto& [method, path, body] = input;
  if (method != "POST" || body.size() > limits.max_request_bytes) {
    throw std::invalid_argument("invalid request method or size");
  }
  const auto document = tensors::formats::detail::parse_metadata(
                            std::as_bytes(std::span(body.data(), body.size())),
                            limits.max_request_bytes * metadata_expansion)
                            .value;
  try {
    if (path == "/api/v1/runs") {
      fields(document, {"request", "prompt", "settings"}, {"replace"});
      const auto& prompt = document.at("prompt");
      fields(prompt, {"user", "thinking"}, {"system"});
      const auto thinking = string(prompt.at("thinking"), max_enum_bytes);
      if (thinking != "enabled" && thinking != "disabled") {
        throw std::invalid_argument("unknown thinking mode");
      }
      experiment::StartRequest request{
          {string(document.at("request"), limits.max_id_bytes)},
          {},
          {{},
           string(prompt.at("user"), limits.max_request_bytes),
           thinking == "enabled" ? text::ThinkingMode::enabled
                                 : text::ThinkingMode::disabled},
          settings(document.at("settings"))};
      if (document.contains("replace") && !document.at("replace").is_null()) {
        request.replace = {
            decimal(string(document.at("replace"), decimal_u64_digits))};
      }
      if (prompt.contains("system") && !prompt.at("system").is_null()) {
        request.prompt.system =
            string(prompt.at("system"), limits.max_request_bytes);
      }
      return request;
    }
    constexpr std::string_view prefix = "/api/v1/runs/";
    if (!path.starts_with(prefix)) {
      throw std::invalid_argument("unknown command route");
    }
    const auto rest = path.substr(prefix.size());
    const auto slash = rest.find('/');
    if (slash == std::string_view::npos) {
      throw std::invalid_argument("unknown command route");
    }
    const experiment::RunId run{decimal(rest.substr(0, slash))};
    const auto suffix = rest.substr(slash);
    if (suffix == "/stop") {
      fields(document, {"request", "active_request"});
      return experiment::StopRequest{
          {string(document.at("request"), limits.max_id_bytes)},
          run,
          {string(document.at("active_request"), limits.max_id_bytes)}};
    }
    if (suffix == "/steering") {
      fields(document,
             {"request", "variant", "explanation", "strength", "settings"});
      if (!document.at("strength").is_number()) {
        throw std::invalid_argument("invalid strength");
      }
      const auto strength = document.at("strength").get<float>();
      if (!std::isfinite(strength) || strength < 0 || strength > 2) {
        throw std::invalid_argument("strength must be between 0 and 2");
      }
      auto explanation =
          string(document.at("explanation"), nla::max_explanation_bytes);
      if (explanation.empty()) throw std::invalid_argument("empty explanation");
      return experiment::SteerRequest{
          {string(document.at("request"), limits.max_id_bytes)},
          run,
          {{decimal(string(document.at("variant"), decimal_u64_digits))},
           std::move(explanation),
           strength,
           settings(document.at("settings"))}};
    }
    if (suffix != "/interpretations") {
      throw std::invalid_argument("unknown command route");
    }
    fields(document, {"request", "jobs"});
    const auto& jobs = document.at("jobs");
    if (!jobs.is_array() || jobs.empty() || jobs.size() > limits.max_jobs) {
      throw std::length_error("job count limit");
    }
    experiment::InterpretRequest request{
        {string(document.at("request"), limits.max_id_bytes)}, run, {}};
    for (const auto& job : jobs) {
      fields(job, {"capture", "settings"});
      request.jobs.push_back(
          {key(job.at("capture")), settings(job.at("settings"))});
    }
    return request;
  } catch (const Json::exception&) {
    throw std::invalid_argument("invalid request schema");
  }
}
std::string encode_page(experiment::EventPageView page, std::string_view epoch,
                        WireLimits limits) {
  Json events = Json::array();
  for (const auto& owner : page.events) {
    if (!owner) throw std::invalid_argument("missing event");
    events.push_back(event(*owner));
  }
  return bounded({{"epoch", epoch},
                  {"run", std::to_string(page.run.value)},
                  {"status", status(page.status.get())},
                  {"events", std::move(events)},
                  {"next", std::to_string(page.next.value)},
                  {"more", page.more}},
                 limits.max_response_bytes);
}
std::string encode_status(const experiment::SessionStatus& value,
                          const HttpOptions& options) {
  auto data = status(value);
  data.emplace("epoch", options.epoch);
  data.emplace("backend", options.backend_identity);
  data.emplace("pair", options.pair_identity);
  data.emplace("steering_available", options.steering_available);
  data.emplace(
      "limits",
      Json{{"target_tokens", options.run_limits.max_target_tokens},
           {"verbalizer_tokens", options.run_limits.max_verbalizer_tokens},
           {"variants", options.run_limits.record.max_variants},
           {"trials", options.run_limits.record.max_trials},
           {"jobs", options.wire.max_jobs}});
  return bounded(data, options.wire.max_response_bytes);
}
std::string encode_receipt(const experiment::Receipt& value, std::size_t cap) {
  constexpr std::array names = {"accepted", "replayed", "busy",       "stale",
                                "conflict", "capacity", "unavailable"};
  return bounded({{"code", names.at(static_cast<std::size_t>(value.code))},
                  {"request", value.request.value},
                  {"run", value.run ? Json(std::to_string(value.run->value))
                                    : Json(nullptr)}},
                 cap);
}
}  // namespace dolittle::server
