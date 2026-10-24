/*
 * A1 address units: configured hosts and loopback forwarded ports are accepted;
 * malformed/lookalike hosts and unrelated origins are rejected. Exercise pure
 * predicates on in-memory strings without a server. Cost O(test input bytes).
 * Design/implementation corpus: Core I.1 (#ri-explicit), test public contracts;
 * SL.str.2 (#rstr-view), borrowed inputs; performance COPY.7, view-based cases
 * (guidelines/copy-move/COPY.7-eliminate-hidden-copies.md).
 *
 * Steering units preserve full-width variant IDs and reject invalid strength
 * before typed admission. No session, socket or model execution occurs.
 * Steering corpus consultation, header design and implementation 2026-10-20:
 * Core I.1 (#ri-explicit), typed contracts; performance MEM.9, bounded setup
 * before execution (guidelines/memory/MEM.9-embedded-allocate-at-init.md).
 * Both local corpora queried and full rules read.
 *
 * Axis A1/D3: pure wire contracts; no sockets, threads, filesystem or GPU.
 * Inputs are owned JSON strings and presentation values; outputs preserve IDs,
 * binary token bytes and contiguous cursors. Reject ambiguous fields and
 * numeric precision loss. Algorithm: live-application.md, Transport contract:
 * decode, inspect typed values, encode an immutable in-memory page, inspect its
 * fields. Cost O(input/output bytes). Implementation corpus 2026-10-19: Core
 * CP.22
 * (#rconc-unknown), serialization requires no worker lock; performance COPY.7,
 * borrowed page values; guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 */
#include <nlohmann/json.hpp>

#include "server/http.h"
#include "tests/support/check.h"
namespace server = dolittle::server;
namespace experiment = dolittle::experiment;
using dolittle::tests::check;
using dolittle::tests::expect_throws;
using dolittle::tests::report_failure;
int main() try {
  const std::vector<std::string> hosts{"spark.example:8787"};
  const std::vector<std::string> origins{"http://spark.example:8787"};
  check(server::allowed_host(hosts.front(), hosts), "configured LAN host");
  check(server::allowed_origin(origins.front(), hosts.front(), origins),
        "configured LAN origin");
  for (const auto host : {"localhost", "127.0.0.1", "[::1]", "localhost:8877",
                          "127.0.0.1:1", "[::1]:65535"}) {
    check(server::allowed_host(host, hosts), "loopback forwarded authority");
    check(server::allowed_origin(std::string("http://") + host, host, origins),
          "forwarded page origin matches Host");
    check(server::allowed_origin("", host, origins), "absent Origin");
    for (const auto origin :
         {"null", "https://evil.example", "http://localhost:9",
          "http://localhost.evil.example", "http://[::1]:9"}) {
      check(!server::allowed_origin(origin, host, origins), "unrelated origin");
    }
  }
  for (const auto host :
       {"", "evil.example:8787", "localhost.evil.example:8787",
        "127.0.0.1.evil.example", "localhost@evil.example",
        "localhost:", "localhost:0", "localhost:65536", "localhost:-1",
        "localhost:+80", "localhost:80/path", "localhost:80?x",
        "localhost:80#x", "localhost:80 ", "127.0.0.2:8787", "::1",
        "[::1]:", "localhost:1,evil"}) {
    check(!server::allowed_host(host, hosts),
          "malformed/unconfigured authority");
  }
  const server::WireLimits limits{65536, 65536, 16, 128};
  const std::string body =
      R"({"request":"abc","prompt":{"user":"hello","thinking":"enabled"},"settings":{"mode":"greedy","temperature":1,"top_k":0,"top_p":1,"seed":"18446744073709551615","max_new_tokens":8}})";
  auto command = server::decode_command({"POST", "/api/v1/runs", body}, limits);
  const auto& start = std::get<experiment::StartRequest>(command);
  check(start.settings.seed == std::numeric_limits<std::uint64_t>::max(),
        "64-bit seed is exact");
  check(start.prompt.user == "hello", "prompt preserved");
  expect_throws<std::invalid_argument>([&] {
    (void)server::decode_command(
        {"POST", "/api/v1/runs", R"({"request":"a","request":"b"})"}, limits);
  });
  auto malformed = nlohmann::json::parse(body);
  malformed.at("settings").at("seed") = 18446744073709551615ULL;
  expect_throws<std::invalid_argument>([&] {
    (void)server::decode_command({"POST", "/api/v1/runs", malformed.dump()},
                                 limits);
  });
  malformed = nlohmann::json::parse(body);
  malformed.at("prompt").emplace("unknown", true);
  expect_throws<std::invalid_argument>([&] {
    (void)server::decode_command({"POST", "/api/v1/runs", malformed.dump()},
                                 limits);
  });
  const auto steering_settings = nlohmann::json::parse(body).at("settings");
  auto steering = nlohmann::json{{"request", "steer"},
                                 {"variant", "9007199254740993"},
                                 {"explanation", "edited text"},
                                 {"strength", 0.5},
                                 {"settings", steering_settings}};
  const auto steered = server::decode_command(
      {"POST", "/api/v1/runs/1/steering", steering.dump()}, limits);
  check(std::get<experiment::SteerRequest>(steered).edit.variant.value ==
            9007199254740993ULL,
        "variant identity preserves 64 bits");
  steering.at("strength") = 2.1;
  expect_throws<std::invalid_argument>([&] {
    (void)server::decode_command(
        {"POST", "/api/v1/runs/1/steering", steering.dump()}, limits);
  });
  experiment::TokenDelta delta{
      {experiment::StreamRole::target, 0}, 0, 1, 0, {}, {}};
  delta.decoded.bytes = std::string("\xff\0", 2);
  delta.decoded.tokens = {{17, 0, 2, true}};
  auto event = std::make_shared<const experiment::PublishedEvent>(
      experiment::PublishedEvent{{1}, {"abc"}, std::move(delta)});
  const std::array events{event};
  const experiment::SessionStatus status{
      experiment::SessionPhase::generating_target,
      experiment::RunId{1},
      experiment::RequestId{"abc"},
      0,
      {1},
      false};
  const auto page = nlohmann::json::parse(server::encode_page(
      {{1}, std::cref(status), events, {1}, false}, "epoch", limits));
  check(page.at("events").at(0).at("bytes") == "/wA=",
        "raw invalid UTF-8 and NUL survive base64");
  check(page.at("next") == "1" && page.at("events").at(0).at("sequence") == "1",
        "cursor identity preserved");
  expect_throws<std::length_error>([&] {
    auto small = limits;
    small.max_response_bytes = 1;
    (void)server::encode_page({{1}, std::cref(status), events, {1}, false},
                              "epoch", small);
  });
} catch (const std::exception& error) {
  return report_failure(error.what());
}
