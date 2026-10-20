/*
 * Axis D2: Linux/NVIDIA composition of the live experiment application.
 * Contract: local operator chooses the manifest/cache/assets, GPU ordinal and
 * listener. HTTP clients cannot change these inputs. Secrets come from a local
 * token file, not model metadata. Hashes are verified by setup when the trusted
 * launcher selects prepared integrity; direct calls default to full SHA-256.
 * Sizes and structure are checked before model construction; downloaded code is
 * never executed. Factory captures share verified immutable archives; each Run
 * owns its tokenizer/pair and sampler. Backing files must remain unchanged
 * until service shutdown (format.h).
 *
 * Algorithm: live-application.md, "Algorithm and ownership": read bounded
 * configuration, admit target/AV archives concurrently before HTTP readiness,
 * then give Session factories sharing those read-only mappings. A new Run
 * builds text helpers and resolves the pair from the retained bytes. Providers
 * bind the selected archive without reopening or hashing it; execution uploads
 * that role. Its worker opens the CUDA device and later destroys Run before
 * device. Main owns HTTP and Session; signal handling only sets a flag, a
 * joined watcher requests stop, HTTP joins first, then Session drains. Port
 * binding failure unwinds these owners. Model/provider failures are reported
 * without fallback.
 *
 * Cost: two persistent read-only archive mappings, at most eight parallel file
 * checks (four per role) only for direct sha256 mode. Prepared mode reads
 * bounded metadata without a full payload scan. Mapped pages are
 * OS-reclaimable; mapping size is distinct from heap admission and physical
 * RSS. Charge both archive metadata allowances plus per-run text/pair metadata.
 * Optimization: providers share verified mappings across all turns and role
 * switches. Only the active role's weights are uploaded/resident on the GPU.
 * Target capacity is P+G <= 8192 (prompt plus generated tokens); Run rejects
 * oversize requests before upload. AV capacity remains independent at 512.
 * At the target ceiling, KV uses 147456*8192 = 1.125 GiB and raw captures
 * use 8192*8192 = 64 MiB, excluding weights, scratch and metadata. Actual
 * sequence allocations use admitted P+G, not the global ceiling.
 * Budget design/implementation corpus, 2026-10-19: cpp-guidelines SL.con.3
 * (#rsl-bounds), retain checked capacity admission; cpp-perf-guidelines GPU.9,
 * preserve separate resident weights and per-sequence arenas, and measure
 * device high-water use;
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md. The
 * numerical kernels, precision, frozen oracle and fixtures are unchanged.
 * Implementation corpus queried 2026-10-19: cpp-guidelines CP.23 (#rconc-join),
 * join before owner destruction; CP.22 (#rconc-unknown), no callback under a
 * lock. cpp-perf-guidelines COPY.7, immutable provider data and moved
 * RunInputs; guidelines/copy-move/COPY.7-eliminate-hidden-copies.md. GPU.1,
 * upload only the active role; guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Archive reuse design/implementation corpus, 2026-10-19: cpp-guidelines
 * R.21 (#rr-unique), F.27 (#rf-shared_ptr): service and bindings share const
 * ownership; CP.23 (#rconc-join): startup tasks join before inputs die.
 * cpp-perf-guidelines CDSA.32: prepare once and share immutable source data;
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 * Prepared installation design/implementation, 2026-10-19: setup hashes pinned
 * artifacts and atomically records their manifest and file fingerprints. The
 * trusted launcher admits unchanged installations without reading weight
 * payloads. Full hash checking remains the default for direct archive/service
 * callers. Both modes preserve size, path, header, index and extent checks and
 * canonical identity. Optimization: move O(weight bytes) integrity work to
 * setup/repair; normal startup maps payloads and reads bounded metadata only.
 * No numerical change. cpp-guidelines I.1 (#ri-explicit): name the trusted
 * preparation precondition. cpp-perf-guidelines CDSA.32: persist versioned
 * preparation for immutable data;
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 * Steady-state design, 2026-10-20:
 * Compose device sampling for the live CUDA path. Host categorical
 * selection remains an explicit reference path; no per-step vocabulary
 * transfer is required by live composition.
 * Overlay: transformer-execution.md, "Device selection and allocation-free
 * generation". Core I.7 (#ri-post): borrowing, bounds and failure contracts.
 * Performance GPU.1: device-resident decisions; MEM.9: prepare storage before
 * steady state and check the invariant. Sources:
 * guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/memory/MEM.9-embedded-allocate-at-init.md.
 * Both corpora queried/read for header design and implementation, 2026-10-20.
 */
#include <openssl/rand.h>

#include <array>
#include <charconv>
#include <csignal>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <thread>

#include "bounds.h"
#include "gpu/cuda/device.h"
#include "inference/sampling/categorical.h"
#include "models/qwen3/checkpoint.h"
#include "nla/pairs/qwen3_8b.h"
#include "server/http.h"
#include "tensors/formats/json_metadata.h"
#include "tensors/formats/safetensors.h"
#include "text/qwen3_chat.h"
#include "tokenizers/byte_bpe.h"

namespace {
constexpr auto http_workers = 4;
constexpr auto verification_workers = 4;
constexpr auto hex_mask = 15;
constexpr auto prefill_chunk = 16;
constexpr auto safetensors_header_mib = 16;
constexpr auto manifest_allowance_mib = 16;
constexpr auto max_variants = 16;
constexpr auto max_jobs = 16;
constexpr auto max_connections = 16;
constexpr auto epoch_bytes = 16;
constexpr auto max_receipts = 256;
constexpr auto max_page_events = 128;
constexpr auto max_readers = 4;
constexpr auto max_commands = 8;
constexpr auto max_token_file_bytes = 256;
constexpr auto publication_mib = 128;
constexpr auto max_id_bytes = 128;
constexpr auto pair_template_tokens = 4096;
constexpr auto hex_shift = 4;
constexpr auto device_gib = 23;
constexpr auto tokenizer_mib = 128;
constexpr auto metadata_mib = 256;
constexpr auto target_context = 8192;
constexpr auto av_context = 512;
constexpr auto socket_timeout_ms = 5000;
constexpr auto signal_poll_ms = 100;
constexpr auto max_tensor_count = 10000;
constexpr auto event_limit = 32768;
constexpr auto maximum_port = 65535;
constexpr auto text_byte_limit = 65536;
constexpr auto request_byte_limit = 131072;
constexpr auto qwen3_vocabulary = 151936;
namespace tensors = dolittle::tensors;
namespace tokenizers = dolittle::tokenizers;
namespace text = dolittle::text;
namespace nla = dolittle::nla;
namespace models = dolittle::models;
namespace inference = dolittle::inference;
namespace experiment = dolittle::experiment;
namespace gpu = dolittle::gpu;
namespace server = dolittle::server;
using Json = nlohmann::json;
constexpr std::size_t mib = std::size_t{1024} * 1024;
constexpr std::size_t gib = 1024 * mib;
// The C signal ABI requires signal-safe storage; only the handler writes it.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
volatile std::sig_atomic_t interrupted = 0;
void signal_handler(int) { interrupted = 1; }
int integer(std::string_view value) {
  int result{};
  const auto parsed = std::from_chars(std::to_address(value.begin()),
                                      std::to_address(value.end()), result);
  if (parsed.ec != std::errc{} || parsed.ptr != std::to_address(value.end())) {
    throw std::invalid_argument("invalid integer option");
  }
  return result;
}
std::string read(const std::filesystem::path& path, std::size_t cap) {
  const auto size = std::filesystem::file_size(path);
  if (size > cap) {
    throw std::length_error("configuration file limit");
  }
  std::ifstream source(path, std::ios::binary);
  std::string bytes(static_cast<std::size_t>(size), '\0');
  if (!source.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("configuration read failed");
  }
  return bytes;
}
std::string artifact(const tensors::formats::TensorArchive& archive,
                     std::string_view name) {
  const auto bytes = archive.artifact(name);
  std::string result(bytes.size(), '\0');
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}
struct Text {
  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
  std::unique_ptr<text::Protocol> protocol;
};
Text text_helpers(const tensors::formats::TensorArchive& archive) {
  const auto tokenizer = artifact(archive, "tokenizer.json"),
             config = artifact(archive, "tokenizer_config.json"),
             generation = artifact(archive, "generation_config.json");
  const auto& source = archive.identity();
  const auto identity =
      source.repository + "/" + source.revision + "/" + source.artifact_digest;
  return {tokenizers::make_byte_bpe({tokenizer, config, identity},
                                    {tokenizer_mib * mib, mib, mib, mib}),
          text::make_qwen3_chat({config, generation, identity}, mib)};
}
using Request = tensors::formats::safetensors::OpenRequest;
using Integrity = tensors::formats::safetensors::Integrity;
std::shared_ptr<const Request> request(const Json& model,
                                       const std::filesystem::path& cache,
                                       Integrity integrity) {
  auto repo = model.at("repo").get<std::string>();
  auto folder = repo;
  const auto separator = folder.find('/');
  if (separator == std::string::npos) {
    throw std::invalid_argument("model repository");
  }
  folder.replace(separator, 1, "--");
  Request result{cache / folder / model.at("revision").get<std::string>(),
                 repo,
                 model.at("revision"),
                 {},
                 {},
                 model.at("index").get<std::string>(),
                 safetensors_header_mib * mib,
                 tokenizer_mib * mib,
                 max_tensor_count,
                 verification_workers,
                 integrity};
  for (const auto& f : model.at("files")) {
    result.artifacts.push_back({f.at("path"), f.at("bytes"), f.at("sha256")});
    const auto& path = result.artifacts.back().relative_path;
    if (path.ends_with(".safetensors")) {
      result.shards.push_back(path);
    }
  }
  return std::make_shared<const Request>(std::move(result));
}
using SharedArchive = std::shared_ptr<const tensors::formats::TensorArchive>;
experiment::ModelProvider provider(SharedArchive archive) {
  return [archive = std::move(archive)](std::size_t allowance) {
    if (allowance < tokenizer_mib * mib) {
      throw std::length_error("loader allowance");
    }
    auto identity = archive->identity();
    return experiment::LoadedModel{
        std::move(identity),
        models::qwen3::bind_causal_lm(archive, "config.json"),
        tokenizer_mib * mib};
  };
}
struct Sources {
  SharedArchive target;
  SharedArchive verbalizer;
};
experiment::RunInputs inputs(const Sources& sources) {
  const auto& [target_archive, av_archive] = sources;
  auto target_text = text_helpers(*target_archive),
       av_text = text_helpers(*av_archive);
  auto pair = nla::pairs::make_qwen3_8b(
      {target_archive->identity(), av_archive->identity(),
       av_archive->artifact("nla_meta.yaml"), *av_text.tokenizer,
       *av_text.protocol},
      {mib, mib, pair_template_tokens});
  const tensors::ComputationMode mode{tensors::ElementType::bfloat16,
                                      tensors::ElementType::float32};
  auto target_provider = provider(target_archive);
  auto av_provider = provider(av_archive);
  experiment::RunInputs result{std::move(pair),
                               std::move(target_provider),
                               std::move(av_provider),
                               mode,
                               mode,
                               std::move(target_text.tokenizer),
                               std::move(av_text.tokenizer),
                               std::move(target_text.protocol),
                               {},
                               "cuda",
                               (metadata_mib + 2 * tokenizer_mib) * mib,
                               inference::sampling::Execution::device};
  return result;
}
std::string epoch() {
  std::array<unsigned char, epoch_bytes> bytes{};
  if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
    throw std::runtime_error("random epoch failed");
  }
  constexpr std::string_view hex = "0123456789abcdef";
  std::string result;
  for (auto b : bytes) {
    result += hex.at(b >> hex_shift);
    result += hex.at(b & hex_mask);
  }
  return result;
}
}  // namespace
int main(int argc, char** argv) try {
  std::map<std::string, std::string> args;
  const auto arguments = std::span(argv, static_cast<std::size_t>(argc));
  for (std::size_t i = 1; i < arguments.size(); i += 2) {
    if (i + 1 >= arguments.size()) {
      throw std::invalid_argument("expected --name value");
    }
    if (!args.emplace(dolittle::at(arguments, i),
                      dolittle::at(arguments, i + 1))
             .second) {
      throw std::invalid_argument("duplicate option");
    }
  }
  for (const auto& [name, value] : args) {
    (void)value;
    if (name != "--manifest" && name != "--cache" && name != "--assets" &&
        name != "--token-file" && name != "--bind" && name != "--port" &&
        name != "--host" && name != "--device" && name != "--model-integrity") {
      throw std::invalid_argument("unknown option");
    }
  }
  auto integrity = Integrity::sha256;
  if (args.contains("--model-integrity")) {
    const auto& selected = args.at("--model-integrity");
    if (selected == "prepared") {
      integrity = Integrity::prepared_setup;
    } else if (selected != "sha256") {
      throw std::invalid_argument("unknown model integrity policy");
    }
  }
  const auto manifest_bytes = read(args.at("--manifest"), mib);
  const auto manifest = tensors::formats::detail::parse_metadata(
                            std::as_bytes(std::span(manifest_bytes.data(),
                                                    manifest_bytes.size())),
                            manifest_allowance_mib * mib)
                            .value;
  if (manifest.at("schema") != 1 || manifest.at("pair") != "qwen3-8b-nla" ||
      manifest.at("models").size() != 2) {
    throw std::invalid_argument("unsupported production manifest");
  }
  std::shared_ptr<const Request> target, av;
  for (const auto& model : manifest.at("models")) {
    if (model.at("role") == "target" && !target) {
      target = request(model, args.at("--cache"), integrity);
    } else if (model.at("role") == "verbalizer" && !av) {
      av = request(model, args.at("--cache"), integrity);
    } else {
      throw std::invalid_argument("duplicate model role");
    }
  }
  if (!target || !av) {
    throw std::invalid_argument("missing model role");
  }
  const int port = integer(args.at("--port"));
  if (port < 1 || port > maximum_port) {
    throw std::invalid_argument("port range");
  }
  const int ordinal =
      args.contains("--device") ? integer(args.at("--device")) : 0;
  const experiment::RunLimits limits{device_gib * gib,
                                     gib,
                                     prefill_chunk,
                                     target_context,
                                     av_context,
                                     text_byte_limit,
                                     {metadata_mib * mib, max_variants}};
  const auto verification_started = std::chrono::steady_clock::now();
  std::cerr << (integrity == Integrity::prepared_setup
                    ? "Opening models verified during setup...\n"
                    : "Verifying model files with SHA-256...\n");
  auto target_ready = std::async(std::launch::async, [target] {
    return tensors::formats::safetensors::open(*target);
  });
  auto av_ready = std::async(std::launch::async, [av] {
    return tensors::formats::safetensors::open(*av);
  });
  const Sources sources{target_ready.get(), av_ready.get()};
  std::cerr << "Model archives ready in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             verification_started)
                   .count()
            << " seconds; retaining read-only mappings.\n";
  experiment::Session session(
      {[ordinal] { return gpu::cuda::open_device(ordinal); },
       [sources] { return inputs(sources); },
       limits,
       {max_commands, max_jobs, max_receipts, event_limit,
        publication_mib * mib, max_page_events, mib, max_readers}});
  const auto host = args.at("--host");
  server::HttpOptions options{
      args.at("--bind"),
      static_cast<std::uint16_t>(port),
      {host},
      {"http://" + host},
      read(args.at("--token-file"), max_token_file_bytes),
      epoch(),
      args.at("--assets"),
      http_workers,
      max_connections,
      socket_timeout_ms,
      {request_byte_limit, mib, max_jobs, max_id_bytes},
      limits,
      "cuda",
      "Qwen3-8B / Qwen3-8B-nla"};
  server::Server server(session, std::move(options));
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  std::stop_source stop;
  std::jthread signals([&stop](const std::stop_token& finished) {
    while (!finished.stop_requested() && !interrupted) {
      std::this_thread::sleep_for(std::chrono::milliseconds(signal_poll_ms));
    }
    if (interrupted) {
      stop.request_stop();
    }
  });
  std::cout << "Dolittle listening at http://" << host << "\n" << std::flush;
  server.serve(stop.get_token());
  signals.request_stop();
  session.close();
  return 0;
} catch (const std::exception& error) {
  std::cerr << "dolittle: " << error.what() << '\n';
  return 1;
}
