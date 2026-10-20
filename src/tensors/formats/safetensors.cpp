/*
 * Implementation of the reviewed ownership and decoding contracts.
 * Axis M2, safetensors kind: pure metadata decoding plus verified archive
 * loading. Contract: implement safetensors.h and the archive-owned view
 * lifetime in format.h. Parsing consumes memory; loading alone performs file
 * I/O. Enforce limits before metadata growth. All
 * input/layout/overflow/integrity failures follow the header categories;
 * unwinding closes every acquired mapping/handle. Algorithm: transformer
 * overlay, "Checkpoint loading and binding":
 * 1. Validate explicit artifact/shard/index paths and expected digests; reject
 *    escapes/duplicates and open only the declared artifacts.
 * 2. Check actual byte counts and hashes (unless admitted by prepared setup);
 * decode the 8-byte little-endian header length only after checking framing.
 * Bound all file-offset arithmetic.
 * 3. Call parse_header on the bounded JSON span; check dense ranges, dtypes and
 *    complete payload coverage. Reject duplicate JSON keys rather than
 * last-wins.
 * 4. Match the supplied index to actual shard contents, construct canonical
 *    source identity, retain backing owners and publish only the complete
 * archive. Design and cost: sha256 mode reads each file; prepared mode reads
 * only metadata. Both keep O(tensors) owned
 * metadata, mapped payload views and bounded staging later. Optimization: no
 * eager host copy/conversion of full weights. nlohmann/json 3.12.0 parses
 * bounded metadata; OpenSSL EVP computes SHA-256. SAX preflight rejects
 * duplicate keys before DOM construction. Range/catalog ordering costs O(T log
 * T), outside execution. Format reference:
 * https://github.com/huggingface/safetensors#format. Unit tests call the same
 * in-memory parser. File/hash/index integration coverage belongs to the small
 * integration budget, not a file-spawning unit fixture. Overlay:
 * docs/architecture/transformer-execution.md. Corpus consultation,
 * implementation (full rules read), 2026-10-18: cpp-guidelines R.1 (#rr-raii):
 * resource ownership survives exceptions and teardown. cpp-perf-guidelines
 * GPU.1: keep transfers/conversions explicit and bounded. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 *
 * Static-analysis design/implementation, 2026-10-19:
 * The mapping owner retains a releasable address but exposes only const spans.
 * Fixed-width parsing uses named format constants and checked byte access.
 * cpp-guidelines SL.con.3 (#rsl-bounds), I.30 (#ri-encapsulate): make checked
 * host access explicit and confine necessary native API operations.
 * cpp-perf-guidelines COPY.7: borrow spans/references without hidden payload
 * copies. Source: guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 *
 * Parallel integrity implementation, 2026-10-19: after mapping/size checks,
 * hash immutable file views in batches of at most verification_workers. Each
 * task owns a digest; the caller alone mutates archive metadata. Drain the
 * batch before scheduling another, then parse only after all checks succeed.
 * Async futures join on both success and unwinding before mappings die.
 * SHA-256 bytes/order per artifact and sorted manifest identity are unchanged.
 * Optimization: overlap independent full-file hashes without payload copies;
 * total O(bytes) work and O(workers) transient digest/thread state.
 * cpp-guidelines CP.23 (#rconc-join): scoped tasks retain valid borrowed spans.
 * cpp-perf-guidelines CDSA.32: parallel one-time source preparation;
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
 */
#include "tensors/formats/safetensors.h"

#include <fcntl.h>
#include <openssl/evp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <future>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include "bounds.h"
#include "tensors/formats/json_metadata.h"

namespace dolittle::tensors::formats::safetensors {
namespace {
inline constexpr unsigned byte_bits = 8;
inline constexpr unsigned byte_mask = 255;
inline constexpr unsigned sha256_bytes = 32;
inline constexpr std::size_t sha256_hex_chars = 64;
inline constexpr unsigned hex_digit_mask = 15;
inline constexpr std::size_t artifact_overhead = 128;

constexpr std::size_t prefix_bytes = sizeof(std::uint64_t);
using detail::Json;

std::size_t as_size(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::length_error("artifact exceeds host address range");
  }
  return static_cast<std::size_t>(value);
}

ElementType dtype(const Json& value) {
  if (!value.is_string()) {
    throw std::invalid_argument("tensor dtype must be a string");
  }
  const auto& name = value.get_ref<const std::string&>();
  if (name == "BF16") {
    return ElementType::bfloat16;
  }
  if (name == "F32") {
    return ElementType::float32;
  }
  if (name == "U32") {
    return ElementType::uint32;
  }
  throw std::invalid_argument("unsupported safetensors dtype");
}

struct ParsedHeader {
  std::vector<TensorRange> ranges;
  std::size_t metadata_bytes;
};

ParsedHeader decode_header(std::span<const std::byte> header,
                           std::uint64_t data_bytes, HeaderLimits limits) {
  if (header.size() > limits.max_header_bytes) {
    throw std::length_error("header limit exceeded");
  }
  if (header.empty() || header.front() != std::byte{'{'}) {
    throw std::invalid_argument("safetensors header must begin with an object");
  }
  auto document = detail::parse_metadata(header, limits.max_metadata_bytes);
  const auto& root = document.value;
  const auto count = root.size() - (root.contains("__metadata__") ? 1 : 0);
  if (count > limits.max_tensor_count) {
    throw std::length_error("tensor count limit exceeded");
  }
  auto used = document.metadata_bytes;
  std::vector<TensorRange> ranges;
  if (count > (limits.max_metadata_bytes - used) / sizeof(TensorRange)) {
    throw std::length_error("tensor catalog metadata limit exceeded");
  }
  detail::charge(used, count * sizeof(TensorRange), limits.max_metadata_bytes);
  ranges.reserve(count);
  for (const auto& [name, entry] : root.items()) {
    if (name == "__metadata__") {
      if (!entry.is_object()) {
        throw std::invalid_argument("metadata must be a string map");
      }
      for (const auto& value : entry) {
        if (!value.is_string()) {
          throw std::invalid_argument("metadata values must be strings");
        }
      }
      continue;
    }
    if (!entry.is_object() || entry.size() != 3 || !entry.contains("dtype") ||
        !entry.contains("shape") || !entry.contains("data_offsets") ||
        !entry.at("shape").is_array() || !entry.at("data_offsets").is_array() ||
        entry.at("data_offsets").size() != 2) {
      throw std::invalid_argument("malformed tensor descriptor");
    }
    detail::charge(used, name.size(), limits.max_metadata_bytes);
    const auto rank = entry.at("shape").size();
    if (rank > (limits.max_metadata_bytes - used) / (2 * sizeof(std::size_t))) {
      throw std::length_error("tensor shape metadata limit exceeded");
    }
    detail::charge(used, rank * 2 * sizeof(std::size_t),
                   limits.max_metadata_bytes);
    std::vector<std::size_t> shape;
    shape.reserve(rank);
    for (const auto& dimension : entry.at("shape")) {
      shape.push_back(detail::size_integer(dimension));
    }
    auto layout = contiguous_layout(dtype(entry.at("dtype")), shape);
    const auto begin = detail::unsigned_integer(entry.at("data_offsets").at(0));
    const auto end = detail::unsigned_integer(entry.at("data_offsets").at(1));
    if (begin > end || end > data_bytes ||
        end - begin != required_bytes(layout)) {
      throw std::invalid_argument("tensor payload range does not match shape");
    }
    ranges.push_back({{name, std::move(layout)}, begin, end - begin});
  }
  // Zero-byte entries do not occupy ranges; they may name any in-bounds offset.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> occupied;
  if (ranges.size() >
      (limits.max_metadata_bytes - used) / sizeof(occupied.at(0))) {
    throw std::length_error("range metadata limit exceeded");
  }
  detail::charge(used, ranges.size() * sizeof(occupied.at(0)),
                 limits.max_metadata_bytes);
  occupied.reserve(ranges.size());
  for (const auto& range : ranges) {
    if (range.byte_count != 0) {
      occupied.emplace_back(range.data_offset, range.byte_count);
    }
  }
  std::sort(occupied.begin(), occupied.end());
  std::uint64_t next = 0;
  for (const auto& [offset, count_bytes] : occupied) {
    if (offset != next) {
      throw std::invalid_argument("payload has a gap or overlapping tensors");
    }
    next += count_bytes;  // Each checked end is <= data_bytes.
  }
  if (next != data_bytes) {
    throw std::invalid_argument("payload is not completely indexed");
  }
  return {std::move(ranges), used};
}

class Digest {
 public:
  Digest() : context_(EVP_MD_CTX_new(), &EVP_MD_CTX_free) {
    if (!context_ ||
        EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1) {
      throw std::runtime_error("cannot initialize SHA-256");
    }
  }
  void update(std::span<const std::byte> bytes) {
    constexpr std::size_t chunk_size =
        4 * std::size_t{1024} * std::size_t{1024};
    while (!bytes.empty()) {
      const auto count = std::min(bytes.size(), chunk_size);
      if (EVP_DigestUpdate(context_.get(), bytes.data(), count) != 1) {
        throw std::runtime_error("SHA-256 update failed");
      }
      bytes = bytes.subspan(count);
    }
  }
  void text(std::string_view value) {
    update(std::as_bytes(std::span(value.data(), value.size())));
  }
  void number(std::uint64_t value) {
    std::array<std::byte, sizeof(std::uint64_t)> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      bytes.at(i) = std::byte((value >> (i * byte_bits)) & byte_mask);
    }
    update(bytes);
  }
  std::string finish() {
    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
    unsigned int count = 0;
    if (EVP_DigestFinal_ex(context_.get(), bytes.data(), &count) != 1 ||
        count != sha256_bytes) {
      throw std::runtime_error("SHA-256 finalization failed");
    }
    constexpr std::string_view hex = "0123456789abcdef";
    std::string result;
    result.reserve(sha256_hex_chars);
    for (std::size_t i = 0; i < count; ++i) {
      result.push_back(hex.at(bytes.at(i) >> 4));
      result.push_back(hex.at(bytes.at(i) & hex_digit_mask));
    }
    return result;
  }

 private:
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_;
};

std::string normalized_hash(std::string_view value) {
  if (value.size() != sha256_hex_chars) {
    throw std::invalid_argument("expected a SHA-256 hex digest");
  }
  std::string result;
  result.reserve(sha256_hex_chars);
  for (const auto character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f') ||
          (character >= 'A' && character <= 'F'))) {
      throw std::invalid_argument("invalid SHA-256 hex digest");
    }
    result.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
  }
  return result;
}

void validate_relative(std::string_view name) {
  if (name.empty() || name.find('\0') != std::string_view::npos ||
      name.find('\\') != std::string_view::npos) {
    throw std::invalid_argument("invalid artifact path");
  }
  const std::filesystem::path path{name};
  if (path.is_absolute() || path.has_root_name() ||
      path.generic_string() != name ||
      path.lexically_normal().generic_string() != name) {
    throw std::invalid_argument(
        "artifact path must be normalized and relative");
  }
  for (const auto& component : path) {
    if (component == "." || component == ".." || component.empty()) {
      throw std::invalid_argument("artifact path escapes root");
    }
  }
}

std::filesystem::path inside_root(const std::filesystem::path& root,
                                  std::string_view name) {
  const auto resolved =
      std::filesystem::canonical(root / std::filesystem::path{name});
  const auto relative = resolved.lexically_relative(root);
  if (relative.empty() || relative.is_absolute()) {
    throw std::invalid_argument("artifact outside root");
  }
  for (const auto& component : relative) {
    if (component == "..") {
      throw std::invalid_argument("artifact outside root");
    }
  }
  return resolved;
}

class FileDescriptor {
 public:
  explicit FileDescriptor(int value) : value_(value) {
    if (value_ < 0) {
      throw std::runtime_error("cannot open artifact");
    }
  }
  FileDescriptor(FileDescriptor&&) = delete;
  FileDescriptor& operator=(FileDescriptor&&) = delete;
  ~FileDescriptor() {
    if (value_ >= 0) {
      (void)::close(value_);
    }
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  int get() const noexcept { return value_; }

 private:
  int value_;
};

class Mapping {
 public:
  Mapping(const std::filesystem::path& path, std::uint64_t expected_size) {
    // POSIX open is variadic; O_CREAT is absent, so no mode argument is needed.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    const FileDescriptor file(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
    struct stat information{};
    if (::fstat(file.get(), &information) != 0 ||
        !S_ISREG(information.st_mode) || information.st_size < 0 ||
        static_cast<std::uint64_t>(information.st_size) != expected_size) {
      throw std::runtime_error("artifact type or byte size mismatch");
    }
    size_ = as_size(expected_size);
    if (size_ == 0) {
      return;
    }
    auto* address =
        ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, file.get(), 0);
    if (address == MAP_FAILED) {
      throw std::runtime_error("cannot map artifact");
    }
    data_ = static_cast<std::byte*>(address);
  }
  Mapping(Mapping&&) = delete;
  Mapping& operator=(Mapping&&) = delete;
  ~Mapping() {
    if (data_) {
      (void)::munmap(data_, size_);
    }
  }
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  std::span<const std::byte> bytes() const noexcept { return {data_, size_}; }

 private:
  std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

struct LocatedTensor {
  std::size_t catalog_index;
  std::span<const std::byte> payload;
  std::string shard;
};

class Archive final : public TensorArchive {
 public:
  SourceIdentity source;
  std::map<std::string, std::unique_ptr<Mapping>, std::less<>> mappings;
  std::vector<NamedTensor> catalog;
  std::map<std::string, LocatedTensor, std::less<>> locations;

  const SourceIdentity& identity() const noexcept override { return source; }
  std::span<const NamedTensor> tensors() const noexcept override {
    return catalog;
  }
  HostView tensor(std::string_view name) const override {
    const auto found = locations.find(name);
    if (found == locations.end()) {
      throw std::invalid_argument("unknown tensor name");
    }
    return {&catalog.at(found->second.catalog_index).layout,
            found->second.payload};
  }
  std::span<const std::byte> artifact(std::string_view path) const override {
    const auto found = mappings.find(path);
    if (found == mappings.end()) {
      throw std::invalid_argument("unknown artifact path");
    }
    return found->second->bytes();
  }
};

std::uint64_t header_length(std::span<const std::byte> bytes) {
  if (bytes.size() < prefix_bytes) {
    throw std::invalid_argument("truncated safetensors prefix");
  }
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < prefix_bytes; ++i) {
    value |= static_cast<std::uint64_t>(
                 std::to_integer<unsigned char>(dolittle::at(bytes, i)))
             << (byte_bits * i);
  }
  return value;
}

}  // namespace

std::vector<TensorRange> parse_header(std::span<const std::byte> header_json,
                                      std::uint64_t data_bytes,
                                      HeaderLimits limits) {
  return decode_header(header_json, data_bytes, limits).ranges;
}

std::unique_ptr<TensorArchive> open(const OpenRequest& request) {
  if (request.integrity != Integrity::sha256 &&
      request.integrity != Integrity::prepared_setup) {
    throw std::invalid_argument("unknown integrity policy");
  }
  constexpr std::size_t max_verification_workers = 8;
  if (request.verification_workers == 0 ||
      request.verification_workers > max_verification_workers) {
    throw std::invalid_argument("verification worker limit");
  }
  if (request.artifacts.empty() || request.shards.empty() ||
      request.repository.empty() || request.revision.empty()) {
    throw std::invalid_argument("archive request is incomplete");
  }
  std::map<std::string, Artifact, std::less<>> artifacts;
  std::size_t metadata_used = 0;
  for (const auto& artifact : request.artifacts) {
    validate_relative(artifact.relative_path);
    detail::charge(metadata_used,
                   sizeof(Artifact) + sizeof(Mapping) + artifact_overhead,
                   request.max_metadata_bytes);
    detail::charge(metadata_used, artifact.relative_path.size(),
                   request.max_metadata_bytes);
    Artifact copy{artifact.relative_path, artifact.byte_size,
                  normalized_hash(artifact.sha256)};
    if (!artifacts.emplace(artifact.relative_path, std::move(copy)).second) {
      throw std::invalid_argument("duplicate artifact path");
    }
  }
  std::set<std::string, std::less<>> shards;
  for (const auto& shard : request.shards) {
    validate_relative(shard);
    if (!artifacts.contains(shard) || !shards.insert(shard).second) {
      throw std::invalid_argument("undeclared or duplicate shard");
    }
  }
  if (request.index) {
    validate_relative(*request.index);
    if (!artifacts.contains(*request.index) ||
        shards.contains(*request.index)) {
      throw std::invalid_argument(
          "index must be a declared non-shard artifact");
    }
  }
  const auto root = std::filesystem::canonical(request.root);
  auto archive = std::make_unique<Archive>();
  Digest manifest;
  manifest.text("dolittle-artifacts-v1");
  // Declared after archive so async readers join before mapping destruction.
  std::vector<std::future<void>> checks;
  checks.reserve(request.verification_workers);
  for (const auto& [path, expected] : artifacts) {
    auto mapping =
        std::make_unique<Mapping>(inside_root(root, path), expected.byte_size);
    const auto bytes = mapping->bytes();
    manifest.number(path.size());
    manifest.text(path);
    manifest.number(expected.byte_size);
    manifest.text(expected.sha256);
    archive->mappings.emplace(path, std::move(mapping));
    if (request.integrity == Integrity::prepared_setup) continue;
    const auto verify = [bytes, expected_hash = expected.sha256] {
      Digest actual;
      actual.update(bytes);
      if (actual.finish() != expected_hash) {
        throw std::runtime_error("artifact SHA-256 mismatch");
      }
    };
    if (request.verification_workers == 1) {
      verify();
    } else {
      checks.push_back(std::async(std::launch::async, verify));
      if (checks.size() == request.verification_workers) {
        for (auto& check : checks) check.get();
        checks.clear();
      }
    }
  }
  for (auto& check : checks) check.get();
  archive->source = {request.repository, request.revision, manifest.finish()};
  for (const auto& shard : shards) {
    const auto bytes = archive->artifact(shard);
    const auto size = header_length(bytes);
    if (size > request.max_header_bytes) {
      throw std::length_error("safetensors header limit exceeded");
    }
    if (size > bytes.size() - prefix_bytes) {
      throw std::invalid_argument("truncated safetensors header");
    }
    const auto prefix = prefix_bytes + as_size(size);
    auto parsed = decode_header(
        bytes.subspan(prefix_bytes, as_size(size)), bytes.size() - prefix,
        {request.max_header_bytes, request.max_metadata_bytes - metadata_used,
         request.max_tensor_count - archive->catalog.size()});
    detail::charge(metadata_used, parsed.metadata_bytes,
                   request.max_metadata_bytes);
    for (auto& range : parsed.ranges) {
      detail::charge(metadata_used, sizeof(LocatedTensor) + sha256_hex_chars,
                     request.max_metadata_bytes);
      detail::charge(metadata_used, range.tensor.name.size(),
                     request.max_metadata_bytes);
      detail::charge(metadata_used, shard.size(), request.max_metadata_bytes);
      const auto index = archive->catalog.size();
      const auto payload = bytes.subspan(prefix + as_size(range.data_offset),
                                         as_size(range.byte_count));
      if (!archive->locations
               .emplace(range.tensor.name, LocatedTensor{index, payload, shard})
               .second) {
        throw std::invalid_argument("tensor name appears in multiple shards");
      }
      archive->catalog.push_back(std::move(range.tensor));
    }
  }
  if (request.index) {
    auto parsed =
        detail::parse_metadata(archive->artifact(*request.index),
                               request.max_metadata_bytes - metadata_used);
    detail::charge(metadata_used, parsed.metadata_bytes,
                   request.max_metadata_bytes);
    const auto& index = parsed.value;
    if (!index.contains("weight_map") || !index.at("weight_map").is_object() ||
        index.at("weight_map").size() != archive->locations.size()) {
      throw std::invalid_argument("index tensor membership mismatch");
    }
    for (const auto& [key, value] : index.items()) {
      if (key != "weight_map" && key != "metadata") {
        throw std::invalid_argument("unsupported index field");
      }
      if (key == "metadata" && !value.is_object()) {
        throw std::invalid_argument("invalid index metadata");
      }
    }
    for (const auto& [name, shard] : index.at("weight_map").items()) {
      const auto found = archive->locations.find(name);
      if (!shard.is_string() || found == archive->locations.end() ||
          found->second.shard != shard.get_ref<const std::string&>()) {
        throw std::invalid_argument("index tensor shard mismatch");
      }
    }
  }
  return archive;
}
}  // namespace dolittle::tensors::formats::safetensors
