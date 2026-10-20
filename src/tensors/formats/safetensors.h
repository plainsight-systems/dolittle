/*
 * Safetensors decoding and verified mapped archive loading.
 * Axis M2, safetensors kind. Own file/mapping resources behind TensorArchive.
 * Contract: OpenRequest names an explicit shard set, optional index, and
 * expected artifacts including model configuration. Paths must be relative,
 * unique and resolve inside root; every opened file must appear in artifacts.
 * Check actual sizes and, by default, SHA-256; prepared_setup instead requires
 * the caller to have validated a setup receipt for these unchanged files.
 * Both modes check bounded header sizes, duplicate JSON keys,
 * supported dtypes, tensor shapes, checked offsets and non-overlapping payload
 * ranges before publishing the archive. When an index is supplied, its tensor
 * to shard mapping must exactly match decoded entries. No inferred globbing,
 * missing-shard fallback, pickle execution or automatic dtype conversion.
 * Reject malformed/unsupported data with invalid_argument, overflow/limits
 * with length_error, I/O or digest mismatch with runtime_error; propagate
 * bad_alloc. Failure releases partial mappings. Request strings are copied;
 * archive views obey format.h. Caller must keep backing files unchanged.
 * parse_header is the pure decoding boundary: its span is the JSON header
 * alone, not the eight-byte file prefix or tensor payload. data_bytes is the
 * declared payload-region size. It validates duplicate names/keys, dtypes,
 * shapes and a complete, non-overlapping payload partition (empty tensors
 * occupy no bytes). Offsets are relative to that payload region. Input, decoded
 * node/string accounting and output sizes are bounded before their growth;
 * results own names/layouts and retain no span. It opens nothing and verifies
 * no file/hash. open performs framing/integrity checks before calling this same
 * parser. Unit tests use small in-memory headers.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Checkpoint loading
 * and binding": verify declared files, parse bounded headers/index, validate
 * named extents, and expose immutable source bytes to model binding.
 * Cost: sha256 mode adds O(artifact bytes) hashing; prepared_setup reads only
 * metadata and file sizes, with O(tensor count) metadata;
 * max_header_bytes is per file. max_metadata_bytes bounds aggregate
 * source/node/string/catalog accounting (not exact allocator RSS); JSON nesting
 * is capped at 64, and max_tensor_count bounds catalog growth. Optimization:
 * map weight payloads without eagerly converting/copying them; pin only bounded
 * transfer staging. Range/catalog ordering takes O(T log T) setup work for T
 * tensors. Artifact digest v1 hashes "dolittle-artifacts-v1", then sorted paths
 * encoded as u64-LE byte length, UTF-8 path bytes, u64-LE size and 64 lowercase
 * hex bytes.
 *
 * Corpus consultation, implementation (full rules read), 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): partial mappings are reclaimed on exceptions.
 * cpp-guidelines I.7 (#ri-post): only fully checked archives escape loading.
 * cpp-perf-guidelines GPU.1: mapped input and bounded upload staging avoid
 * repeated host/device round trips. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 *
 * Startup verification design, 2026-10-19: verification_workers selects 1..8
 * bounded concurrent file hashes (default 1) in sha256 mode. Each file uses
 * SHA-256; concurrency changes neither digests nor canonical source identity.
 * Map and size-check each artifact, hash disjoint immutable views in bounded
 * batches, join all checks, then decode metadata and publish. No archive
 * escapes on a mismatch. All workers join before mappings are destroyed, even
 * on error. Optimization: overlap independent artifact checks; O(total bytes)
 * CPU work, at most verification_workers digest states/threads, no weight-sized
 * copies. cpp-guidelines CP.23 (#rconc-join): worker scopes cannot outlive
 * mapped views. cpp-perf-guidelines CDSA.32: parallelize one-time preparation,
 * retain its immutable result;
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
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>

#include "tensors/formats/format.h"

namespace dolittle::tensors::formats::safetensors {

struct HeaderLimits {
  std::size_t max_header_bytes;
  std::size_t max_metadata_bytes;
  std::size_t max_tensor_count;
};
struct TensorRange {
  NamedTensor tensor;
  std::uint64_t data_offset{};
  std::uint64_t byte_count{};
};
[[nodiscard]] std::vector<TensorRange> parse_header(
    std::span<const std::byte> header_json, std::uint64_t data_bytes,
    HeaderLimits limits);

// prepared_setup is a trusted caller precondition, not a content check.
// The source launcher validates the persisted setup receipt before selecting
// it.
enum class Integrity : std::uint8_t { sha256, prepared_setup };

struct OpenRequest {
  std::filesystem::path root;
  std::string repository;
  std::string revision;
  std::vector<Artifact> artifacts;
  std::vector<std::string> shards;
  std::optional<std::string> index;
  std::size_t max_header_bytes;
  std::size_t max_metadata_bytes;
  std::size_t max_tensor_count;
  std::size_t verification_workers = 1;
  Integrity integrity = Integrity::sha256;
};

[[nodiscard]] std::unique_ptr<TensorArchive> open(const OpenRequest& request);

}  // namespace dolittle::tensors::formats::safetensors
