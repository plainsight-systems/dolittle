/*
 * Header design; safetensors parsing and hash verification are pending.
 * Axis M2, safetensors kind. Own file/mapping resources behind TensorArchive.
 * Contract: OpenRequest names an explicit shard set, optional index, and
 * expected artifacts including model configuration. Paths must be relative,
 * unique and resolve inside root; every opened file must appear in artifacts.
 * Check actual file sizes/digests, bounded header sizes, duplicate JSON keys,
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
 * occupy no bytes). Offsets are relative to that payload region. Limits apply
 * before allocation; results own names/layouts and retain no span. It opens
 * nothing and verifies no file/hash. open performs framing/integrity checks
 * before calling this same parser. Unit tests use small in-memory headers.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Checkpoint loading
 * and binding": verify declared files, parse bounded headers/index, validate
 * named extents, and expose immutable source bytes to model binding.
 * Cost: O(artifact bytes) hashing plus O(tensor count) metadata; max_header_bytes
 * is per file, max_metadata_bytes bounds aggregate decoded metadata, and
 * max_tensor_count bounds catalog growth. Optimization: map weight payloads
 * without eagerly converting/copying them; pin only bounded transfer staging.
 *
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): partial mappings are reclaimed on exceptions.
 * cpp-guidelines I.7 (#ri-post): only fully checked archives escape loading.
 * cpp-perf-guidelines GPU.1: mapped input and bounded upload staging avoid
 * repeated host/device round trips. Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include "tensors/formats/format.h"
#include <filesystem>
#include <memory>
#include <optional>

namespace dolittle::tensors::formats::safetensors {

struct HeaderLimits {
    std::size_t max_header_bytes;
    std::size_t max_metadata_bytes;
    std::size_t max_tensor_count;
};
struct TensorRange {
    NamedTensor tensor;
    std::uint64_t data_offset;
    std::uint64_t byte_count;
};
[[nodiscard]] std::vector<TensorRange> parse_header(
    std::span<const std::byte> header_json, std::uint64_t data_bytes,
    HeaderLimits limits);

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
};

[[nodiscard]] std::unique_ptr<TensorArchive> open(const OpenRequest& request);

}  // namespace dolittle::tensors::formats::safetensors
