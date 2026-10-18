/*
 * Header design; archive loading is not implemented.
 * Axis M2: named tensor payloads and verified source artifacts, with no model
 * parameter roles. Acquisition/network access remains M1.
 * Contract: an archive owns the backing storage for its immutable entries and
 * metadata artifacts. Names are unique; payload layout/extent has been checked.
 * tensor() and artifact() borrow data valid until archive destruction. Missing
 * names throw invalid_argument. Implementations must verify size and SHA-256
 * against supplied expected artifacts before exposing any bytes as verified.
 * SourceIdentity records repository/revision plus a deterministic digest of
 * the sorted (relative path, byte size, SHA-256) list, not a caller's label.
 * Metadata such as config/index is part of that same identity. No GPU work.
 * Callers keep files unchanged while open; hostile concurrent file mutation is
 * not supported. Failure exposes no partially verified archive.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Checkpoint loading
 * and binding": verify artifacts, decode named layouts, check extents, retain
 * backing storage, then let the selected model bind names to parameter roles.
 * Cost: verification reads artifact bytes once; mappings need not duplicate
 * all weights in a second host buffer. Optimization: index metadata once and
 * retain immutable byte views; packing belongs to its kernel consumer.
 *
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): archive owns files/mappings on failure too.
 * cpp-guidelines I.7 (#ri-post): verified identity and borrowed lifetime stated.
 * cpp-perf-guidelines CDSA.32: resolve stable named metadata once at load.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include "tensors/tensor.h"
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dolittle::tensors::formats {

struct Artifact {
    std::string relative_path;
    std::uint64_t byte_size;
    std::string sha256;
};

struct SourceIdentity {
    std::string repository;
    std::string revision;
    std::string artifact_digest;
};

struct NamedTensor {
    std::string name;
    Layout layout;
};

class TensorArchive {
public:
    virtual ~TensorArchive();
    [[nodiscard]] virtual const SourceIdentity& identity() const noexcept = 0;
    [[nodiscard]] virtual std::span<const NamedTensor> tensors() const noexcept = 0;
    [[nodiscard]] virtual HostView tensor(std::string_view name) const = 0;
    [[nodiscard]] virtual std::span<const std::byte> artifact(
        std::string_view relative_path) const = 0;

    TensorArchive(const TensorArchive&) = delete;
    TensorArchive& operator=(const TensorArchive&) = delete;

protected:
    TensorArchive();
};

}  // namespace dolittle::tensors::formats
