/*
 * Target-generation interface and implementation contract.
 * Axis R1: explicit token selection inputs and independent random state.
 * Inputs: one host FP32 logit per model vocabulary entry, selection settings,
 * and a caller-owned RNG stream. Output: one vocabulary ID. Generation owns
 * stop decisions and consumed-token positions; the sampler does neither.
 * Settings are copied into the run record; target and each future AV variant
 * receive separate seeds/streams. No process-global RNG or mutable defaults.
 *
 * Algorithm contract: a concrete sampler specifies logit filtering, ties,
 * normalization and draw mapping. Greedy selection consumes no random values.
 * The initial categorical implementation uses one mt19937_64 output per draw,
 * with its exact conversion described in categorical.h; a seed alone does not
 * promise matching PyTorch or trajectories across different GPU arithmetic.
 * Reject empty inputs, NaN/+infinity, all-masked logits and invalid settings
 * before changing RNG state. Negative infinity represents a masked candidate.
 * Invalid input -> invalid_argument; allocation failure propagates without
 * advancing the stream. Read-only logits are borrowed for the call.
 *
 * Design and cost: host logits are an explicit transfer boundary. At vocabulary
 * 151936, one row costs 303872 bytes in BF16 or 607744 in FP32 before widening,
 * plus queue completion latency. This is the proposed inspectable baseline;
 * measure transfer, synchronization and selection separately before optimizing.
 * Optimization: retain logits/selection scratch capacity across draws; keep all
 * transformer intermediates and KV storage on device.
 * Overlay: docs/architecture/file-mapping.md, "Sampling without a class hierarchy";
 * logical-overview.md, "2 Generate and capture".
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): explicit RNG ownership,
 * filtering settings and failure postconditions.
 * cpp-perf-guidelines GPU.1: name and budget the per-token host decision boundary.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <functional>
#include <span>

namespace dolittle::inference::sampling {
enum class SelectionMode { greedy, categorical };
struct Settings {
    SelectionMode mode;
    double temperature;
    std::size_t top_k; // Zero disables top-k filtering.
    double top_p;
};
void validate_settings(Settings);
using RandomStream = std::mt19937_64;
// Bound once at composition. Invoked once per draw, never per vocabulary item.
using TokenSelector = std::function<std::uint32_t(std::span<const float>, Settings, RandomStream &)>;
} // namespace dolittle::inference::sampling
