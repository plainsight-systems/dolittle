/*
 * Header design; implementations and numerical validation are pending.
 * Axis: X3, numerical representation. Shared declarations, no model constants.
 *
 * Contract: an element type describes stored bits, not a host-native pointer
 * type. ComputationMode declares activation storage and reduction accumulation
 * separately. These are the first supported encodings, not a claim that every
 * backend implements every combination. Plan preparation rejects unsupported
 * modes before allocation/dispatch; it never substitutes another precision.
 * uint32 is an index-buffer encoding only, never an activation or accumulator.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Model dimensions and
 * notation", "Configurable observation points", and "Cost model". Carry the
 * selected numerical mode into planning; use actual observed storage type to
 * size captures; preserve bits on capture; convert separately for comparisons.
 * Optimization: no implicit conversion or buffer ownership in these types.
 * Storage byte counts and accumulation choices stay explicit at consumers.
 *
 * Corpus guidance queried for header design, 2026-10-17 and 2026-10-18:
 * cpp-guidelines I.4 (#ri-typed): distinguish storage type from accumulation.
 * cpp-guidelines I.7 (#ri-post): a capture reports its original representation.
 * cpp-perf-guidelines GPU.1: budget bytes using real storage type; do not hide
 * conversions/readbacks. MCP source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include <cstdint>

namespace dolittle::tensors {

enum class ElementType : std::uint8_t { bfloat16, float32, uint32 };

struct ComputationMode {
  ElementType activation_storage;
  ElementType reduction_accumulation;
};

}  // namespace dolittle::tensors
