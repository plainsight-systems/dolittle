/*
 * Axis X3, BF16 kind: portable host storage and conversion, not GPU execution.
 * Contract: BFloat16 holds exactly the supplied 16 bits. to_float widens the
 * encoding into IEEE binary32 without rounding; from_float rounds finite
 * binary32 to BF16 nearest/even, preserving signed zero, subnormals and
 * infinities (finite overflow rounds to infinity). NaNs narrow to canonical
 * quiet 0x7fc0; raw read/write and widening do not canonicalize payloads.
 * No host rounding-mode dependency, allocation, exceptions or aliasing casts.
 * Require IEEE binary32 float at implementation compile time. read_le/write_le
 * explicitly decode/encode two little-endian bytes, including unaligned input;
 * their spans are borrowed for the call only. No inference arithmetic here.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Model computation":
 * move bits through bit_cast, detect exceptional encodings, widen by placement
 * in binary32 or narrow with a tie-even integer bias; handle NaNs first so
 * rounding never turns them into infinity. Serialize bytes explicitly.
 * Cost: O(1) per value, zero allocations. Optimization: capture paths copy
 * original bytes; conversion runs only where an explicit consumer needs it.
 * Unit contracts: exact widening/byte round trips, ties, signed zero and NaNs;
 * do not assert the bit-manipulation instruction sequence.
 *
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines C.183 (#ru-pun): use bit_cast, not union type punning.
 * cpp-guidelines I.1 (#ri-explicit): explicit rounding and byte order.
 * cpp-perf-guidelines GPU.1: avoid hidden conversion/readback in capture paths.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace dolittle::tensors {
struct BFloat16 {
  std::uint16_t bits;
};
[[nodiscard]] float to_float(BFloat16 value) noexcept;
[[nodiscard]] BFloat16 from_float(float value) noexcept;
[[nodiscard]] BFloat16 read_le(std::span<const std::byte, 2> bytes) noexcept;
void write_le(BFloat16 value, std::span<std::byte, 2> bytes) noexcept;
}  // namespace dolittle::tensors
