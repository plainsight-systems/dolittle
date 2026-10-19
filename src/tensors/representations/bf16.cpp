/*
 * Axis X3, BF16 kind: exact host conversion and byte encoding.
 * Contract: implement bf16.h's rounding, NaN, infinity, subnormal and signed-zero
 * rules without allocation or exceptions. Raw read/write preserves every bit.
 * Algorithm: transformer overlay, "Model computation": assert binary32 host
 * representation, bit_cast to unsigned bits, handle NaNs before rounding, apply
 * nearest/even narrowing, widen by bit placement, and encode byte order explicitly.
 * Design and cost: constant work per scalar. Optimization: no full-buffer
 * conversion on capture; callers convert only values needed for comparison.
 * Unit verification exercises ties and representational boundaries, all raw-bit
 * round trips and exact widening. No device or reference model is invoked.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines C.183 (#ru-pun): bit_cast preserves valid object representation access.
 * cpp-perf-guidelines GPU.1: keep transfers/conversions explicit and bounded.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "tensors/representations/bf16.h"

#include <bit>
#include <limits>

namespace dolittle::tensors {

static_assert(sizeof(float) == sizeof(std::uint32_t) &&
              std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::radix == 2 &&
              std::numeric_limits<float>::digits == 24 &&
              std::numeric_limits<float>::max_exponent == 128,
              "BF16 conversion requires IEEE binary32 float");

float to_float(BFloat16 value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value.bits) << 16);
}

BFloat16 from_float(float value) noexcept {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return BFloat16{0x7fc0};
    }
    const auto rounded = bits + 0x7fffU + ((bits >> 16) & 1U);
    return BFloat16{static_cast<std::uint16_t>(rounded >> 16)};
}

BFloat16 read_le(std::span<const std::byte, 2> bytes) noexcept {
    const auto low = std::to_integer<std::uint16_t>(bytes[0]);
    const auto high = std::to_integer<std::uint16_t>(bytes[1]);
    return BFloat16{static_cast<std::uint16_t>(low | (high << 8))};
}

void write_le(BFloat16 value, std::span<std::byte, 2> bytes) noexcept {
    bytes[0] = static_cast<std::byte>(value.bits & 0xffU);
    bytes[1] = static_cast<std::byte>(value.bits >> 8);
}

}  // namespace dolittle::tensors
