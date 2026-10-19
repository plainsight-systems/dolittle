/*
 * Axis X3, BF16 kind: exact host conversion and byte encoding.
 * Contract: implement bf16.h's rounding, NaN, infinity, subnormal and
 * signed-zero rules without allocation or exceptions. Raw read/write preserves
 * every bit. Algorithm: transformer overlay, "Model computation": assert
 * binary32 host representation, bit_cast to unsigned bits, handle NaNs before
 * rounding, apply nearest/even narrowing, widen by bit placement, and encode
 * byte order explicitly. Design and cost: constant work per scalar.
 * Optimization: no full-buffer conversion on capture; callers convert only
 * values needed for comparison. Unit verification exercises ties and
 * representational boundaries, all raw-bit round trips and exact widening. No
 * device or reference model is invoked. Overlay:
 * docs/architecture/transformer-execution.md. Corpus consultation,
 * implementation, 2026-10-18: cpp-guidelines C.183 (#ru-pun): bit_cast
 * preserves valid object representation access. cpp-perf-guidelines GPU.1: keep
 * transfers/conversions explicit and bounded. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "tensors/representations/bf16.h"

#include <bit>
#include <limits>

namespace dolittle::tensors {
inline constexpr int binary32_digits = 24;
inline constexpr int binary32_max_exponent = 128;
inline constexpr unsigned bf16_shift = 16;
inline constexpr std::uint32_t magnitude_mask = 0x7fffffffU;
inline constexpr std::uint32_t infinity_bits = 0x7f800000U;
inline constexpr std::uint16_t quiet_nan_bits = 0x7fc0;
inline constexpr std::uint32_t rounding_bias = 0x7fffU;
inline constexpr unsigned byte_bits = 8;
inline constexpr unsigned byte_mask = 0xffU;

static_assert(sizeof(float) == sizeof(std::uint32_t) &&
                  std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::radix == 2 &&
                  std::numeric_limits<float>::digits == binary32_digits &&
                  std::numeric_limits<float>::max_exponent ==
                      binary32_max_exponent,
              "BF16 conversion requires IEEE binary32 float");

float to_float(BFloat16 value) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value.bits)
                              << bf16_shift);
}

BFloat16 from_float(float value) noexcept {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & magnitude_mask) > infinity_bits) {
    return BFloat16{quiet_nan_bits};
  }
  const auto rounded = bits + rounding_bias + ((bits >> bf16_shift) & 1U);
  return BFloat16{static_cast<std::uint16_t>(rounded >> bf16_shift)};
}

BFloat16 read_le(std::span<const std::byte, 2> bytes) noexcept {
  const auto low = std::to_integer<std::uint16_t>(bytes.front());
  const auto high = std::to_integer<std::uint16_t>(bytes.back());
  return BFloat16{static_cast<std::uint16_t>(low | (high << byte_bits))};
}

void write_le(BFloat16 value, std::span<std::byte, 2> bytes) noexcept {
  bytes.front() = static_cast<std::byte>(value.bits & byte_mask);
  bytes.back() = static_cast<std::byte>(value.bits >> byte_bits);
}

}  // namespace dolittle::tensors
