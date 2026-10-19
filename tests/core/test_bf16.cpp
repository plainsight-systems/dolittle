/*
 * Axis D3: isolated verification of X3 BF16 representation contracts.
 * Contract: fixed in-memory values exercise bit-preserving serialization and
 * exact widening, nearest-even narrowing, NaN canonicalization and signed zero.
 * Failure throws through the unit helper; no files, GPU or reference model.
 * Algorithm: docs/architecture/transformer-execution.md, "Model computation":
 * enumerate all stored encodings, compare widening with binary32 placement,
 * and probe rounding neighborhoods at subnormal, normal and overflow boundaries.
 * Design/cost: 65,536 scalar encodings, constant scratch storage. Exact bits are
 * mathematical/serialization contracts; the instruction sequence is unobserved.
 * Optimization: no full tensor or repeated allocation is needed for coverage.
 * Corpus consultation, test design and implementation, 2026-10-18:
 * cpp-guidelines C.183 (#ru-pun): inspect representations with bit_cast.
 * cpp-guidelines I.1 (#ri-explicit): rounding and byte order are explicit inputs.
 * cpp-perf-guidelines GPU.1: numerical host verification needs no GPU readback.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "tensors/representations/bf16.h"
#include "tests/support/check.h"

#include <array>
#include <bit>
#include <cstdint>

using namespace dolittle::tensors;
using dolittle::tests::check;

int main() {
    std::array<std::byte, 3> storage{};
    auto bytes = std::span<std::byte>(storage).subspan<1, 2>();
    for (std::uint32_t bits = 0; bits <= 0xffffU; ++bits) {
        const BFloat16 value{static_cast<std::uint16_t>(bits)};
        write_le(value, bytes);
        check(std::to_integer<unsigned>(bytes[0]) == (bits & 0xffU) &&
              std::to_integer<unsigned>(bytes[1]) == (bits >> 8),
              "explicit little-endian encoding");
        check(read_le(bytes).bits == bits, "raw encoding round trip");
        const auto widened = to_float(value);
        check(std::bit_cast<std::uint32_t>(widened) == bits << 16,
              "exact widening preserves every stored bit");
        const bool nan = (bits & 0x7fffU) > 0x7f80U;
        check(from_float(widened).bits == (nan ? 0x7fc0U : bits),
              "exact representable values survive narrowing; NaNs canonicalize");
    }

    struct RoundingCase { std::uint32_t input; std::uint16_t expected; };
    constexpr std::array cases{
        RoundingCase{0x00000000, 0x0000}, RoundingCase{0x80000000, 0x8000},
        RoundingCase{0x00008000, 0x0000}, RoundingCase{0x80008000, 0x8000},
        RoundingCase{0x00008001, 0x0001}, RoundingCase{0x80008001, 0x8001},
        RoundingCase{0x00018000, 0x0002}, RoundingCase{0x80018000, 0x8002},
        RoundingCase{0x007f8000, 0x0080}, RoundingCase{0x807f8000, 0x8080},
        RoundingCase{0x3f807fff, 0x3f80}, RoundingCase{0x3f808000, 0x3f80},
        RoundingCase{0x3f808001, 0x3f81}, RoundingCase{0x3f818000, 0x3f82},
        RoundingCase{0xbf808000, 0xbf80}, RoundingCase{0xbf818000, 0xbf82},
        RoundingCase{0x7f7f7fff, 0x7f7f}, RoundingCase{0x7f7f8000, 0x7f80},
        RoundingCase{0xff7f8000, 0xff80}, RoundingCase{0x7f800000, 0x7f80},
        RoundingCase{0xff800000, 0xff80}, RoundingCase{0x7f800001, 0x7fc0},
        RoundingCase{0xff800001, 0x7fc0}, RoundingCase{0x7fffffff, 0x7fc0},
        RoundingCase{0xffffffff, 0x7fc0}};
    for (const auto& test : cases) {
        check(from_float(std::bit_cast<float>(test.input)).bits == test.expected,
              "nearest-even and exceptional boundary encoding");
    }
}
