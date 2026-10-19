/*
 * Axis D3: isolated verification of X5 mathematical operation contracts.
 * Contract: exercise shape, numerical mode, constants and distinct tensor access
 * roles with small in-memory metadata. No kernels, devices, files or subprocesses.
 * Failed expectations throw through the test helper; diagnostic wording and
 * backend launch geometry are deliberately outside these assertions.
 * Algorithm: docs/architecture/transformer-execution.md, "One Qwen3 transformer
 * block" and "Preparing the first execution recipe": admit each declared
 * equation, check its dependency roles, then reject incompatible descriptors.
 * Design/cost: small fixed tables plus gathered row lists. Optimization: verify
 * preparation contracts without constructing tensor payloads or GPU resources.
 * Corpus consultation, test design and implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): assert mathematical results and error classes.
 * cpp-guidelines ES.103 (#res-overflow): cover shape and position overflow.
 * cpp-perf-guidelines GPU.2: verify dense logical addresses, not physical tiling.
 * Source: guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 */
#include "kernels/operations.h"
#include "tests/support/check.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>

using namespace dolittle::kernels;
using namespace dolittle::tensors;
using dolittle::tests::check;
using dolittle::tests::expect_throws;

namespace {

Layout dense(ElementType type, std::initializer_list<std::size_t> shape) {
    return contiguous_layout(type, std::span<const std::size_t>(shape.begin(), shape.size()));
}

void accepted(const Operation& op, const std::vector<Layout>& layouts,
              ComputationMode mode, std::initializer_list<std::size_t> reads,
              std::size_t output) {
    const auto uses = validate_operation(op, layouts, mode);
    check(uses.size() == reads.size() + 1, "one role per distinct tensor");
    for (const auto id : reads) {
        check(std::ranges::count_if(uses, [id](const OperandUse& use) {
            return use.tensor.value == id && use.access == OperandAccess::read;
        }) == 1, "every input has exactly one read role");
    }
    check(std::ranges::count_if(uses, [output](const OperandUse& use) {
        return use.tensor.value == output && use.access == OperandAccess::write;
    }) == 1, "output has exactly one write role");
}

}  // namespace

int main() {
    for (const auto type : {ElementType::bfloat16, ElementType::float32}) {
        const ComputationMode mode{type, ElementType::float32};
        accepted(Embedding{{0}, {1}, {2}},
                 {dense(ElementType::uint32, {3}), dense(type, {11, 8}), dense(type, {3, 8})},
                 mode, {0, 1}, 2);
        accepted(Projection{{0}, {1}, {2}},
                 {dense(type, {3, 8}), dense(type, {12, 8}), dense(type, {3, 3, 4})},
                 mode, {0, 1}, 2);
        accepted(Projection{{0}, {1}, {2}},
                 {dense(type, {3, 3, 4}), dense(type, {8, 12}), dense(type, {3, 8})},
                 mode, {0, 1}, 2);
        accepted(RmsNorm{{0}, {1}, {2}, 1e-6F},
                 {dense(type, {3, 2, 4}), dense(type, {4}), dense(type, {3, 2, 4})},
                 mode, {0, 1}, 2);
        accepted(Rotary{{0}, {1}, 19, 1'000'000.0},
                 {dense(type, {3, 2, 4}), dense(type, {3, 2, 4})}, mode, {0}, 1);
        accepted(CausalAttention{{0}, {1}, {2}, {3}},
                 {dense(type, {3, 4, 8}), dense(type, {3, 2, 8}),
                  dense(type, {3, 2, 8}), dense(type, {3, 4, 8})}, mode, {0, 1, 2}, 3);
        accepted(Add{{0}, {0}, {1}}, {dense(type, {3, 8}), dense(type, {3, 8})},
                 mode, {0}, 1);
        accepted(SiluMultiply{{0}, {1}, {2}},
                 {dense(type, {3, 8}), dense(type, {3, 8}), dense(type, {3, 8})},
                 mode, {0, 1}, 2);
        accepted(GatherRows{{0}, {1}, {0, 2}},
                 {dense(type, {3, 2, 4}), dense(type, {2, 2, 4})}, mode, {0}, 1);
        accepted(Add{{0}, {0}, {1}}, {dense(type, {}), dense(type, {})}, mode, {0}, 1);

        auto singleton = dense(type, {2, 1, 4});
        singleton.byte_strides[1] = 0;
        accepted(Add{{0}, {0}, {1}}, {singleton, dense(type, {2, 1, 4})}, mode, {0}, 1);
        auto wide_singleton = dense(type, {1, 4});
        wide_singleton.byte_strides[0] = 100 * element_bytes(type);
        accepted(Add{{0}, {0}, {1}}, {wide_singleton, dense(type, {1, 4})}, mode, {0}, 1);
    }

    constexpr auto f32 = ElementType::float32;
    constexpr ComputationMode mode{f32, f32};
    const auto reject = [&](const Operation& op, const std::vector<Layout>& layouts) {
        expect_throws<std::invalid_argument>([&] {
            (void)validate_operation(op, layouts, mode);
        });
    };
    const std::vector<Layout> pairs{dense(f32, {2, 4}), dense(f32, {2, 4})};
    reject(Add{{0}, {0}, {7}}, pairs);
    reject(Add{{0}, {1}, {0}}, pairs);
    reject(Add{{0}, {0}, {1}}, {dense(f32, {2, 4}), dense(f32, {4, 2})});
    reject(SiluMultiply{{0}, {1}, {2}},
           {dense(f32, {2, 4}), dense(f32, {1, 4}), dense(f32, {2, 4})});
    reject(Add{{0}, {0}, {1}},
           {dense(ElementType::bfloat16, {2, 4}), dense(f32, {2, 4})});
    reject(Add{{0}, {0}, {1}}, {dense(f32, {0, 4}), dense(f32, {0, 4})});
    auto padded = dense(f32, {2, 4});
    padded.byte_strides[0] += 4;
    reject(Add{{0}, {0}, {1}}, {padded, dense(f32, {2, 4})});
    auto broadcast = dense(f32, {2, 4});
    broadcast.byte_strides[0] = 0;
    reject(Add{{0}, {0}, {1}}, {broadcast, dense(f32, {2, 4})});
    for (const ComputationMode invalid : {ComputationMode{f32, ElementType::bfloat16},
                                          ComputationMode{ElementType::uint32, f32}}) {
        expect_throws<std::invalid_argument>([&] {
            (void)validate_operation(Add{{0}, {0}, {1}}, pairs, invalid);
        });
    }

    reject(Embedding{{0}, {1}, {2}},
           {dense(f32, {2}), dense(f32, {11, 4}), dense(f32, {2, 4})});
    reject(Embedding{{0}, {1}, {2}},
           {dense(ElementType::uint32, {2}), dense(f32, {11, 4}), dense(f32, {3, 4})});
    reject(Projection{{0}, {1}, {2}},
           {dense(f32, {2, 3, 4}), dense(f32, {4, 8}), dense(f32, {2, 4})});
    reject(Projection{{0}, {1}, {2}},
           {dense(f32, {2, 8}), dense(f32, {4, 8}), dense(f32, {1, 8})});
    reject(Projection{{0}, {1}, {2}},
           {dense(f32, {8}), dense(f32, {4, 8}), dense(f32, {4})});
    reject(RmsNorm{{0}, {1}, {2}, 1e-6F},
           {dense(f32, {2, 4}), dense(f32, {8}), dense(f32, {2, 4})});
    for (float epsilon : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        reject(RmsNorm{{0}, {1}, {2}, epsilon},
               {dense(f32, {2, 4}), dense(f32, {4}), dense(f32, {2, 4})});
    }
    reject(Rotary{{0}, {1}, 0, 1000.0},
           {dense(f32, {2, 3, 5}), dense(f32, {2, 3, 5})});
    for (double theta : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        reject(Rotary{{0}, {1}, 0, theta},
               {dense(f32, {2, 3, 4}), dense(f32, {2, 3, 4})});
    }
    reject(CausalAttention{{0}, {1}, {2}, {3}},
           {dense(f32, {3, 3, 8}), dense(f32, {3, 2, 8}),
            dense(f32, {3, 2, 8}), dense(f32, {3, 3, 8})});
    reject(CausalAttention{{0}, {1}, {2}, {3}},
           {dense(f32, {3, 4, 8}), dense(f32, {3, 2, 8}),
            dense(f32, {3, 2, 4}), dense(f32, {3, 4, 8})});
    for (const auto& rows : {std::vector<std::size_t>{1, 1},
                            std::vector<std::size_t>{2, 1},
                            std::vector<std::size_t>{0, 3}}) {
        reject(GatherRows{{0}, {1}, rows}, {dense(f32, {3, 4}), dense(f32, {2, 4})});
    }
    reject(GatherRows{{0}, {1}, {}}, {dense(f32, {3, 4}), dense(f32, {0, 4})});
    reject(GatherRows{{0}, {1}, {0, 2}}, {dense(f32, {3, 4}), dense(f32, {2, 5})});

    const auto maximum = std::numeric_limits<std::size_t>::max();
    expect_throws<std::length_error>([&] {
        const std::vector<Layout> layouts{dense(f32, {2, 1, 4}), dense(f32, {2, 1, 4})};
        (void)validate_operation(Rotary{{0}, {1}, maximum, 1.0}, layouts, mode);
    });
    // The final position may equal size_t's maximum; only overflow is forbidden.
    accepted(Rotary{{0}, {1}, maximum - 1, 1.0},
             {dense(f32, {2, 1, 4}), dense(f32, {2, 1, 4})}, mode, {0}, 1);
    expect_throws<std::length_error>([&] {
        const std::vector<Layout> layouts{
            Layout{f32, {2, maximum}, {4, 4}}, dense(f32, {2, 4})};
        (void)validate_operation(Add{{0}, {0}, {1}}, layouts, mode);
    });
}
