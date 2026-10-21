/*
 * Implementation: prepared Metal target-kernel geometry.
 *
 * Axis X5: Metal kernel geometry and bounded on-chip storage. Shared operation
 * descriptors remain in kernels/operations.h; these plans describe only how
 * the Metal backend executes them. Dimensions count elements, budgets bytes.
 * Device limits are captured during preparation, not queried per token.
 *
 * Contract: dimensions are positive and all products/rounded sizes checked.
 * Plans cover every active row/feature exactly once, guard tails, and fit the
 * pipeline's thread/SIMD width and device threadgroup-memory limits.
 * Unsupported layouts/limits -> invalid_argument; arithmetic overflow ->
 * length_error. Planning is pure and allocates no tensor, pipeline or command
 * resources. Row tiles may change traffic/parallelism, never reduction
 * ownership/order. Threadgroup bytes are reported before encoding and included
 * in capability checks. Device scratch is declared through
 * PreparedOperation::workspace().
 *
 * Algorithm: metal-port.md, Target kernel design; transformer-execution.md,
 * One Qwen3 transformer block. Projection selects 16/32 partials from I/O,
 * then packs independent output columns and prompt rows into a group. Attention
 * selects bounded retained-score or tiled recomputation storage from admitted
 * context/head width and native limits. RMS uses staged rows only when its
 * whole scratch fits. Elementwise work uses guarded contiguous vector loads.
 * No plan value depends on a sampled token or a host read of device position.
 * The implementation files below describe each kernel's complete arithmetic.
 *
 * Optimization: choose tiles once, use SIMD shuffles for fixed local trees,
 * and reserve threadgroup memory for actual shared reuse. The initial
 * projection candidates use 128 threads, 128 input features and 4/8 prompt
 * rows. Benchmark both on M3; do not copy CUDA's winning row tile as an assumed
 * Metal optimum. Attention admits a maximum 4096-score retained path; larger
 * visible ranges use bounded recomputation. Its input capacity does not change
 * the sum tree. Cost: O(1) plan construction; kernels report their own O(work)
 * costs.
 *
 * Header-design and implementation consultation, 2026-10-21: both local corpora
 * searched and cited rules read. Core I.7 (#ri-post) fixes numerical/completion
 * results; SL.con.3 (#rsl-bounds) requires checked extents before native
 * indexing; R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.2: map adjacent feature addresses to adjacent SIMD lanes.
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md
 * GPU.5: bound shared tiles and account for barriers/occupancy.
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md
 * GDSA.2: select arithmetic partitions independently of batch size.
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "tensors/representations/representation.h"

namespace dolittle::kernels::metal {
struct KernelLimits {
  std::size_t simd_width{};
  std::size_t max_threads{};
  std::size_t max_threadgroup_bytes{};
};
struct ProjectionShape {
  std::size_t rows{}, input_width{}, output_width{};
};
struct ProjectionPlan {
  std::size_t partials{}, row_tile{}, outputs_per_group{};
  std::size_t threads{}, threadgroup_bytes{};
};
struct AttentionShape {
  std::size_t rows{}, query_heads{}, kv_heads{}, width{}, capacity{};
};
struct AttentionPlan {
  std::size_t query_tile{}, key_tile{}, retained_keys{};
  std::size_t threads{}, threadgroup_bytes{};
};
[[nodiscard]] ProjectionPlan plan_projection(ProjectionShape,
                                             tensors::ElementType,
                                             KernelLimits);
[[nodiscard]] AttentionPlan plan_attention(AttentionShape, tensors::ElementType,
                                           KernelLimits);
}  // namespace dolittle::kernels::metal

// Identical byte layout to OperationArgs in common.metal. Addresses are private
// backend values; every pointed-to allocation is also retained in bindings.
namespace dolittle::kernels::metal {
inline constexpr std::size_t address_slots = 8;
inline constexpr std::size_t position_slot = 5;
inline constexpr std::size_t constant_bits_slot = 7;
inline constexpr std::size_t operation_argument_bytes = 128;
struct OperationArgs {
  std::array<std::uint64_t, address_slots> address{};
  std::array<std::uint64_t, 4> dimensions{};
  std::uint64_t first{}, capacity{};
  std::uint32_t type{}, flags{};
  float scalar{};
  std::uint32_t row_tile{};
};
static_assert(sizeof(OperationArgs) == operation_argument_bytes);
}  // namespace dolittle::kernels::metal
