/*
 * Axis X1: pure scratch admission/placement within ordered forward execution.
 * Contract: each request has a unique ID, byte size, nonzero power-of-two
 * alignment and inclusive [first_step,last_step] with first<=last. All
 * arithmetic is checked. Simultaneously live nonempty resources never overlap
 * in bytes; each placement meets its alignment and fits arena_bytes<=budget.
 * Sharing a step means overlapping lifetimes, including producer/read steps.
 * Zero-byte requests retain an ID but occupy no range. Empty input yields
 * zero arena bytes with alignment 1. Results own metadata and borrow no input.
 * Invalid requests -> invalid_argument; overflow/budget -> length_error;
 * allocation failure propagates. No files, device calls or tensor allocations.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Model computation":
 * validate extents/lifetimes, order by first use, release intervals strictly
 * before the next first use, reuse an aligned free range or extend the arena,
 * then verify placement invariants. Callers include observation-copy readers
 * when computing last use. Separate weight/output arenas are not aliased here.
 * Cost: O(R) metadata; a simple free-range search may take O(R^2) setup work.
 * Optimization: pay once per prepared shape and reuse placements. Optimal
 * packing is not promised; budget failure describes this placement policy,
 * not proof that no packing exists. Tests check invariants, never fixed offsets
 * or allocator choices. Allocation granularity/driver overhead is budgeted by
 * the executor in addition to these logical extents.
 *
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit): lifetimes, alignments and budget are
 * inputs. cpp-guidelines I.7 (#ri-post): non-overlap and containment are
 * postconditions. cpp-guidelines ES.103 (#res-overflow): check offset and
 * extent arithmetic. cpp-perf-guidelines GPU.9: reuse bytes only after every
 * last reader. Source:
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines MEM.11: prepare offsets once from explicit lifetimes.
 * Source: guidelines/memory/MEM.11-plan-arena-offsets-from-lifetimes.md.
 * The accepted zero-byte identity contract takes precedence over that rule's
 * example restriction to positive sizes; placement remains first-use ordered.
 */
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace dolittle::inference {
struct ResourceId {
  std::size_t value;
};
struct ResourceLifetime {
  ResourceId id;
  std::size_t bytes;
  std::size_t alignment;
  std::size_t first_step;
  std::size_t last_step;
};
struct ResourcePlacement {
  ResourceId id;
  std::size_t offset;
  std::size_t bytes;
};
struct StoragePlan {
  std::vector<ResourcePlacement> placements;
  std::size_t arena_bytes;
  std::size_t arena_alignment;
};
[[nodiscard]] StoragePlan plan_storage(
    std::span<const ResourceLifetime> resources, std::size_t budget_bytes);
}  // namespace dolittle::inference
