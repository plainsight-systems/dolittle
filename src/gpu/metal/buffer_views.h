/*
 * Implemented foundation, 2026-10-21; header design approved before
 * implementation. Axis X4: pure placement of native Metal views over one
 * logical allocation. Contract: backing_bytes is page-aligned; page_bytes is a
 * nonzero power of two; max_view_bytes is a positive page multiple. Every
 * required nonempty binding lies inside backing and must fit inside one
 * page-aligned native view. Each view borrows the same backing owner; overlaps
 * do not allocate or charge another copy of physical bytes. Invalid input ->
 * invalid_argument; overflow/unsupported operand span -> length_error;
 * bad_alloc propagates.
 *
 * Algorithm: metal-port.md, Memory and model residency; llama.cpp's native
 * buffer-view treatment, adapted to known tensor bindings.
 * 1. Cover the backing with adjacent views no larger than max_view_bytes.
 * 2. For each operand [offset,offset+bytes), find a containing existing view.
 *    If none exists, align its start down to a page and its end up to a page;
 *    reject if that extent exceeds max_view_bytes, otherwise add that view.
 * 3. Return all views and one view index per operand in original order. The
 *    backend creates native no-copy views before commands, retaining backing
 *    ownership until every view and its GPU uses have retired. Transfer code
 *    splits larger copies at view boundaries; an operation binding never does.
 *
 * Cost: baseline ceil(backing_bytes/max_view_bytes) views plus at most one per
 * crossing operand. Straightforward lookup is O(bindings*views) setup only.
 * Optimization: current model arenas fit the checked M3 allocation limit, so
 * the common case has one view and no overlapping metadata. This capability
 * does not introduce a generic tensor paging system or copies between views.
 * No Metal APIs, host mapping or physical allocation occurs in this planner.
 *
 * Both local corpora queried and rules read for header design:
 * cpp-guidelines I.7 (#ri-post), ES.103 (#res-overflow): binding containment
 * and checked alignment arithmetic. cpp-perf-guidelines GPU.9
 * (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md): aliases
 * share backing ownership; reuse follows completion. GPU.6
 * (guidelines/gpu/GPU.6-batch-tiny-gpu-work.md): resolve views once for stable
 * prepared commands, not per token. Core R.1 (#rr-raii) applies to the native
 * owner consuming this plan, not to a fictional GPU resource in this unit.
 */
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace dolittle::gpu::metal {
struct BufferExtent {
  std::size_t offset;
  std::size_t bytes;
};
struct BufferViewPlan {
  std::vector<BufferExtent> views;
  std::vector<std::size_t> binding_views;
};
[[nodiscard]] BufferViewPlan plan_buffer_views(
    std::size_t backing_bytes, std::size_t max_view_bytes,
    std::size_t page_bytes, std::span<const BufferExtent> bindings);
}  // namespace dolittle::gpu::metal
