/*
 * Axis X1: pure reusable scratch placement.
 * Contract: implement storage_plan.h with checked align-up and non-overlap for
 * inclusive lifetimes. Every requested ID is represented exactly once. A budget
 * failure returns no partial plan. Empty/zero-byte resources follow the header.
 * Algorithm: transformer overlay, "Model computation": validate IDs/ranges and
 * alignments; order starts; retire intervals whose last use is strictly earlier;
 * choose an aligned free range or grow the arena; coalesce free ranges and verify
 * containment/alignment/lifetime non-overlap before returning owned metadata.
 * Design and cost: simple O(R^2) worst-case preparation, O(R) metadata, no device
 * allocation. Optimization: reuse one arena across calls and non-overlapping
 * values. No claim of optimal packing. Unit tests assert invariants under varied
 * input order and lifetimes, never exact offsets, free-list order or pool size
 * unless required by an explicit admission bound.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): enforce the stated results and failure boundaries.
 * cpp-guidelines ES.103 (#res-overflow): guard align-up and extent addition.
 * cpp-perf-guidelines GPU.9: prepare storage once and preserve last-use lifetimes.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines MEM.11: replay precomputed offsets from known lifetimes;
 * preserve the reviewed first-use policy and zero-byte identity exception.
 * Source: guidelines/memory/MEM.11-plan-arena-offsets-from-lifetimes.md.
 */
#include "inference/storage_plan.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace dolittle::inference {
namespace {
struct FreeRange {
    std::size_t begin;
    std::size_t end;
};

std::optional<std::size_t> aligned_offset(std::size_t value,
                                        std::size_t alignment) noexcept {
    const auto remainder = value & (alignment - 1);
    const auto padding = remainder == 0 ? 0 : alignment - remainder;
    if (padding > std::numeric_limits<std::size_t>::max() - value) {
        return std::nullopt;
    }
    return value + padding;
}

// Ranges are sorted, disjoint and coalesced, with exclusive end offsets.
void release_range(std::vector<FreeRange>& ranges, FreeRange released) {
    if (released.begin == released.end) {
        return;
    }
    auto at = std::lower_bound(ranges.begin(), ranges.end(), released.begin,
                              [](const FreeRange& range, std::size_t begin) {
                                  return range.begin < begin;
                              });
    if (at != ranges.begin() && (at - 1)->end == released.begin) {
        --at;
        at->end = released.end;
    } else {
        at = ranges.insert(at, released);
    }
    if (at + 1 != ranges.end() && at->end == (at + 1)->begin) {
        at->end = (at + 1)->end;
        ranges.erase(at + 1);
    }
}

bool lifetimes_overlap(const ResourceLifetime& a,
                       const ResourceLifetime& b) noexcept {
    return a.first_step <= b.last_step && b.first_step <= a.last_step;
}

void verify_plan(std::span<const ResourceLifetime> resources,
                 const StoragePlan& plan, std::size_t budget_bytes) {
    if (plan.arena_bytes > budget_bytes) {
        throw std::logic_error("scratch placement exceeded its budget");
    }
    for (std::size_t i = 0; i < resources.size(); ++i) {
        const auto& place = plan.placements[i];
        const auto& resource = resources[i];
        if (place.id.value != resource.id.value || place.bytes != resource.bytes ||
            place.offset % resource.alignment != 0 ||
            plan.arena_alignment % resource.alignment != 0 ||
            place.offset > plan.arena_bytes ||
            place.bytes > plan.arena_bytes - place.offset) {
            throw std::logic_error("scratch placement violated containment or alignment");
        }
        if (place.bytes == 0) {
            continue;
        }
        for (std::size_t j = 0; j < i; ++j) {
            const auto& other = plan.placements[j];
            if (other.bytes != 0 && lifetimes_overlap(resource, resources[j]) &&
                place.offset < other.offset + other.bytes &&
                other.offset < place.offset + place.bytes) {
                throw std::logic_error("scratch placement aliased live resources");
            }
        }
    }
}
}  // namespace

StoragePlan plan_storage(std::span<const ResourceLifetime> resources,
                         std::size_t budget_bytes) {
    StoragePlan plan{{}, 0, 1};
    std::vector<std::size_t> order(resources.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    for (const auto& resource : resources) {
        if (resource.alignment == 0 ||
            (resource.alignment & (resource.alignment - 1)) != 0 ||
            resource.first_step > resource.last_step) {
            throw std::invalid_argument("invalid scratch alignment or lifetime");
        }
        plan.arena_alignment = std::max(plan.arena_alignment, resource.alignment);
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return resources[a].id.value < resources[b].id.value;
    });
    for (std::size_t i = 1; i < order.size(); ++i) {
        if (resources[order[i - 1]].id.value == resources[order[i]].id.value) {
            throw std::invalid_argument("duplicate scratch resource identity");
        }
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (resources[a].first_step != resources[b].first_step) {
            return resources[a].first_step < resources[b].first_step;
        }
        return a < b;
    });
    plan.placements.resize(resources.size());
    std::vector<std::size_t> active;
    std::vector<FreeRange> free_ranges;
    active.reserve(resources.size());
    free_ranges.reserve(resources.size());

    for (const auto index : order) {
        const auto& resource = resources[index];
        for (std::size_t i = 0; i < active.size();) {
            const auto prior = active[i];
            if (resources[prior].last_step < resource.first_step) {
                const auto& placement = plan.placements[prior];
                release_range(free_ranges,
                              {placement.offset, placement.offset + placement.bytes});
                active[i] = active.back();
                active.pop_back();
            } else {
                ++i;
            }
        }
        if (resource.bytes == 0) {
            plan.placements[index] = {resource.id, 0, 0};
            continue;
        }

        std::optional<std::size_t> offset;
        for (std::size_t i = 0; i < free_ranges.size(); ++i) {
            const auto range = free_ranges[i];
            const auto candidate = aligned_offset(range.begin, resource.alignment);
            if (!candidate || *candidate > range.end ||
                resource.bytes > range.end - *candidate) {
                continue;
            }
            offset = candidate;
            free_ranges.erase(free_ranges.begin() +
                              static_cast<std::ptrdiff_t>(i));
            release_range(free_ranges, {range.begin, *offset});
            release_range(free_ranges, {*offset + resource.bytes, range.end});
            break;
        }
        if (!offset) {
            offset = aligned_offset(plan.arena_bytes, resource.alignment);
            if (!offset || *offset > budget_bytes ||
                resource.bytes > budget_bytes - *offset) {
                throw std::length_error("scratch placement exceeds extent or budget");
            }
            release_range(free_ranges, {plan.arena_bytes, *offset});
            plan.arena_bytes = *offset + resource.bytes;
        }
        plan.placements[index] = {resource.id, *offset, resource.bytes};
        active.push_back(index);
    }
    verify_plan(resources, plan, budget_bytes);
    return plan;
}
}  // namespace dolittle::inference
