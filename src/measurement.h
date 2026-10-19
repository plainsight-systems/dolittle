/*
 * Experiment phase measurement payloads.
 * Axis D4: distinguish elapsed phase time from admitted allocation estimates.
 * Contract: seconds are nonnegative elapsed wall time from a monotonic clock;
 * an absent value means the phase did not complete a measurement, not zero cost.
 * GPU-related phase timing ends only after its dependent queue work completes.
 * Device estimates count planned arenas, excluding driver reservations. Host
 * estimates cover payload/staging plus supplied loader/text metadata allowances;
 * recipe/kernel metadata, OS page cache and allocator/process overhead are excluded.
 * They are not measured peaks or process RAM limits.
 * Values own no resources and carry no backend handles. Run/Record supplies the
 * associated backend, model revision, precision, token counts and settings.
 *
 * Algorithm: docs/architecture/logical-overview.md, "Resource and lifetime
 * constraints"; transformer-execution.md, "Cost model":
 * 1. Before allocation, compute checked weight, KV, scratch, staging and retained
 *    output requirements. Keep these estimates separately from elapsed timings.
 * 2. Bracket loading, generation, release and decoding with monotonic timestamps.
 *    End GPU phases at their existing completion boundary; add no per-layer wait.
 * 3. Store per-attempt durations with its variant, and accumulate completed phase
 *    totals in Run. A failed phase may have an elapsed duration if cleanup completed;
 *    failure status remains in the record and is not inferred from a time value.
 * Cost: O(1) counters per phase/variant. Optimization: use coarse existing phase
 * boundaries, not instrumentation or synchronization inside each transformer block.
 * Corpus consultation, header design and implementation, 2026-10-19 (full rules read):
 * cpp-guidelines I.7 (#ri-post): absent, elapsed and estimated values are distinct.
 * cpp-perf-guidelines GPU.1: include queue completion in transfer/compute costs;
 * source guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.9: report allocation estimates separately by lifetime;
 * source guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#pragma once

#include <cstddef>
#include <optional>

namespace dolittle::measurement {
struct InterpretationCost {
    std::optional<double> conversion_seconds;
    std::optional<double> load_seconds;
    std::optional<double> prepare_seconds;
    std::optional<double> generate_seconds;
    std::optional<double> decode_seconds;
    std::optional<double> release_seconds;
};
struct PhaseMeasurements {
    std::optional<double> target_load_seconds;
    std::optional<double> target_generate_seconds;
    std::optional<double> target_release_seconds;
    std::optional<double> verbalizer_load_seconds;
    std::optional<double> verbalizer_generate_seconds; // Accumulated across variants.
    std::optional<double> decode_seconds;
    std::size_t estimated_peak_device_bytes{};
    std::size_t estimated_peak_host_bytes{};
};
} // namespace dolittle::measurement
