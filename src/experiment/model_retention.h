/*
 * Implemented foundation, 2026-10-21; header design approved before
 * implementation. Axis R3: experiment scheduling and model retention. Pure
 * admission and eviction decisions for target/AV/AR weights; native GPU
 * residency is X4. Contract: all byte counts describe physical owners, with no
 * duplicated charge for aliases. device/host limits remain separate logical
 * allowances; when unified_bytes is present it additionally bounds the unique
 * physical footprint across them. A shared allocation can count against both
 * logical allowances but exactly once against the unified allowance. All model
 * roles are unique. The active model and pinned entries cannot be eviction
 * victims. If active is already resident, its reported weight cost must match
 * the request; inconsistent ownership/accounting is invalid_argument. Callers
 * supply verified model identity via the role-scoped owning cache; a plan never
 * authorizes using weights for another revision or numerical mode.
 *
 * Algorithm: metal-port.md, Memory and model residency.
 * 1. Add persistent captures/records, the active model and the request's peak
 *    transient footprint (including loading, commands, KV and scratch). Each
 *    owner occurs once; an already-resident active model is not added twice.
 * 2. Add other resident entries. If all limits fit, retain them all.
 * 3. Otherwise remove unpinned inactive entries in ascending last_use order;
 *    break ties by role. Stop when all limits fit. If required owners alone
 *    exceed any limit, return nullopt and change nothing.
 * 4. Return victims and the peak after eviction. Run retires their completed
 *    sequences before release, rechecks admission, then loads/prepares work.
 * A valid plan performs no eviction itself and reserves no driver memory.
 * Allocation failure remains possible and must not weaken requested precision,
 * capture coverage or sequence capacity. Cancellation never mutates this plan.
 *
 * Cost: O(M log M) setup for M resident roles; O(M) host metadata. No model
 * loading, GPU calls, filesystem access or token-loop work in this unit.
 * Optimization: retain expensive weights when the next full request fits,
 * without retaining obsolete KV/scratch or forcing simultaneous execution.
 * Required and active costs include conservative host mapping/loader costs
 * when backing bytes cannot be proven to share physical ownership.
 *
 * Both local corpora queried and rules read for header design:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): explicit budgets and a
 * nonmutating decision. ES.103 (#res-overflow): checked sums, invalid counts
 * -> invalid_argument, arithmetic overflow -> length_error, bad_alloc passes
 * through. cpp-perf-guidelines GPU.9
 * (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md): account
 * for persistent and transient lifetimes. GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md): reuse device-resident weights.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "gpu/memory.h"
#include "nla/protocol.h"

namespace dolittle::experiment {
using MemoryCost = gpu::MemoryFootprint;
struct ModelRetentionBudget {
  std::size_t device_bytes{};
  std::size_t host_bytes{};
  std::optional<std::size_t> unified_bytes;
};
struct ResidentEntry {
  nla::ModelRole role{};
  MemoryCost weights;
  std::uint64_t last_use{};
  bool pinned{};
};
struct ModelRetentionRequest {
  nla::ModelRole active{};
  MemoryCost active_weights;
  MemoryCost retained;
  MemoryCost transient_peak;
};
struct ModelRetentionDecision {
  std::vector<nla::ModelRole> evict;
  MemoryCost admitted_peak;
};
[[nodiscard]] std::optional<ModelRetentionDecision> plan_model_retention(
    ModelRetentionBudget, ModelRetentionRequest,
    std::span<const ResidentEntry>);
}  // namespace dolittle::experiment
