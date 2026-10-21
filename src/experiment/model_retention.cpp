/*
 * R3 model retention implementation; contract: model_retention.h. No resources
 * are mutated. Sum required owners, count an active cache hit once, then evict
 * eligible idle roles oldest-first until all budgets fit. Return nullopt when
 * pinned/required owners alone cannot fit. Checked arithmetic precedes every
 * sum; unique roles and consistent active cost are validated before sorting.
 * O(M log M) setup, O(M) metadata, no device work or per-token allocation.
 * Overlay: metal-port.md, Memory and model residency. Implementation corpus
 * queries/read, 2026-10-21: Core I.1/I.7 (explicit nonmutating result), ES.103
 * (#res-overflow); performance GPU.9 (persistent/transient accounting), GPU.1
 * (retain useful device weights). Sources: guidelines/gpu/GPU.9-suballocate-
 * and-alias-transient-gpu-memory.md; GPU.1-keep-data-on-device.md.
 */
#include "experiment/model_retention.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace dolittle::experiment {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::length_error("model retention byte sum overflows");
  }
  return a + b;
}
MemoryCost add(MemoryCost a, MemoryCost b) {
  return {add(a.device_bytes, b.device_bytes), add(a.host_bytes, b.host_bytes),
          add(a.unified_bytes, b.unified_bytes)};
}
std::size_t role_index(nla::ModelRole role) {
  switch (role) {
    case nla::ModelRole::target:
      return 0;
    case nla::ModelRole::verbalizer:
      return 1;
    case nla::ModelRole::reconstructor:
      return 2;
  }
  throw std::invalid_argument("invalid model retention role");
}
bool fits(MemoryCost cost, ModelRetentionBudget budget) {
  return cost.device_bytes <= budget.device_bytes &&
         cost.host_bytes <= budget.host_bytes &&
         (!budget.unified_bytes || cost.unified_bytes <= *budget.unified_bytes);
}
bool same(MemoryCost a, MemoryCost b) {
  return a.device_bytes == b.device_bytes && a.host_bytes == b.host_bytes &&
         a.unified_bytes == b.unified_bytes;
}
}  // namespace
std::optional<ModelRetentionDecision> plan_model_retention(
    ModelRetentionBudget budget, ModelRetentionRequest request,
    std::span<const ResidentEntry> resident) {
  (void)role_index(request.active);
  auto peak = add(add(request.retained, request.active_weights),
                  request.transient_peak);
  std::array<bool, 3> seen{};
  std::vector<ResidentEntry> candidates;
  candidates.reserve(resident.size());
  for (const auto& entry : resident) {
    auto& present = seen.at(role_index(entry.role));
    if (present) throw std::invalid_argument("duplicate resident role");
    present = true;
    if (entry.role == request.active) {
      if (!same(entry.weights, request.active_weights)) {
        throw std::invalid_argument("active model accounting differs");
      }
      continue;
    }
    peak = add(peak, entry.weights);
    if (!entry.pinned) candidates.push_back(entry);
  }
  std::ranges::sort(candidates, [](const auto& a, const auto& b) {
    return a.last_use != b.last_use ? a.last_use < b.last_use : a.role < b.role;
  });
  ModelRetentionDecision result{{}, peak};
  for (const auto& entry : candidates) {
    if (fits(result.admitted_peak, budget)) break;
    result.evict.push_back(entry.role);
    // Each entry was added exactly once above; subtraction cannot underflow.
    result.admitted_peak.device_bytes -= entry.weights.device_bytes;
    result.admitted_peak.host_bytes -= entry.weights.host_bytes;
    result.admitted_peak.unified_bytes -= entry.weights.unified_bytes;
  }
  if (!fits(result.admitted_peak, budget)) return std::nullopt;
  return result;
}
}  // namespace dolittle::experiment
