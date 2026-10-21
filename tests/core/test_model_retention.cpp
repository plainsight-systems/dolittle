/*
 * D3 isolated R3 admission contracts: required/pinned owners survive, idle
 * eviction is oldest-first, aliases count once, and invalid arithmetic fails.
 * In-memory inputs only. O(roles) fixtures. Core I.7 (#ri-post), ES.103
 * (#res-overflow); performance GPU.9
 * (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md), read
 * during implementation, motivate budget/lifetime assertions rather than
 * layout.
 */
#include <array>
#include <limits>

#include "experiment/model_retention.h"
#include "tests/support/check.h"
namespace tests = dolittle::tests;
namespace nla = dolittle::nla;
int main() {
  try {
    using dolittle::experiment::ModelRetentionRequest;
    using dolittle::experiment::plan_model_retention;
    using dolittle::experiment::ResidentEntry;
    using nla::ModelRole;
    using tests::check;
    const ModelRetentionRequest request{
        ModelRole::target, {40, 10, 50}, {5, 5, 10}, {10, 5, 15}};
    std::array resident{
        ResidentEntry{ModelRole::target, {40, 10, 50}, 3, false},
        ResidentEntry{ModelRole::verbalizer, {30, 10, 40}, 1, false},
        ResidentEntry{ModelRole::reconstructor, {20, 10, 30}, 2, false}};
    auto decision = plan_model_retention({105, 40, 145}, request, resident);
    check(decision && decision->evict.empty(),
          "all models retained when full peak fits");
    check(decision && decision->admitted_peak.device_bytes == 105 &&
              decision->admitted_peak.unified_bytes == 145,
          "active cache hit counted once");
    decision = plan_model_retention({80, 40, 110}, request, resident);
    check(decision && decision->evict == std::vector{ModelRole::verbalizer},
          "evict oldest idle role");
    resident.at(1).pinned = true;
    decision = plan_model_retention({90, 40, 120}, request, resident);
    check(decision && decision->evict == std::vector{ModelRole::reconstructor},
          "pinned owner survives");
    check(!plan_model_retention({50, 40, 145}, request, resident),
          "required device peak cannot fit");
    check(!plan_model_retention({200, 10, 200}, request, resident),
          "host limit is independent");
    check(!plan_model_retention({200, 200, 74}, request, {}),
          "unified physical budget enforced");
    check(plan_model_retention({200, 200, std::nullopt}, request, resident)
              .has_value(),
          "discrete has no unified limit");
    resident.at(0).weights.device_bytes++;
    tests::expect_throws<std::invalid_argument>([&] {
      (void)plan_model_retention({200, 200, 200}, request, resident);
    });
    resident.at(0) = resident.at(1);
    tests::expect_throws<std::invalid_argument>([&] {
      (void)plan_model_retention({200, 200, 200}, request, resident);
    });
    auto overflow = request;
    overflow.transient_peak.device_bytes =
        std::numeric_limits<std::size_t>::max();
    tests::expect_throws<std::length_error>(
        [&] { (void)plan_model_retention({200, 200, 200}, overflow, {}); });
  } catch (const std::exception& error) {
    return tests::report_failure(error.what());
  }
}
