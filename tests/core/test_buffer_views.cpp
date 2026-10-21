/*
 * D3 pure view-placement contracts. Every operand fits one bounded page-aligned
 * view; all views stay inside backing. Empty/overflow/oversize cases are
 * checked without Metal or allocation of backing. Core I.7/ES.103
 * (#ri-post/#res-overflow), performance GPU.9
 * (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md), read
 * for implementation: verify ownership extents, not view-count heuristics.
 */
#include <array>
#include <limits>

#include "gpu/metal/buffer_views.h"
#include "tests/support/check.h"
namespace tests = dolittle::tests;
int main() {
  try {
    using dolittle::gpu::metal::BufferExtent;
    using dolittle::gpu::metal::plan_buffer_views;
    const std::array bindings{BufferExtent{100, 100}, BufferExtent{4000, 3000},
                              BufferExtent{9000, 7000}};
    const auto plan = plan_buffer_views(16384, 8192, 4096, bindings);
    tests::check(plan.binding_views.size() == bindings.size(),
                 "one result per operand");
    for (const auto view : plan.views) {
      tests::check(view.offset % 4096 == 0 && view.bytes % 4096 == 0 &&
                       view.bytes <= 8192 && view.offset + view.bytes <= 16384,
                   "bounded aligned view");
    }
    for (std::size_t i = 0; i < bindings.size(); ++i) {
      const auto view = plan.views.at(plan.binding_views.at(i));
      const auto binding = bindings.at(i);
      tests::check(
          view.offset <= binding.offset &&
              view.offset + view.bytes >= binding.offset + binding.bytes,
          "whole operand contained");
    }
    tests::check(plan_buffer_views(0, 4096, 4096, {}).views.empty(),
                 "empty backing");
    const auto reject = [](BufferExtent binding) {
      return plan_buffer_views(16384, 8192, 4096, std::span(&binding, 1));
    };
    tests::expect_throws<std::invalid_argument>(
        [&] { (void)reject({16384, 1}); });
    tests::expect_throws<std::invalid_argument>([&] { (void)reject({0, 0}); });
    tests::expect_throws<std::length_error>(
        [&] { (void)reject({4095, 8192}); });
    tests::expect_throws<std::length_error>(
        [&] { (void)reject({std::numeric_limits<std::size_t>::max(), 2}); });
  } catch (const std::exception& error) {
    return tests::report_failure(error.what());
  }
}
