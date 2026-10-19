/*
 * Axis D3: isolated scratch-placement contract checks, with no device work.
 * Contract: use only in-memory lifetime records and inspect owned plan results;
 * fail through the shared test helper on missing identities, invalid extents,
 * misalignment, simultaneous aliasing or incorrect exception categories.
 * No files, subprocesses, system launches or private placement snapshots.
 * Algorithm: docs/architecture/transformer-execution.md, "Model computation"
 * and "Storage lifetimes": prepare varied valid schedules, check each public
 * placement against its input identity and every intersecting lifetime, then
 * exercise invalid input and finite-size admission boundaries. Tight budgets
 * check basic sequential reuse without fixing offsets or free-list choices.
 * Design and cost: bounded fixtures, O(R^2) contract checks and O(R) metadata;
 * no tensor payloads are allocated, including size_t boundary cases.
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): assert externally meaningful postconditions.
 * cpp-guidelines ES.103 (#res-overflow): test representable extent boundaries.
 * cpp-perf-guidelines GPU.9: overlapping inclusive lifetimes cannot alias.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines MEM.11: verify plan alignment, containment and lifetime
 * safety without treating a greedy policy as an optimal packing guarantee.
 * Source: guidelines/memory/MEM.11-plan-arena-offsets-from-lifetimes.md.
 */
#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "bounds.h"
#include "inference/storage_plan.h"
#include "tests/support/check.h"

namespace {
using dolittle::inference::plan_storage;
using dolittle::inference::ResourceLifetime;
using dolittle::inference::ResourcePlacement;
using dolittle::inference::StoragePlan;
using dolittle::tests::check;
using dolittle::tests::expect_throws;

const ResourcePlacement& find_placement(const StoragePlan& plan,
                                        std::size_t id) {
  const auto matches = std::count_if(
      plan.placements.begin(), plan.placements.end(),
      [id](const auto& placement) { return placement.id.value == id; });
  check(matches == 1, "each requested resource must occur exactly once");
  return *std::find_if(
      plan.placements.begin(), plan.placements.end(),
      [id](const auto& placement) { return placement.id.value == id; });
}

void check_plan(std::span<const ResourceLifetime> resources,
                const StoragePlan& plan, std::size_t budget) {
  check(plan.placements.size() == resources.size(), "placement count");
  check(plan.arena_bytes <= budget, "arena budget");
  check(plan.arena_alignment != 0 &&
            (plan.arena_alignment & (plan.arena_alignment - 1)) == 0,
        "arena alignment must be a power of two");
  std::vector<const ResourcePlacement*> by_resource;
  by_resource.reserve(resources.size());
  for (const auto& resource : resources) {
    by_resource.push_back(&find_placement(plan, resource.id.value));
  }
  for (std::size_t i = 0; i < resources.size(); ++i) {
    const auto& resource = dolittle::at(resources, i);
    const auto& placement = *by_resource.at(i);
    check(placement.bytes == resource.bytes, "resource byte extent");
    check(placement.offset % resource.alignment == 0, "resource alignment");
    check(plan.arena_alignment % resource.alignment == 0,
          "arena base supports resource alignment");
    check(placement.offset <= plan.arena_bytes, "offset within arena");
    check(placement.bytes <= plan.arena_bytes - placement.offset,
          "extent within arena");
    for (std::size_t j = 0; j < i; ++j) {
      const auto& prior = dolittle::at(resources, j);
      if (resource.bytes == 0 || prior.bytes == 0 ||
          resource.first_step > prior.last_step ||
          prior.first_step > resource.last_step) {
        continue;
      }
      const auto& other = *by_resource.at(j);
      check(placement.offset + placement.bytes <= other.offset ||
                other.offset + other.bytes <= placement.offset,
            "simultaneously live resources must not overlap");
    }
  }
}

void empty_and_zero_size() {
  const auto empty = plan_storage({}, 0);
  check(empty.placements.empty() && empty.arena_bytes == 0 &&
            empty.arena_alignment == 1,
        "empty plan contract");
  const std::array resources{
      ResourceLifetime{{9}, 0, 64, 0, 100},
      ResourceLifetime{{42}, 0, 1, 5, 7},
  };
  check_plan(resources, plan_storage(resources, 0), 0);
}

void lifetime_and_budget_boundaries() {
  const std::array sequential{
      ResourceLifetime{{100}, 8, 8, 0, 2},
      ResourceLifetime{{200}, 8, 8, 3, 7},
  };
  check_plan(sequential, plan_storage(sequential, 8), 8);
  auto shared_endpoint = sequential;
  shared_endpoint.at(1).first_step = shared_endpoint.at(0).last_step;
  check_plan(shared_endpoint, plan_storage(shared_endpoint, 16), 16);
  expect_throws<std::length_error>(
      [&] { (void)plan_storage(shared_endpoint, 15); });

  const std::array nested{
      ResourceLifetime{{0}, 7, 1, 0, 10},  ResourceLifetime{{1}, 8, 16, 2, 4},
      ResourceLifetime{{2}, 3, 4, 5, 5},   ResourceLifetime{{3}, 0, 32, 3, 8},
      ResourceLifetime{{4}, 19, 64, 0, 0}, ResourceLifetime{{5}, 5, 8, 6, 9},
  };
  check_plan(nested, plan_storage(nested, 256), 256);
}

void varied_orders_and_lifetimes() {
  for (std::size_t variant = 0; variant < 16; ++variant) {
    std::vector<ResourceLifetime> resources;
    std::size_t sequential_bound = 0;
    for (std::size_t i = 0; i < 24; ++i) {
      const auto bytes = (i * 13 + variant) % 41;
      const auto alignment = std::size_t{1} << ((i + variant) % 7);
      const auto start = (i * 5 + variant * 3) % 19;
      resources.push_back(
          {{1000 + i * 3}, bytes, alignment, start, start + (i + variant) % 7});
      sequential_bound += bytes + alignment - 1;
    }
    check_plan(resources, plan_storage(resources, sequential_bound),
               sequential_bound);
    std::reverse(resources.begin(), resources.end());
    check_plan(resources, plan_storage(resources, sequential_bound),
               sequential_bound);
  }
}

void rejected_inputs() {
  const std::array<ResourceLifetime, 4> invalid{{
      {{1}, 1, 0, 0, 0},
      {{1}, 1, 3, 0, 0},
      {{1}, 1, 8, 2, 1},
      {{1}, 0, 0, 0, 0},
  }};
  for (const auto& resource : invalid) {
    expect_throws<std::invalid_argument>(
        [&] { (void)plan_storage(std::span{&resource, 1}, 100); });
  }
  const std::array duplicate{
      ResourceLifetime{{7}, 1, 1, 0, 0},
      ResourceLifetime{{7}, 0, 1, 2, 3},
  };
  expect_throws<std::invalid_argument>(
      [&] { (void)plan_storage(duplicate, 100); });
  const std::array nonempty{ResourceLifetime{{0}, 1, 1, 0, 0}};
  expect_throws<std::length_error>([&] { (void)plan_storage(nonempty, 0); });
}

void extreme_extents_and_steps() {
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  const std::array full{
      ResourceLifetime{{maximum}, maximum, 1, maximum, maximum}};
  check_plan(full, plan_storage(full, maximum), maximum);
  const std::array overflow{
      ResourceLifetime{{0}, maximum, 1, 0, 1},
      ResourceLifetime{{1}, 1, 1, 0, 1},
  };
  expect_throws<std::length_error>(
      [&] { (void)plan_storage(overflow, maximum); });
  auto alignment_overflow = overflow;
  alignment_overflow.at(1).alignment = 2;
  expect_throws<std::length_error>(
      [&] { (void)plan_storage(alignment_overflow, maximum); });
  const std::array final_steps{
      ResourceLifetime{{4}, 8, 8, maximum - 1, maximum - 1},
      ResourceLifetime{{8}, 8, 8, maximum, maximum},
  };
  check_plan(final_steps, plan_storage(final_steps, 8), 8);
}
}  // namespace

int main() try {
  empty_and_zero_size();
  lifetime_and_budget_boundaries();
  varied_orders_and_lifetimes();
  rejected_inputs();
  extreme_extents_and_steps();
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
