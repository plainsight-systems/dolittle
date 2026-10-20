/*
 * Axis D3: one integration scenario for the CUDA resource/dispatch protocol.
 * Contract: requires a CUDA device supported by the configured build. Exercises
 * actual allocation, pinned staging, ordered transfers, completion and prepared
 * binding validation. This is an integration test, never a unit-test fixture.
 * All resources outlive pending work; intentional rejection cases fail before
 * launch and must leave the same device usable. No intentional device fault is
 * injected, because that would destroy the shared runtime under this scenario.
 * Algorithm: transformer overlay "Preparing the first execution recipe" and
 * "Storage lifetimes": prepare a byte-preserving gather, allocate storage,
 * upload/copy/read back two ordered prefixes, reject invalid resource/binding
 * relations, gather selected raw rows, and release resources with pending work.
 * Design/cost: bounded tiny buffers and one prepared operation; numerical
 * transformer equations belong to the independently referenced forward tests.
 * Optimization: one ordered queue and batched transfers mirror production;
 * assertions check ready bytes and public failures, not CUDA instruction
 * counts. Corpus consultation, design/implementation, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): lifetimes survive failed submission and
 * teardown. cpp-guidelines I.7 (#ri-post): explicit completion establishes
 * readable bytes. cpp-perf-guidelines GPU.7: verify prefix events and staging
 * lifetimes. Source: guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 * cpp-perf-guidelines GPU.9: bind separate ranges within admitted storage.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * Direct-cache-window design and implementation, 2026-10-19:
 * Within this resource/dispatch scenario, write two disjoint borrowed cache
 * windows, read back only initialized rows and require earlier bytes to remain
 * unchanged. Reject a foreign queue, empty/out-of-range windows and bad block.
 * This tests storage bounds/ownership, not a second numerical workflow.
 * cpp-guidelines I.7 (#ri-post): window access cannot overwrite the prefix.
 * cpp-perf-guidelines GPU.9: validate disjoint subregions of one owned arena.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * Prepared-buffer/device-selection contract consultation, 2026-10-20:
 * Core I.7 (#ri-post) and MEM.9
 * (guidelines/memory/MEM.9-embedded-allocate-at-init.md), both corpora queried
 * and full rules read for design/implementation. Exercise public results and
 * capacity/failure contracts; independent reference inputs remain unchanged.
 * Device-feedback verification, 2026-10-20: Core I.7 (#ri-post) checks
 * standard RNG sequence and terminal postconditions. Performance GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md) checks device-owned sampling
 * with host comparison only at this diagnostic boundary. Both corpora queried
 * and rules read for header design and implementation.
 * Autonomous-loop verification, 2026-10-20: drain the entire device loop
 * before collecting any output; require exact seeded IDs, positions, budget,
 * EOS and cancellation. This tests the existing device resource/dispatch
 * contract, with no transformer fixture. Core I.7/CP.3 and performance GPU.6
 * were queried/read in both phases: results must be immutable and device
 * progress independent of CPU collection. Source:
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md.
 */
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "diagnostics/allocation.h"
#include "gpu/cuda/device.h"
#include "inference/cache/contiguous.h"
#include "inference/decoding/device.h"
#include "inference/injection/device.h"
#include "inference/sampling/categorical.h"
#include "inference/sampling/device.h"
#include "kernels/operations.h"
#include "tensors/representations/bf16.h"
#include "tests/support/check.h"

namespace tensors = dolittle::tensors;
namespace kernels = dolittle::kernels;
namespace gpu = dolittle::gpu;
using dolittle::tests::check;
using dolittle::tests::expect_throws;

int main() try {
  auto device = gpu::cuda::open_device(0);
  constexpr auto fp32 = tensors::ElementType::float32;
  const std::array layouts{
      tensors::contiguous_layout(fp32, std::array<std::size_t, 2>{4, 2}),
      tensors::contiguous_layout(fp32, std::array<std::size_t, 2>{2, 2})};
  auto gather = kernels::prepare(*device, kernels::GatherRows{{0}, {1}, {0, 3}},
                                 layouts, {fp32, fp32});
  check(gather.workspace().bytes == 0, "gather needs no global scratch");
  auto input = device->allocate({64, 64});
  auto copied = device->allocate({64, 64});
  auto output = device->allocate({64, 64});
  auto staging = device->allocate_staging({64, 64});
  auto readback = device->allocate_staging({64, 64});
  auto queue = device->create_queue();

  // Includes signed zero, a noncanonical NaN payload and arbitrary original
  // bits.
  constexpr std::array<std::uint32_t, 8> raw{
      0x80000000U, 0x7fc01234U, 0x3f800000U, 0xbf800000U,
      0x00000001U, 0x7f800000U, 0xff800000U, 0x3f123456U};
  std::array<std::byte, 32> first{};
  for (std::size_t i = 0; i < raw.size(); ++i) {
    for (unsigned byte = 0; byte < 4; ++byte) {
      first.at(i * 4 + byte) =
          static_cast<std::byte>((raw.at(i) >> (8 * byte)) & 0xffU);
    }
  }
  auto second = first;
  std::ranges::reverse(second);
  std::ranges::copy(first, staging.bytes().begin());
  std::ranges::copy(second, staging.bytes().begin() + 32);

  queue.upload({&input, 0, 32}, {&staging, 0, 32});
  queue.copy({&copied, 0, 32}, {&input, 0, 32});
  queue.download({&readback, 0, 32}, {&copied, 0, 32});
  const auto first_ready = queue.mark();
  queue.upload({&input, 0, 32}, {&staging, 32, 32});
  queue.copy({&copied, 0, 32}, {&input, 0, 32});
  queue.download({&readback, 32, 32}, {&copied, 0, 32});
  const auto second_ready = queue.mark();
  queue.wait(first_ready);
  check(std::ranges::equal(first, readback.bytes().first(32)),
        "first prefix makes its bytes ready");
  queue.wait(second_ready);
  check(std::ranges::equal(second, readback.bytes().subspan(32, 32)),
        "later prefix keeps transfer order");
  queue.wait(first_ready);
  expect_throws<std::invalid_argument>(
      [&] { queue.wait({first_ready.queue_identity, 0}); });
  expect_throws<std::invalid_argument>([&] {
    queue.wait({second_ready.queue_identity, second_ready.sequence + 1});
  });
  expect_throws<std::invalid_argument>([&] { (void)device->allocate({8, 3}); });
  expect_throws<std::invalid_argument>(
      [&] { queue.copy({&input, 4, 8}, {&input, 0, 8}); });
  expect_throws<std::invalid_argument>(
      [&] { queue.upload({&input, 0, 16}, {&staging, 0, 32}); });
  expect_throws<std::invalid_argument>(
      [&] { queue.download({&readback, 60, 8}, {&input, 0, 8}); });
  expect_throws<std::length_error>([&] {
    queue.copy({&output, std::numeric_limits<std::size_t>::max(), 2},
               {&input, 0, 2});
  });
  // Empty ranges still have valid owning resources and admitted endpoints.
  queue.copy({&output, output.size_bytes(), 0},
             {&input, input.size_bytes(), 0});

  const std::array<kernels::BufferBinding, 2> bindings{
      {{{1}, &output, 0, 16},
       {{0}, static_cast<const gpu::DeviceBuffer*>(&input), 0, 32}}};
  const gpu::DeviceRegion workspace{nullptr, 0, 0};
  auto invalid = bindings;
  invalid.at(1).offset = 1;
  expect_throws<std::invalid_argument>(
      [&] { gather.submit(queue, invalid, workspace); });
  invalid = bindings;
  invalid.at(0).storage = static_cast<const gpu::DeviceBuffer*>(&output);
  expect_throws<std::invalid_argument>(
      [&] { gather.submit(queue, invalid, workspace); });
  invalid = bindings;
  invalid.at(0).storage = &input;
  expect_throws<std::invalid_argument>(
      [&] { gather.submit(queue, invalid, workspace); });
  invalid = bindings;
  invalid.at(0).bytes = 8;
  expect_throws<std::invalid_argument>(
      [&] { gather.submit(queue, invalid, workspace); });
  invalid = bindings;
  invalid.at(0).tensor = {0};
  expect_throws<std::invalid_argument>(
      [&] { gather.submit(queue, invalid, workspace); });
  expect_throws<std::invalid_argument>([&] {
    gather.submit(queue,
                  std::span<const kernels::BufferBinding>(bindings).first(1),
                  workspace);
  });
  expect_throws<std::invalid_argument>(
      [&] { gather.submit(queue, bindings, {&copied, 0, 4}); });

  {
    auto foreign_device = gpu::cuda::open_device(0);
    auto foreign_buffer = foreign_device->allocate({64, 64});
    auto foreign_queue = foreign_device->create_queue();
    expect_throws<std::invalid_argument>(
        [&] { queue.copy({&foreign_buffer, 0, 8}, {&input, 0, 8}); });
    invalid = bindings;
    invalid.at(0).storage = &foreign_buffer;
    expect_throws<std::invalid_argument>(
        [&] { gather.submit(queue, invalid, workspace); });
    expect_throws<std::invalid_argument>(
        [&] { gather.submit(foreign_queue, bindings, workspace); });
    expect_throws<std::invalid_argument>(
        [&] { foreign_queue.wait(first_ready); });
  }

  queue.upload({&input, 0, 32}, {&staging, 0, 32});
  gather.submit(queue, bindings, workspace);
  queue.download({&readback, 0, 16}, {&output, 0, 16});
  queue.wait(queue.mark());
  check(std::ranges::equal(std::span<const std::byte>(first).first(8),
                           readback.bytes().first(8)) &&
            std::ranges::equal(std::span<const std::byte>(first).subspan(24, 8),
                               readback.bytes().subspan(8, 8)),
        "gather preserves selected raw row bits and order after rejected "
        "submissions");

  // Resource destruction must complete outstanding uses, even without an
  // explicit wait. The surviving readback becomes safe when these owners exit.
  {
    auto temporary = device->allocate({32, 64});
    auto temporary_staging = device->allocate_staging({32, 64});
    std::ranges::copy(second, temporary_staging.bytes().begin());
    queue.upload({&temporary, 0, 32}, {&temporary_staging, 0, 32});
    queue.download({&readback, 0, 32}, {&temporary, 0, 32});
  }
  check(std::ranges::equal(second, readback.bytes().first(32)),
        "teardown drains outstanding uses");
  {
    namespace cache = dolittle::inference::cache;
    const cache::Specification spec{
        "device-window", {1}, {fp32, fp32}, 5, {{1, 4}}};
    auto cache_storage = cache::allocate_contiguous(*device, queue, spec, 160);
    auto other_queue = device->create_queue();
    expect_throws<std::invalid_argument>(
        [&] { (void)cache_storage->writable_window(other_queue, 0, 0, 1); });
    expect_throws<std::invalid_argument>(
        [&] { (void)cache_storage->writable_window(queue, 0, 0, 0); });
    expect_throws<std::invalid_argument>(
        [&] { (void)cache_storage->writable_window(queue, 0, 4, 2); });
    expect_throws<std::invalid_argument>(
        [&] { (void)cache_storage->writable_window(queue, 1, 0, 1); });
    const auto prefix = cache_storage->writable_window(queue, 0, 0, 2);
    const auto extension = cache_storage->writable_window(queue, 0, 2, 2);
    const auto final_row = cache_storage->writable_window(queue, 0, 4, 1);
    check(final_row.keys.bytes == 16 && final_row.values.bytes == 16,
          "last cache row fits its admitted boundary");
    queue.upload(prefix.keys, {&staging, 0, 32});
    queue.upload(extension.keys, {&staging, 32, 32});
    queue.upload(prefix.values, {&staging, 32, 32});
    queue.upload(extension.values, {&staging, 0, 32});
    const auto view = cache_storage->layer(0);
    queue.download({&readback, 0, 64}, {view.keys.owner, view.keys.offset, 64});
    queue.wait(queue.mark());
    check(std::ranges::equal(first, readback.bytes().first(32)) &&
              std::ranges::equal(second, readback.bytes().subspan(32, 32)),
          "direct key writes preserve earlier cache rows");
    queue.download({&readback, 0, 64},
                   {view.values.owner, view.values.offset, 64});
    queue.wait(queue.mark());
    check(std::ranges::equal(second, readback.bytes().first(32)) &&
              std::ranges::equal(first, readback.bytes().subspan(32, 32)),
          "direct value writes remain separate from keys");
  }
  // Selection is part of the same prepared device dispatch contract: equal
  // keys, masks, tails, signed zeros, full vocabulary, filter boundaries and
  // explicit rejection. The independent CPU selector remains unchanged.
  namespace sampling = dolittle::inference::sampling;
  for (const auto type : {fp32, tensors::ElementType::bfloat16}) {
    for (const std::size_t n :
         {std::size_t{1}, std::size_t{513}, std::size_t{151936}}) {
      std::vector<float> logits(n);
      for (std::size_t i = 0; i < n; ++i) {
        logits.at(i) = static_cast<float>(static_cast<int>(i % 97) - 48) / 8.0F;
      }
      logits.front() = -0.0F;
      if (n > 1) {
        logits.at(1) = 0.0F;
        logits.back() = -std::numeric_limits<float>::infinity();
      }
      const auto bytes = n * tensors::element_bytes(type);
      auto row = device->allocate({bytes, 256});
      auto upload = device->allocate_staging({bytes, 8});
      auto answer = device->allocate_staging({8, 8});
      const auto upload_row = [&] {
        for (std::size_t i = 0; i < n; ++i) {
          if (type == fp32) {
            std::memcpy(upload.bytes().subspan(i * 4).data(), &logits.at(i), 4);
          } else {
            const auto value = tensors::from_float(logits.at(i));
            tensors::write_le(value,
                              std::span<std::byte, 2>(
                                  upload.bytes().subspan(i * 2).data(), 2));
          }
        }
        queue.upload({&row, 0, bytes}, {&upload, 0, bytes});
      };
      upload_row();
      sampling::DeviceCategorical selected(
          *device, queue, n, type,
          sampling::DeviceCategorical::required_bytes(n));
      sampling::Categorical reference(n);
      for (const auto mode : {sampling::SelectionMode::greedy,
                              sampling::SelectionMode::categorical}) {
        for (const auto k : {std::size_t{0}, std::min(n, std::size_t{20})}) {
          for (const double p : {0.01, 0.5, 0.95, 1.0}) {
            for (std::uint64_t seed = 0; seed < 8; ++seed) {
              const sampling::Settings settings =
                  mode == sampling::SelectionMode::greedy
                      ? sampling::Settings{mode, 1.0, 0, 1.0}
                      : sampling::Settings{mode, 0.6, k, p};
              sampling::RandomStream expected_rng(seed), draw_rng(seed);
              const auto expected =
                  reference.select(logits, settings, expected_rng);
              const auto bits =
                  mode == sampling::SelectionMode::categorical ? draw_rng() : 0;
              std::array<std::uint32_t, 2> result{};
              {
                const dolittle::diagnostics::AllocationScope scope(true);
                const auto region =
                    selected.submit({&row, 0, bytes}, {settings, bits});
                queue.download({&answer, 0, 8}, region);
                queue.wait(queue.mark());
                std::memcpy(result.data(), answer.bytes().data(), 8);
              }
              check(result.at(1) == 0 && result.at(0) == expected,
                    "device selection differs from host law");
            }
          }
        }
      }
      if (n == 513) {
        // Cross multiple MT19937-64 twist boundaries against the independent
        // standard-library sequence; all results use the public sampler API.
        const sampling::Settings settings{sampling::SelectionMode::categorical,
                                          0.6, 0, 0.95};
        // A fixed seed is the reproducibility contract being checked.
        // NOLINTNEXTLINE(bugprone-random-generator-seed)
        sampling::RandomStream expected_rng(42);
        selected.start(settings, 42, {});
        for (unsigned i = 0; i < 700; ++i) {
          const auto expected =
              reference.select(logits, settings, expected_rng);
          const dolittle::diagnostics::AllocationScope scope(true);
          const auto region = selected.submit_next({&row, 0, bytes});
          queue.download({&answer, 0, 8}, region);
          queue.wait(queue.mark());
          std::array<std::uint32_t, 2> result{};
          std::memcpy(result.data(), answer.bytes().data(), 8);
          check(result.at(1) == 0 && result.at(0) == expected,
                "device-owned RNG sequence differs from reference");
        }
        // The device must complete the entire bounded loop while the CPU only
        // waits once. No output collection can replenish or acknowledge work.
        for (unsigned scenario = 0; scenario < 7; ++scenario) {
          sampling::DeviceCategorical autonomous(
              *device, queue, n, type,
              sampling::DeviceCategorical::required_bytes(n));
          // NOLINTNEXTLINE(bugprone-random-generator-seed)
          sampling::RandomStream graph_rng(42);
          const auto first_token =
              reference.select(logits, settings, graph_rng);
          const std::array stop_ids{first_token};
          autonomous.start(settings, 42,
                           scenario == 2
                               ? std::span<const std::uint32_t>(stop_ids)
                               : std::span<const std::uint32_t>{});
          auto token_input = device->allocate({2 * sizeof(std::uint32_t), 8});
          auto prompt_ids = device->allocate({3 * sizeof(std::uint32_t), 8});
          auto prompt_staging =
              device->allocate_staging({3 * sizeof(std::uint32_t), 8});
          std::fill(prompt_staging.bytes().begin(),
                    prompt_staging.bytes().end(), std::byte{});
          queue.upload({&prompt_ids, 0, prompt_ids.size_bytes()},
                       {&prompt_staging, 0, prompt_staging.bytes().size()});
          auto captures = device->allocate({11 * sizeof(std::size_t), 8});
          auto injection_source =
              device->allocate({sizeof(float), alignof(float)});
          auto injection_staging =
              device->allocate_staging({sizeof(float), alignof(float)});
          std::unique_ptr<dolittle::inference::DeviceInjection> injection;
          if (scenario >= 3) {
            const float value =
                scenario == 3   ? 0.0F
                : scenario == 4 ? std::numeric_limits<float>::quiet_NaN()
                : scenario == 5 ? std::numeric_limits<float>::max()
                                : 2.0F;
            std::memcpy(injection_staging.bytes().data(), &value,
                        sizeof(value));
            queue.upload({&injection_source, 0, sizeof(value)},
                         {&injection_staging, 0, sizeof(value)});
            injection = std::make_unique<dolittle::inference::DeviceInjection>(
                *device, queue,
                gpu::ConstDeviceRegion{&injection_source, 0, sizeof(value)},
                fp32, fp32, 1, 150.0F);
          }
          using Sequence = dolittle::inference::DeviceSequence;
          Sequence loop(*device, queue, {3, 8, 2});
          loop.begin(autonomous.result_storage(), autonomous.terminal_state(),
                     {&token_input, 0, token_input.size_bytes()},
                     {&prompt_ids, 0, prompt_ids.size_bytes()},
                     injection ? injection->status() : gpu::ConstDeviceRegion{},
                     injection ? injection->norm() : gpu::ConstDeviceRegion{});
          if (injection) {
            injection->submit();
          }
          loop.begin_stage(Sequence::Stage::prefill);
          loop.end_stage();
          loop.begin_stage(Sequence::Stage::tail);
          loop.end_stage();
          loop.begin_stage(Sequence::Stage::first_selection);
          (void)autonomous.submit_next({&row, 0, bytes});
          loop.end_stage();
          loop.begin_stage(Sequence::Stage::decode);
          loop.capture(loop.position(), {&captures, 0, captures.size_bytes()},
                       sizeof(std::size_t), 1);
          loop.begin_selection();
          (void)autonomous.submit_next({&row, 0, bytes});
          loop.end_stage();
          loop.complete();
          if (scenario == 1) {
            loop.request_stop();
          }
          {
            const dolittle::diagnostics::AllocationScope scope(true);
            loop.start();
            queue
                .drain();  // No collection or acknowledgement until completion.
          }
          const bool invalid_injection = scenario >= 3 && scenario < 6;
          const unsigned expected_count = scenario == 1 || invalid_injection ? 0
                                          : scenario == 2                    ? 1
                                                          : 8;
          unsigned selected_count = 0;
          std::size_t consumed = 0;
          auto expected = first_token;
          for (std::size_t i = 0;; ++i) {
            const auto event = loop.collect(i, {});
            if (event.kind == Sequence::Kind::terminal) {
              const auto reason = invalid_injection ? Sequence::Reason::failure
                                  : scenario == 1 ? Sequence::Reason::cancelled
                                  : scenario == 2 ? Sequence::Reason::eos
                                                  : Sequence::Reason::limit;
              check(event.status == static_cast<unsigned>(reason) &&
                        event.position == consumed &&
                        event.count == expected_count &&
                        selected_count == expected_count,
                    "autonomous sequence terminal state");
              break;
            }
            if (event.kind == Sequence::Kind::injection) {
              check((event.status != 0) == invalid_injection,
                    "GPU normalization status");
            } else if (event.kind == Sequence::Kind::consumed) {
              check(event.position == consumed, "noncontiguous consumption");
              consumed += event.count;
            } else if (event.kind == Sequence::Kind::selected) {
              check(event.token == expected &&
                        event.position == 3 + selected_count,
                    "autonomous sequence changed selection/position");
              ++selected_count;
              if (selected_count < expected_count) {
                expected = reference.select(logits, settings, graph_rng);
              }
            }
          }
          if (expected_count) {
            auto exported = device->allocate_staging(
                {expected_count * sizeof(std::size_t), 8});
            queue.download(
                {&exported, 0, exported.bytes().size()},
                {&captures, 3 * sizeof(std::size_t), exported.bytes().size()});
            queue.drain();
            for (unsigned i = 0; i < expected_count; ++i) {
              std::size_t position{};
              std::memcpy(&position,
                          exported.bytes().subspan(i * sizeof(position)).data(),
                          sizeof(position));
              check(position == 3 + i,
                    "device captures lost absolute position");
            }
          }
        }
        sampling::DeviceCategorical stopped(
            *device, queue, n, type,
            sampling::DeviceCategorical::required_bytes(n));
        // A fixed seed is the reproducibility contract being checked.
        // NOLINTNEXTLINE(bugprone-random-generator-seed)
        sampling::RandomStream first_rng(42);
        const std::array stop_ids{
            reference.select(logits, settings, first_rng)};
        stopped.start(settings, 42, stop_ids);
        for (unsigned i = 0; i < 3; ++i) {
          const auto region = stopped.submit_next({&row, 0, bytes});
          queue.download({&answer, 0, 8}, region);
          queue.wait(queue.mark());
          std::array<std::uint32_t, 2> result{};
          std::memcpy(result.data(), answer.bytes().data(), 8);
          check(result.at(1) == 0 && result.at(0) == stop_ids.at(0),
                "device terminal latch changed a stopped selection");
        }
      }
      for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity()}) {
        std::fill(logits.begin(), logits.end(), invalid);
        upload_row();
        for (const auto mode : {sampling::SelectionMode::greedy,
                                sampling::SelectionMode::categorical}) {
          const auto region =
              selected.submit({&row, 0, bytes}, {{mode, 1.0, 0, 1.0}, 0});
          queue.download({&answer, 0, 8}, region);
          queue.wait(queue.mark());
          std::array<std::uint32_t, 2> result{};
          std::memcpy(result.data(), answer.bytes().data(), 8);
          check(result.at(1) != 0,
                "device sampling must reject invalid distribution");
        }
      }
    }
  }
  queue.drain();
  const nlohmann::json report{{"passed", true},
                              {"device", device->info().name},
                              {"architecture", device->info().architecture},
                              {"driver_api", device->info().driver_version}};
  std::cout << report.dump() << '\n';
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
