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
 */
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "gpu/cuda/device.h"
#include "kernels/operations.h"
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
