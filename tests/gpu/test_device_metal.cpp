/*
 * D3 Metal instance of integration.device: resource transfer/completion plus
 * autonomous prepared sequence execution. Submit once, delay all collection
 * until drain, then verify prompt/decode captures and final pending
 * consumption. Separate cases cover EOS, zero budget, AR and GPU cancellation.
 * Tiny device fixture kernels replace model arithmetic only in this
 * resource/control test. Core I.7/R.1 (#ri-post/#rr-raii); performance
 * GPU.1/GPU.7/GDSA.4 retrieved for implementation:
 * guidelines/gpu/GPU.1-keep-data-on-device.md,
 * guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md,
 * guidelines/gpu-dsa/GDSA.4-no-cross-workgroup-waits-without-forward-progress.md.
 */
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "gpu/memory.h"
#include "gpu/metal/backend.h"
#include "gpu/metal/commands.h"
#include "gpu/metal/device.h"
#include "gpu/metal/support.h"
#include "inference/decoding/device.h"
#include "tests/support/check.h"
int device_contracts();
namespace {
namespace gpu = dolittle::gpu;
namespace inference = dolittle::inference;
namespace tests = dolittle::tests;
using Sequence = inference::DeviceSequence;
using Access = gpu::detail::BackendAccess;
void numeric(gpu::Device& device) {
  struct Vector {
    std::uint64_t a, b, operation;
  };
  std::vector<Vector> vectors;
  const std::array<std::uint64_t, 18> boundaries{0,
                                                 0x8000000000000000ULL,
                                                 1,
                                                 0x000fffffffffffffULL,
                                                 0x0010000000000000ULL,
                                                 0x0010000000000001ULL,
                                                 0x3ff0000000000000ULL,
                                                 0x3ff0000000000001ULL,
                                                 0x3fefffffffffffffULL,
                                                 0x3c90000000000000ULL,
                                                 0x3ca0000000000000ULL,
                                                 0x7fefffffffffffffULL,
                                                 0x7ff0000000000000ULL,
                                                 0xfff0000000000000ULL,
                                                 0x7ff8000000000000ULL,
                                                 0xbff0000000000000ULL,
                                                 0x8000000000000001ULL,
                                                 0xffefffffffffffffULL};
  for (auto a : boundaries) {
    for (auto b : boundaries) {
      for (unsigned op = 0; op < 7; ++op) vectors.push_back({a, b, op});
    }
  }
  // Fixed pseudorandom bits cover the full exponent range, independent of the
  // production generator and arithmetic. Expected results use native IEEE64.
  std::mt19937_64 random(195711);  // NOLINT(bugprone-random-generator-seed)
  for (std::size_t i = 0; i < 4096; ++i) {
    const auto a = random(), b = random();
    for (unsigned op = 0; op < 7; ++op) vectors.push_back({a, b, op});
    const double x =
        -745.0 + 1454.0 * static_cast<double>(random() >> 11) * 0x1p-53;
    vectors.push_back({std::bit_cast<std::uint64_t>(x), 0, 7});
    const double base =
        std::exp2(-30.0 + 60.0 * static_cast<double>(random() >> 11) * 0x1p-53);
    const double power = static_cast<double>(random() >> 11) * 0x1p-53;
    vectors.push_back({std::bit_cast<std::uint64_t>(base),
                       std::bit_cast<std::uint64_t>(power), 8});
  }
  for (std::size_t i = 0; i <= 100000; ++i) {
    const auto x = static_cast<double>(
        static_cast<float>(-105.0 * static_cast<double>(i) / 100000.0));
    vectors.push_back({std::bit_cast<std::uint64_t>(x), 0, 9});
    const auto norm = static_cast<double>(static_cast<float>(
        std::exp2(-24.0 + 48.0 * static_cast<double>(i) / 100000.0)));
    vectors.push_back({std::bit_cast<std::uint64_t>(norm), 0, 10});
  }
  const auto bytes = vectors.size() * sizeof(Vector),
             out_bytes = vectors.size() * sizeof(std::uint64_t);
  auto queue = device.create_queue();
  auto input = device.allocate({bytes, 8});
  auto output = device.allocate({out_bytes, 8});
  auto upload = device.allocate_staging({bytes, 8});
  auto readback = device.allocate_staging({out_bytes, 8});
  std::memcpy(upload.bytes().data(), vectors.data(), bytes);
  queue.upload({&input, 0, bytes}, {&upload, 0, bytes});
  const std::array bindings{
      Access::binding(gpu::ConstDeviceRegion{&input, 0, bytes}, device),
      Access::binding(gpu::DeviceRegion{&output, 0, out_bytes}, device)};
  auto pipeline = gpu::metal::pipeline(device, DOLITTLE_METAL_TEST_LIBRARY,
                                       "fixture_binary64");
  Access::dispatch(queue, device, *pipeline, bindings, {vectors.size(), 1, 1},
                   {128, 1, 1});
  queue.download({&readback, 0, out_bytes}, {&output, 0, out_bytes});
  queue.wait(queue.mark());
  std::vector<std::uint64_t> result(vectors.size());
  std::memcpy(result.data(), readback.bytes().data(), out_bytes);
  std::size_t failures = 0;
  for (std::size_t i = 0; i < vectors.size(); ++i) {
    const auto v = vectors.at(i);
    const auto a = std::bit_cast<double>(v.a), b = std::bit_cast<double>(v.b);
    double expected = 0;
    switch (v.operation) {
      case 0:
        expected = a + b;
        break;
      case 1:
        expected = a - b;
        break;
      case 2:
        expected = a * b;
        break;
      case 3:
        expected = a / b;
        break;
      case 4:
        expected = std::sqrt(a);
        break;
      case 5:
        expected = static_cast<float>(a);
        break;
      case 6:
        expected = std::bit_cast<float>(static_cast<std::uint32_t>(v.a));
        break;
      case 7:
        expected = std::exp(a);
        break;
      case 8:
        expected = std::pow(a, b);
        break;
      case 9:
        expected = static_cast<float>(std::exp(a));
        break;
      case 10:
        expected = static_cast<float>(1.0 / std::sqrt(a));
        break;
      default:
        throw std::logic_error("bad numeric vector");
    }
    const auto actual_bits = result.at(i);
    const auto expected_bits =
        (v.operation == 5 || v.operation >= 9)
            ? static_cast<std::uint64_t>(
                  std::bit_cast<std::uint32_t>(static_cast<float>(expected)))
            : std::bit_cast<std::uint64_t>(expected);
    const double actual = (v.operation == 5 || v.operation >= 9)
                              ? static_cast<double>(std::bit_cast<float>(
                                    static_cast<std::uint32_t>(actual_bits)))
                              : std::bit_cast<double>(actual_bits);
    const auto distance = actual_bits > expected_bits
                              ? actual_bits - expected_bits
                              : expected_bits - actual_bits;
    if (!(std::isnan(expected) && std::isnan(actual)) &&
        distance > (v.operation >= 7 && v.operation <= 9 ? 1U : 0U)) {
      if (failures++ < 8) {
        std::cerr << "numeric op=" << v.operation << " a=" << std::hex << v.a
                  << " b=" << v.b << " actual=" << actual_bits
                  << " expected=" << expected_bits << std::dec << '\n';
      }
    }
  }
  std::cout << "numeric vectors=" << vectors.size() << " failures=" << failures
            << '\n';
  tests::check(!failures,
               "Metal scalar arithmetic matches independent IEEE/libm vectors");
}

void resources(gpu::Device& device) {
  const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  auto queue = device.create_queue();
  auto viewed = Access::allocate_viewed(device, {5 * page, 4}, 2 * page);
  auto ordinary = device.allocate({5 * page, 4});
  auto host = device.allocate_staging({5 * page, 4});
  std::ranges::fill(host.bytes(), std::byte{0x5a});
  queue.upload({&viewed, 0, 5 * page}, {&host, 0, 5 * page});
  const auto uploaded = queue.mark();
  queue.wait(uploaded);
  const auto crossing =
      Access::binding(gpu::ConstDeviceRegion{&viewed, 2 * page - 4, 8}, device);
  tests::check(crossing.bytes == 8 && crossing.offset + crossing.bytes <=
                                          crossing.buffer->length(),
               "whole crossing operand has a native view");
  queue.copy({&ordinary, 0, 5 * page}, {&viewed, 0, 5 * page});
  queue.download({&host, 0, 5 * page}, {&ordinary, 0, 5 * page});
  queue.wait(queue.mark());
  tests::check(
      std::ranges::all_of(host.bytes(),
                          [](auto value) { return value == std::byte{0x5a}; }),
      "split view transfers preserve physical backing");
  tests::expect_throws<std::invalid_argument>(
      [&] { queue.copy({&viewed, 1, 8}, {&viewed, 0, 8}); });
  tests::expect_throws<std::invalid_argument>(
      [&] { queue.wait({uploaded.queue_identity, uploaded.sequence + 100}); });
  auto other = device.create_queue();
  tests::expect_throws<std::invalid_argument>([&] { other.wait(uploaded); });
  tests::expect_throws<std::length_error>([&] {
    (void)Access::binding(gpu::ConstDeviceRegion{&viewed, page - 4, 2 * page},
                          device);
  });
}
void sequence(gpu::Device& device, std::size_t generated, std::uint32_t eos,
              bool cancel, bool forward_only = false) {
  auto queue = device.create_queue();
  const std::array<std::uint32_t, 5> prompt{1, 2, 3, 4, 5};
  auto ids = device.allocate({sizeof(prompt), 4});
  auto tokens = device.allocate({8, 4});
  auto selected = device.allocate({8, 4});
  auto terminal = device.allocate({4, 4});
  auto eos_device = device.allocate({4, 4});
  auto captures = device.allocate({(prompt.size() + generated) * 4, 4});
  auto upload = device.allocate_staging({sizeof(prompt), 4});
  std::memcpy(upload.bytes().data(), prompt.data(), sizeof(prompt));
  queue.upload({&ids, 0, sizeof(prompt)}, {&upload, 0, sizeof(prompt)});
  queue.drain();
  std::memcpy(upload.bytes().data(), &eos, sizeof(eos));
  queue.upload({&eos_device, 0, 4}, {&upload, 0, 4});
  queue.drain();
  Sequence sequence(device, queue, {prompt.size(), generated, 2, forward_only});
  sequence.begin({&selected, 0, 8}, {&terminal, 0, 4}, {&tokens, 0, 8},
                 {&ids, 0, sizeof(prompt)});
  auto pipeline = gpu::metal::pipeline(device, DOLITTLE_METAL_TEST_LIBRARY,
                                       "fixture_select");
  const auto append_select = [&] {
    const std::array bindings{
        Access::binding(gpu::ConstDeviceRegion{&tokens, 0, 8}, device),
        Access::binding(gpu::DeviceRegion{&selected, 0, 8}, device),
        Access::binding(gpu::DeviceRegion{&terminal, 0, 4}, device),
        Access::binding(gpu::ConstDeviceRegion{&eos_device, 0, 4}, device)};
    Access::dispatch(queue, device, *pipeline.get(), bindings, {1, 1, 1},
                     {1, 1, 1});
  };
  sequence.begin_stage(Sequence::Stage::prefill);
  sequence.capture({&tokens, 0, 8}, {&captures, 0, captures.size_bytes()}, 4,
                   2);
  sequence.end_stage();
  sequence.begin_stage(Sequence::Stage::tail);
  sequence.capture({&tokens, 0, 4}, {&captures, 0, captures.size_bytes()}, 4,
                   1);
  sequence.end_stage();
  if (generated || forward_only) {
    sequence.begin_stage(Sequence::Stage::first_selection);
    append_select();
    sequence.end_stage();
  }
  if (generated) {
    sequence.begin_stage(Sequence::Stage::decode);
    sequence.capture({&tokens, 0, 4}, {&captures, 0, captures.size_bytes()}, 4,
                     1);
    sequence.begin_selection();
    append_select();
    sequence.end_stage();
  }
  sequence.complete();
  const auto resources = sequence.prepared_resources();
  tests::check(resources.command_bytes > 0 && resources.owned.unified_bytes >=
                                                  resources.owned.device_bytes,
               "native commands accounted");
  const auto began = std::chrono::steady_clock::now();
  sequence.start();
  if (cancel) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    sequence.request_stop();
  }
  // The GPU must finish without any collection acknowledgment.
  sequence.drain();
  std::size_t consumed = 0, selections = 0, index = 0;
  for (;;) {
    const auto event = sequence.collect(index++, {});
    if (event.kind == Sequence::Kind::consumed) {
      tests::check(event.position == consumed, "contiguous consumption");
      consumed += event.count;
    } else if (event.kind == Sequence::Kind::selected) {
      tests::check(event.position == prompt.size() + selections &&
                       event.token == 6 + selections,
                   "device recurrence publication");
      ++selections;
    } else if (event.kind == Sequence::Kind::terminal) {
      tests::check(event.count == selections && event.position == consumed,
                   "terminal agrees with published work");
      tests::check(
          event.status ==
              static_cast<unsigned>(cancel ? Sequence::Reason::cancelled
                                           : (eos ? Sequence::Reason::eos
                                                  : Sequence::Reason::limit)),
          "expected terminal reason");
      break;
    }
  }
  tests::check(
      consumed == prompt.size() + selections || (cancel && selections == 0),
      "all selected tokens consumed");
  if (!cancel) {
    tests::check(selections == (eos ? eos - 5 : generated),
                 "expected generated count");
  }
  if (cancel) {
    tests::check(selections < generated,
                 "cancellation reached active GPU work");
  }
  auto download = device.allocate_staging({consumed * 4, 4});
  queue.download({&download, 0, consumed * 4}, {&captures, 0, consumed * 4});
  queue.wait(queue.mark());
  std::vector<std::uint32_t> result(consumed);
  std::memcpy(result.data(), download.bytes().data(), download.bytes().size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    tests::check(result.at(i) == i + 1, "capture uses GPU position");
  }
  std::cout << "sequence G=" << generated << " consumed=" << consumed
            << " selected=" << selections << " elapsed_ms="
            << std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - began)
                   .count()
            << '\n';
}
}  // namespace
int main() {
  try {
    const auto autorelease = dolittle::gpu::metal::pool();
    auto device = dolittle::gpu::metal::open_device();
    const auto memory = dolittle::gpu::query_memory(*device);
    dolittle::tests::check(
        memory.topology == dolittle::gpu::MemoryTopology::unified &&
            memory.max_allocation_bytes > 0,
        "Metal memory capabilities");
    std::cout << device->info().name << '\n';
    numeric(*device);
    if (device_contracts()) return 1;
    resources(*device);
    sequence(*device, 8, 0, false);
    sequence(*device, 8, 8, false);
    sequence(*device, 0, 0, false);
    sequence(*device, 0, 0, false, true);
    sequence(*device, 512, 0, true);
  } catch (const std::exception& error) {
    return dolittle::tests::report_failure(error.what());
  }
}
