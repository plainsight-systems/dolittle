/*
 * Metal implementation of the shared target-generation operations.
 *
 * Axis X5: Metal DeviceDirection and DeviceIntervention ownership/encoding.
 * Contract: borrow complete reconstruction leases or target residual/direction
 * until drain; own separate FP32 direction/status and intervention
 * status/count. Device, queue and immutable parameter snapshots outlive
 * recorded commands. Reject bounds/alias/type errors before submission. GPU
 * checks numerical errors.
 *
 * Algorithm: steering.md, Algorithm and numerical boundaries. Prepare and
 * dispatch steering.metal's reconstruction difference, retain its independent
 * FP32 vector after AR release, then bind it into the target's advertised
 * residual hook. Reset intervention status/count on GPU before replay; the
 * device position selects exactly one row. The target sequence reads status
 * before sampling and checks one application after replay. Direction.finish
 * may collect completion/status at the request boundary, never a vector.
 *
 * Cost O(D) vector storage/work and constant status/count. Optimization:
 * preallocate per request, reuse the immutable direction for target replay,
 * and record mutation in the body. No CPU norm, multiply or position decision.
 * These primitive implementations satisfy the shared executor link boundary;
 * matched baseline/steered application workflow belongs to the next phase.
 *
 * Header-design consultation, 2026-10-21: both local corpora searched and
 * cited rules read. Core I.7 (#ri-post) fixes numerical/completion results;
 * SL.con.3 (#rsl-bounds) requires checked extents before native indexing;
 * R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.1: direction and mutation stay on device.
 * guidelines/gpu/GPU.1-keep-data-on-device.md
 * GPU.6: record the hook as part of the reusable body.
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md
 * GPU.9: independent direction ownership survives AR weights.
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md
 */

#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

#include "gpu/metal/backend.h"
#include "gpu/metal/commands.h"
#include "inference/steering/device.h"
#include "kernels/metal/launch.h"
#include "tensors/tensor.h"
namespace dolittle::inference {
namespace {
using Access = gpu::detail::BackendAccess;
constexpr std::size_t kernel_threads = 256;
constexpr std::size_t vector_offset = 8;
gpu::metal::BufferBinding pointer(gpu::Device& d, gpu::ConstDeviceRegion region,
                                  std::size_t bytes, std::size_t alignment) {
  if (region.bytes != bytes || region.offset % alignment) {
    throw std::invalid_argument("steering region bounds/alignment");
  }
  return Access::binding(region, d);
}
void drain(gpu::Queue& q) noexcept {
  try {
    q.drain();
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Preserve the original execution error during owner teardown.
  }
}
void dispatch(gpu::Device& d, gpu::Queue& q, const char* name,
              const kernels::metal::OperationArgs& args,
              std::span<const gpu::metal::BufferBinding> bindings,
              std::size_t threads = kernel_threads) {
  auto pipeline = gpu::metal::pipeline(d, DOLITTLE_METAL_LIBRARY, name);
  Access::dispatch(q, d, *pipeline, bindings, {threads, 1, 1}, {threads, 1, 1},
                   std::as_bytes(std::span(&args, 1)));
}
}  // namespace
struct DeviceDirection::Impl {
  std::reference_wrapper<gpu::Device> device;
  std::reference_wrapper<gpu::Queue> queue;
  DeviceCapture original, edited;
  tensors::ElementType type;
  std::size_t width;
  std::shared_ptr<gpu::DeviceBuffer> storage;
  gpu::HostBuffer readback;
  bool submitted = false, completed = false;
  Impl(gpu::Device& d, gpu::Queue& q, DeviceCapture a, DeviceCapture b,
       tensors::ElementType t, std::size_t n)
      : device(d),
        queue(q),
        original(std::move(a)),
        edited(std::move(b)),
        type(t),
        width(n),
        storage(std::make_shared<gpu::DeviceBuffer>(
            d.allocate({required_bytes(n), 4}))),
        readback(d.allocate_staging({sizeof(unsigned), alignof(unsigned)})) {}
  ~Impl() { drain(queue); }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
};
std::size_t DeviceDirection::required_bytes(std::size_t width) {
  if (!width ||
      width > (std::numeric_limits<std::size_t>::max() - vector_offset) /
                  sizeof(float)) {
    throw std::length_error("steering width");
  }
  return vector_offset + width * sizeof(float);
}
DeviceDirection::DeviceDirection(gpu::Device& d, gpu::Queue& q, DeviceCapture a,
                                 DeviceCapture b, tensors::ElementType type,
                                 std::size_t width) {
  (void)required_bytes(width);
  if (type != tensors::ElementType::bfloat16 &&
      type != tensors::ElementType::float32) {
    throw std::invalid_argument("steering storage type");
  }
  const auto element = tensors::element_bytes(type);
  (void)pointer(d, a.region(), width * element, element);
  (void)pointer(d, b.region(), width * element, element);
  (void)Access::native_queue(q, d);
  impl_ = std::make_unique<Impl>(d, q, std::move(a), std::move(b), type, width);
}
DeviceDirection::~DeviceDirection() = default;

void DeviceDirection::submit() {
  auto& p = *impl_;
  if (p.submitted) throw std::logic_error("direction already submitted");
  const auto element = tensors::element_bytes(p.type);
  const std::array bindings{
      pointer(p.device, p.original.region(), p.width * element, element),
      pointer(p.device, p.edited.region(), p.width * element, element),
      Access::binding(
          gpu::DeviceRegion{p.storage.get(), 0, p.storage->size_bytes()},
          p.device)};
  kernels::metal::OperationArgs args;
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    args.address.at(i) = bindings.at(i).address;
  }
  args.dimensions.at(0) = p.width;
  args.type = p.type == tensors::ElementType::bfloat16 ? 1 : 0;
  dispatch(p.device, p.queue, "direction_difference", args, bindings);
  p.submitted = true;
}
DeviceCapture DeviceDirection::finish() {
  auto& p = *impl_;
  if (!p.submitted || p.completed) {
    throw std::logic_error("direction completion state");
  }
  p.queue.get().download({&p.readback, 0, sizeof(unsigned)},
                         {p.storage.get(), 0, sizeof(unsigned)});
  p.queue.get().drain();
  unsigned status{};
  std::memcpy(&status, p.readback.bytes().data(), sizeof(status));
  if (status) {
    throw std::invalid_argument(
        "reconstructions have zero or nonfinite difference");
  }
  p.completed = true;
  return {p.storage, vector_offset, p.width * sizeof(float)};
}
struct DeviceIntervention::Impl {
  std::reference_wrapper<gpu::Device> device;
  std::reference_wrapper<gpu::Queue> queue;
  ResidualIntervention input;
  tensors::ElementType type;
  gpu::DeviceBuffer state;
  Impl(gpu::Device& d, gpu::Queue& q, ResidualIntervention in,
       tensors::ElementType t)
      : device(d),
        queue(q),
        input(std::move(in)),
        type(t),
        state(d.allocate({required_bytes, 4})) {}
  ~Impl() { drain(queue); }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
};
DeviceIntervention::DeviceIntervention(gpu::Device& d, gpu::Queue& q,
                                       ResidualIntervention input,
                                       tensors::ElementType type) {
  (void)DeviceDirection::required_bytes(input.width);
  (void)pointer(d, input.direction.region(), input.width * sizeof(float),
                alignof(float));
  if (!std::isfinite(input.strength) || input.strength < 0 ||
      (type != tensors::ElementType::bfloat16 &&
       type != tensors::ElementType::float32)) {
    throw std::invalid_argument("invalid residual intervention");
  }
  impl_ = std::make_unique<Impl>(d, q, std::move(input), type);
}
DeviceIntervention::~DeviceIntervention() = default;
gpu::ConstDeviceRegion DeviceIntervention::status() const {
  return {&impl_->state, 0, sizeof(unsigned)};
}
gpu::ConstDeviceRegion DeviceIntervention::count() const {
  return {&impl_->state, sizeof(unsigned), sizeof(unsigned)};
}

void DeviceIntervention::initialize() {
  auto& p = *impl_;
  const auto binding = Access::binding(
      gpu::DeviceRegion{&p.state, 0, p.state.size_bytes()}, p.device);
  kernels::metal::OperationArgs args;
  args.address.at(2) = binding.address;
  dispatch(p.device, p.queue, "intervention_reset", args,
           std::span(&binding, 1), 1);
}
void DeviceIntervention::submit(gpu::DeviceRegion residual,
                                gpu::ConstDeviceRegion position,
                                std::size_t rows) {
  auto& p = *impl_;
  const auto& in = p.input;
  const auto element = tensors::element_bytes(p.type);
  if (!rows ||
      rows > std::numeric_limits<std::size_t>::max() / in.width / element) {
    throw std::length_error("intervention extent");
  }
  const std::array bindings{
      pointer(p.device, {residual.owner, residual.offset, residual.bytes},
              rows * in.width * element, element),
      pointer(p.device, in.direction.region(), in.width * sizeof(float),
              alignof(float)),
      Access::binding(gpu::DeviceRegion{&p.state, 0, p.state.size_bytes()},
                      p.device),
      pointer(p.device, position, sizeof(std::size_t), alignof(std::size_t))};
  kernels::metal::OperationArgs args;
  for (std::size_t i = 0; i < 3; ++i) {
    args.address.at(i) = bindings.at(i).address;
  }
  args.address.at(kernels::metal::position_slot) = bindings.at(3).address;
  args.dimensions.at(0) = in.width;
  args.dimensions.at(1) = rows;
  args.first = in.position.value;
  args.scalar = in.strength;
  args.type = p.type == tensors::ElementType::bfloat16 ? 1 : 0;
  dispatch(p.device, p.queue, "intervention_apply", args, bindings);
}
}  // namespace dolittle::inference
