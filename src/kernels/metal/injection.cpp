/*
 * Metal implementation of the shared target-generation operations.
 *
 * Axis X5: Metal owner/encoding for inference::DeviceInjection. Own one
 * normalized output row plus FP32 norm and uint32 status, and borrow the
 * complete source capture through drain. Source/output cannot alias. The
 * source remains bitwise unchanged. required_bytes includes alignment and
 * the full output. Preparation checks dimensions, types, identity and extents.
 *
 * Algorithm: execution-boundary.md, H2/H3. Record injection.metal's norm,
 * validation and scaling dispatch before AV prefill. Its status gates the
 * sequence; its output replaces the marker embedding through the existing
 * DeviceSequence hook. Only the norm/status may be published as provenance;
 * the normalized vector never reaches the CPU. Persist output until all
 * embedding readers finish. Construct pipelines/constants before recording.
 *
 * Optimization: one O(D) dispatch, D*b+8 payload bytes plus alignment, no host
 * vector or per-token allocation. Implement this small backend primitive with
 * target kernels because the shared executor links the injection interface;
 * complete AV workflow/launcher acceptance remains the experiment phase.
 *
 * Header-design consultation, 2026-10-21: both local corpora searched and
 * cited rules read. Core I.7 (#ri-post) fixes numerical/completion results;
 * SL.con.3 (#rsl-bounds) requires checked extents before native indexing;
 * R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.1: normalize and inject without a vector readback.
 * guidelines/gpu/GPU.1-keep-data-on-device.md
 * GPU.9: declare output/status owners and retire after last reader.
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md
 */

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "gpu/metal/backend.h"
#include "gpu/metal/commands.h"
#include "inference/injection/device.h"
#include "kernels/metal/launch.h"
#include "tensors/tensor.h"
namespace dolittle::inference {
namespace {
using Access = gpu::detail::BackendAccess;
constexpr std::size_t kernel_threads = 256;
constexpr std::size_t output_offset = 8;
}  // namespace
struct InjectionSpecification {
  tensors::ElementType source_type, output_type;
  std::size_t width;
  float scale;
};
struct DeviceInjection::Impl {
  std::reference_wrapper<gpu::Device> device;
  std::reference_wrapper<gpu::Queue> queue;
  gpu::ConstDeviceRegion source;
  tensors::ElementType input_type, output_type;
  std::size_t width;
  float scale;
  gpu::DeviceBuffer storage;
  Impl(gpu::Device& d, gpu::Queue& q, gpu::ConstDeviceRegion s,
       InjectionSpecification spec)
      : device(d),
        queue(q),
        source(s),
        input_type(spec.source_type),
        output_type(spec.output_type),
        width(spec.width),
        scale(spec.scale),
        storage(d.allocate(
            {DeviceInjection::required_bytes(spec.width, spec.output_type),
             4})) {}
};
std::size_t DeviceInjection::required_bytes(std::size_t width,
                                            tensors::ElementType type) {
  if (type != tensors::ElementType::float32 &&
      type != tensors::ElementType::bfloat16) {
    throw std::invalid_argument("unsupported injection type");
  }
  const auto bytes = tensors::element_bytes(type);
  if (!width ||
      width >
          (std::numeric_limits<std::size_t>::max() - output_offset) / bytes) {
    throw std::length_error("injection capacity");
  }
  return output_offset + width * bytes;
}
DeviceInjection::DeviceInjection(gpu::Device& d, gpu::Queue& q,
                                 gpu::ConstDeviceRegion source,
                                 tensors::ElementType input_type,
                                 tensors::ElementType output_type,
                                 std::size_t width, float scale) {
  const auto expected = required_bytes(width, input_type) - output_offset;
  if (!source.owner || source.offset % tensors::element_bytes(input_type) ||
      source.bytes != expected || source.offset > source.owner->size_bytes() ||
      source.bytes > source.owner->size_bytes() - source.offset ||
      !std::isfinite(scale) || scale <= 0) {
    throw std::invalid_argument("invalid injection source");
  }
  (void)Access::binding(source, d);
  (void)Access::native_queue(q, d);
  impl_ = std::make_unique<Impl>(
      d, q, source,
      InjectionSpecification{input_type, output_type, width, scale});
}
DeviceInjection::~DeviceInjection() = default;
gpu::ConstDeviceRegion DeviceInjection::output() const {
  return {&impl_->storage, output_offset,
          impl_->storage.size_bytes() - output_offset};
}
gpu::ConstDeviceRegion DeviceInjection::norm() const {
  return {&impl_->storage, 0, sizeof(float)};
}
gpu::ConstDeviceRegion DeviceInjection::status() const {
  return {&impl_->storage, sizeof(float), sizeof(unsigned)};
}

void DeviceInjection::submit() {
  auto& p = *impl_;
  const std::array bindings{
      Access::binding(p.source, p.device),
      Access::binding(gpu::DeviceRegion{&p.storage, 0, p.storage.size_bytes()},
                      p.device)};
  kernels::metal::OperationArgs args;
  args.address.at(0) = bindings.at(0).address;
  args.address.at(1) = bindings.at(1).address;
  args.dimensions.at(0) = p.width;
  args.scalar = p.scale;
  args.type = p.input_type == tensors::ElementType::bfloat16 ? 1 : 0;
  args.flags = p.output_type == tensors::ElementType::bfloat16 ? 1 : 0;
  auto pipeline = gpu::metal::pipeline(p.device, DOLITTLE_METAL_LIBRARY,
                                       "injection_normalize");
  Access::dispatch(p.queue, p.device, *pipeline, bindings,
                   {kernel_threads, 1, 1}, {kernel_threads, 1, 1},
                   std::as_bytes(std::span(&args, 1)));
}
}  // namespace dolittle::inference
