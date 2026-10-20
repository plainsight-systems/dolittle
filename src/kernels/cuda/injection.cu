/*
 * Axis X5: CUDA implementation of inference/injection/device.h, H2.
 * Contract: source/output do not alias; typed regions are checked once,
 * borrowed through completion and never exposed to CPU arithmetic. All
 * validation is GPU work; status gates the later sequence. Input remains
 * bitwise unchanged. Algorithm: eight lanes sum FP32 squares at i=lane+8*k,
 * then lane zero combines those sums left-to-right and computes sqrt. All lanes
 * divide by the same norm, multiply by the trained scale and narrow once.
 * Source/nonfinite/norm/output checks latch status. Explicit intrinsics
 * preserve separate rounding boundaries. One block uses eight shared sums and
 * one failure flag; threads cooperatively write the output row. O(D) work and
 * D*b+8 device bytes, no host temporary/copy. Optimization: submission fits the
 * root graph; no allocation or readback in submit. Owner destruction follows
 * queue completion in the sequence executor. Design/implementation corpora
 * queried/read 2026-10-20: Core I.1/R.1 for explicit status and ownership;
 * Performance GPU.1/GPU.9 for device-only intermediate storage
 * (guidelines/gpu/GPU.1-keep-data-on-device.md). Overlay section 5.
 */
#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "gpu/cuda/backend.h"
#include "inference/injection/device.h"
#include "kernels/cuda/common.cuh"
#include "tensors/tensor.h"
namespace dolittle::inference {
namespace {
using Access = gpu::detail::BackendAccess;
namespace detail = kernels::cuda::detail;
constexpr std::size_t output_offset = sizeof(float) + sizeof(unsigned);
constexpr unsigned lanes = 8, threads = 256;
// Every native array index is bounded by D or the eight-lane reduction. Shared
// failure writes use a block-local atomic, followed by a block barrier.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index)
template <class Input, class Output>
struct NormalizationArguments {
  const Input* source;
  Output* output;
  float* norm;
  unsigned* status;
  std::size_t width;
  float scale;
};
template <class Input, class Output>
__global__ void normalize(NormalizationArguments<Input, Output> args) {
  const auto [source, output, norm, status, width, scale] = args;
  __shared__ float sums[lanes];
  __shared__ unsigned invalid;
  if (threadIdx.x == 0) {
    invalid = 0;
  }
  __syncthreads();
  if (threadIdx.x < lanes) {
    float sum = 0;
    for (std::size_t i = threadIdx.x; i < width; i += lanes) {
      const float value = detail::widen(source[i]);
      if (!isfinite(value)) {
        atomicOr(&invalid, 1U);
      }
      sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    sums[threadIdx.x] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0;
    for (unsigned lane = 0; lane < lanes; ++lane) {
      sum = __fadd_rn(sum, sums[lane]);
    }
    *norm = __fsqrt_rn(sum);
    if (!isfinite(*norm) || *norm == 0) {
      invalid = 1;
    }
  }
  __syncthreads();
  // Snapshot block validation before any output-overflow latch changes it.
  const bool valid = invalid == 0;
  __syncthreads();
  if (valid) {
    for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
      const float value =
          __fmul_rn(__fdiv_rn(detail::widen(source[i]), *norm), scale);
      const auto stored = detail::narrow<Output>(value);
      if (!isfinite(value) || !isfinite(detail::widen(stored))) {
        atomicOr(&invalid, 1U);
      }
      output[i] = stored;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *status = invalid;
  }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index)
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
  (void)Access::pointer(*source.owner, d);
  (void)Access::stream(q, d);
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
  const auto stream = Access::stream(p.queue, p.device);
  const auto* source = std::span(static_cast<const std::byte*>(Access::pointer(
                                     *p.source.owner, p.device)),
                                 p.source.owner->size_bytes())
                           .subspan(p.source.offset)
                           .data();
  auto* base = static_cast<std::byte*>(Access::pointer(p.storage, p.device));
  auto bytes = std::span(base, p.storage.size_bytes());
  // Typed CUDA boundaries: byte sizes, owners and alignment checked at
  // preparation. NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* norm = reinterpret_cast<float*>(base);
  auto* status =
      reinterpret_cast<unsigned*>(bytes.subspan(sizeof(float)).data());
  auto* output = bytes.subspan(output_offset).data();
  const auto launch = [&]<class Input, class Output>() {
    normalize<<<1, threads, 0, stream>>>(NormalizationArguments<Input, Output>{
        .source = reinterpret_cast<const Input*>(source),
        .output = reinterpret_cast<Output*>(output),
        .norm = norm,
        .status = status,
        .width = p.width,
        .scale = p.scale});
  };
  if (p.input_type == tensors::ElementType::float32) {
    if (p.output_type == tensors::ElementType::float32) {
      launch.template operator()<float, float>();
    } else {
      launch.template operator()<float, __nv_bfloat16>();
    }
  } else {
    if (p.output_type == tensors::ElementType::float32) {
      launch.template operator()<__nv_bfloat16, float>();
    } else {
      launch.template operator()<__nv_bfloat16, __nv_bfloat16>();
    }
  }
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  Access::check_launch(p.device);
}
}  // namespace dolittle::inference
