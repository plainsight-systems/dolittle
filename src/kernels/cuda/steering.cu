/*
 * Axis X5: CUDA reconstruction difference and one-row residual steering.
 * Implements inference/steering/device.h; overlay steering.md, Algorithm and
 * numerical boundaries. Inputs are complete D-element reconstruction leases
 * or the branch's writable [B,D] residual, FP32 direction and device position.
 * Eight lanes sum FP32 squares in feature-strided order, combine lanes 0..7,
 * and take FP32 sqrt. Difference subtraction, divide, norm scaling and addition
 * use separate rounding. Store direction in FP32; narrow updated residual once.
 * Only the chunk containing absolute p writes row p-S, and increments count.
 * Invalid values latch device status; the sequence consumes that status before
 * sampling. Prepared owners drain before releasing borrowed resources. Bounds,
 * alignment and device identity are checked before submission. O(D) work and
 * preallocated storage, no numerical readback or per-token allocation.
 * Optimization: one block performs each small vector operation; immutable
 * direction storage survives the AR model and is shared with the target plan.
 * Both corpora queried/read for implementation, 2026-10-20: Core I.7
 * (#ri-post), explicit postconditions; performance GPU.1, resident values,
 * and GDSA.2, fixed reductions independent of sequence shape. Sources:
 * guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 */
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

#include "gpu/cuda/backend.h"
#include "inference/steering/device.h"
#include "kernels/cuda/common.cuh"
#include "tensors/tensor.h"

namespace dolittle::inference {
namespace {
using Access = gpu::detail::BackendAccess;
namespace detail = kernels::cuda::detail;
constexpr unsigned lanes = 8, threads = 256;
constexpr std::size_t vector_offset = 2 * sizeof(unsigned);

const std::byte* pointer(gpu::Device& d, gpu::ConstDeviceRegion r,
                         std::size_t bytes, std::size_t alignment) {
  if (!r.owner || r.bytes != bytes || r.offset % alignment ||
      r.offset > r.owner->size_bytes() ||
      r.bytes > r.owner->size_bytes() - r.offset) {
    throw std::invalid_argument("steering region bounds/alignment");
  }
  return std::span(static_cast<const std::byte*>(Access::pointer(*r.owner, d)),
                   r.owner->size_bytes())
      .subspan(r.offset, r.bytes)
      .data();
}
void drain(gpu::Queue& q) noexcept {
  try {
    q.drain();
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Preserve the original execution error during owner teardown.
  }
}
// Native indices are bounded by the admitted D and row count. One block owns
// status/count; block barriers order reductions and status publication.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index)
__global__ void reset(unsigned* words) {
  words[0] = 0;
  words[1] = 0;
}
template <class Input>
__global__ void difference(const Input* original, const Input* edited,
                           float* output, unsigned* status, std::size_t width) {
  __shared__ float sums[lanes], norm;
  if (threadIdx.x < lanes) {
    float sum = 0;
    for (std::size_t i = threadIdx.x; i < width; i += lanes) {
      const float delta =
          __fsub_rn(detail::widen(edited[i]), detail::widen(original[i]));
      sum = __fadd_rn(sum, __fmul_rn(delta, delta));
    }
    sums[threadIdx.x] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0;
    for (unsigned i = 0; i < lanes; ++i) {
      sum = __fadd_rn(sum, sums[i]);
    }
    norm = __fsqrt_rn(sum);
    *status = !isfinite(norm) || norm == 0 ? 1U : 0U;
  }
  __syncthreads();
  if (*status) {
    return;
  }
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    output[i] = __fdiv_rn(
        __fsub_rn(detail::widen(edited[i]), detail::widen(original[i])), norm);
  }
}
struct Mutation {
  const float* direction;
  const std::size_t* position;
  unsigned* state;
  std::size_t width, selected, rows;
  float strength;
};
template <class Value>
__global__ void intervene(Value* residual, Mutation args) {
  const auto [direction, position, state, width, selected, rows, strength] =
      args;
  const auto first = *position;
  if (selected < first || selected - first >= rows) {
    return;
  }
  __shared__ float sums[lanes], beta;
  __shared__ unsigned invalid;
  if (threadIdx.x == 0) {
    invalid = state[0] || state[1] != 0 ? 1U : 0U;
    ++state[1];
  }
  __syncthreads();
  auto* row = residual + (selected - first) * width;
  if (threadIdx.x < lanes) {
    float sum = 0;
    for (std::size_t i = threadIdx.x; i < width; i += lanes) {
      const float value = detail::widen(row[i]);
      sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    sums[threadIdx.x] = sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0;
    for (unsigned i = 0; i < lanes; ++i) {
      sum = __fadd_rn(sum, sums[i]);
    }
    beta = __fmul_rn(strength, __fsqrt_rn(sum));
    if (!isfinite(beta)) {
      invalid = 1;
    }
  }
  __syncthreads();
  const bool valid = invalid == 0;
  __syncthreads();
  if (valid) {
    for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
      const float value =
          __fadd_rn(detail::widen(row[i]), __fmul_rn(beta, direction[i]));
      const auto stored = detail::narrow<Value>(value);
      if (!isfinite(value) || !isfinite(detail::widen(stored))) {
        atomicOr(&invalid, 1U);
      }
      row[i] = stored;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    state[0] = invalid;
  }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index)
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
  (void)Access::stream(q, d);
  impl_ = std::make_unique<Impl>(d, q, std::move(a), std::move(b), type, width);
}
DeviceDirection::~DeviceDirection() = default;
void DeviceDirection::submit() {
  auto& p = *impl_;
  if (p.submitted) {
    throw std::logic_error("direction already submitted");
  }
  const auto stream = Access::stream(p.queue, p.device);
  const auto element = tensors::element_bytes(p.type);
  const auto* a =
      pointer(p.device, p.original.region(), p.width * element, element);
  const auto* b =
      pointer(p.device, p.edited.region(), p.width * element, element);
  auto bytes =
      std::span(static_cast<std::byte*>(Access::pointer(*p.storage, p.device)),
                p.storage->size_bytes());
  // Typed CUDA pointers refer to checked, naturally aligned device storage.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* status = reinterpret_cast<unsigned*>(bytes.data());
  auto* out = reinterpret_cast<float*>(bytes.subspan(vector_offset).data());
  if (p.type == tensors::ElementType::bfloat16) {
    difference<<<1, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b), out, status, p.width);
  } else {
    difference<<<1, threads, 0, stream>>>(reinterpret_cast<const float*>(a),
                                          reinterpret_cast<const float*>(b),
                                          out, status, p.width);
  }
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  Access::check_launch(p.device);
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
  reset<<<1, 1, 0, Access::stream(p.queue, p.device)>>>(
      static_cast<unsigned*>(Access::pointer(p.state, p.device)));
  Access::check_launch(p.device);
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
  (void)pointer(p.device, {residual.owner, residual.offset, residual.bytes},
                rows * in.width * element, element);
  auto bytes =
      std::span(
          static_cast<std::byte*>(Access::pointer(*residual.owner, p.device)),
          residual.owner->size_bytes())
          .subspan(residual.offset, residual.bytes);
  // Native pointers consume validated alignment/extents, never host values.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  const Mutation args{
      reinterpret_cast<const float*>(pointer(p.device, in.direction.region(),
                                             in.width * sizeof(float),
                                             alignof(float))),
      reinterpret_cast<const std::size_t*>(pointer(
          p.device, position, sizeof(std::size_t), alignof(std::size_t))),
      static_cast<unsigned*>(Access::pointer(p.state, p.device)),
      in.width,
      in.position.value,
      rows,
      in.strength};
  const auto stream = Access::stream(p.queue, p.device);
  if (p.type == tensors::ElementType::bfloat16) {
    intervene<<<1, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(bytes.data()), args);
  } else {
    intervene<<<1, threads, 0, stream>>>(reinterpret_cast<float*>(bytes.data()),
                                         args);
  }
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  Access::check_launch(p.device);
}
}  // namespace dolittle::inference
