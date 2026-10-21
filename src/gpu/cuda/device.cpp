/*
 * Shared memory admission (metal-port.md, Memory and model residency):
 * query_memory reads CUDA's current free/total bytes without allocating or
 * synchronizing; it returns a discrete-memory snapshot, not a reservation.
 * Implementation consultation: Core I.7 (#ri-post) makes that limit explicit;
 * GPU.9 (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md)
 * requires persistent and upcoming transient owners to share one budget.
 * Both rules were read from the local corpora for this implementation.
 *
 * Axis X4, CUDA kind: resource ownership, transfers and completion.
 * Contract: implement gpu/device.h and the explicit CUDA factory; preserve
 * buffer lifetime through outstanding submissions and never recycle in-use
 * storage. Vendor handles remain backend-private. Explicit waits report
 * deferred errors; noexcept cleanup drains safely and poisons unrecoverable
 * devices. Algorithm: transformer overlay, "Preparing the first execution
 * recipe" and "Storage lifetimes": initialize the selected device; create an
 * explicit ordered stream; allocate device arenas and bounded pinned staging;
 * validate regions; conservatively drain device work on buffer destruction;
 * record queue prefix events; wait/publish completion; release only after all
 * uses retire. This teardown drain also covers partial-launch failures without
 * per-submission resource allocation. Explicit waits use prefix events, never a
 * device-wide wait. Design and cost: allocations/registration/event resources
 * have explicit setup and teardown costs. Optimization: reuse staging and
 * completion bookkeeping; no legacy-default-stream assumptions or blanket
 * synchronize per observation. A private bridge supplies validated native
 * regions/stream to CUDA kernel launch code; conservative cleanup covers
 * resource lifetime on failure too. It exposes no native handles to model code.
 * Completed event slots are reused; live marks can grow host/event bookkeeping.
 * GPU execution/fault/teardown evidence is integration work, not unit-test
 * setup. Overlay: docs/architecture/transformer-execution.md. Corpus
 * consultation, implementation, 2026-10-18: cpp-guidelines R.1 (#rr-raii):
 * resource ownership survives exceptions and teardown. cpp-perf-guidelines
 * GPU.7: completion tracks resource dependencies, not global waits. Source:
 * guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 *
 * Static-analysis design/implementation, 2026-10-19:
 * Region checks precede bounded byte views passed to CUDA. Alignment uses the
 * allocation padding reserved above; queue identities are synchronized atomics.
 * cpp-guidelines SL.con.3 (#rsl-bounds), I.30 (#ri-encapsulate): make checked
 * host access explicit and confine necessary native API operations.
 * cpp-perf-guidelines COPY.7: borrow spans/references without hidden payload
 * copies. Source: guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 * Steady-state design, 2026-10-20:
 * Warm one reusable completion event when creating a queue. Device/staging
 * allocation records a diagnostic allocation violation when invoked inside
 * the guarded generation loop; queue mark/wait reuse completed events.
 * Overlay: transformer-execution.md, "Device selection and allocation-free
 * generation". Core I.7 (#ri-post): borrowing, bounds and failure contracts.
 * Performance GPU.1: device-resident decisions; MEM.9: prepare storage before
 * steady state and check the invariant. Sources:
 * guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/memory/MEM.9-embedded-allocate-at-init.md.
 * Both corpora queried/read for header design and implementation, 2026-10-20.
 * Queue-completion design/implementation consultation, 2026-10-20:
 * Core I.7 (#ri-post) and CP.3 (#rconc-data): collection establishes host
 * readiness; in-flight slots have exclusive ownership until completion.
 * Performance GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md) keeps
 * sampled IDs and RNG state on-device. GPU.7
 * (guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md) separates submission
 * from readback through explicit prefix completion. Live device-owned decode
 * uses its own graph publication protocol instead of per-token events. Both
 * local corpora were queried and the full rules read in both work phases.
 */
#include "gpu/cuda/device.h"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "diagnostics/allocation.h"
#include "gpu/cuda/backend.h"
#include "gpu/memory.h"

namespace dolittle::gpu {
namespace {
inline constexpr int minimum_compute_major = 8;
inline constexpr int minimum_compute_minor = 6;
inline constexpr int driver_major_scale = 1000;
inline constexpr int driver_minor_scale = 10;

void require(bool value, const char* message) {
  if (!value) {
    throw std::invalid_argument(message);
  }
}

struct State {
  int ordinal{};
  cudaDeviceProp properties{};
  DeviceInfo info;
  bool poisoned = false;

  void check(cudaError_t error, bool completion = false) {
    if (error == cudaSuccess) {
      return;
    }
    if (completion ||
        (error != cudaErrorMemoryAllocation && error != cudaErrorInvalidValue &&
         error != cudaErrorInvalidConfiguration &&
         error != cudaErrorLaunchOutOfResources)) {
      poisoned = true;
    }
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(error));
  }
  void activate() {
    if (poisoned) {
      throw std::runtime_error("CUDA device is poisoned; recreate resources");
    }
    check(cudaSetDevice(ordinal));
  }
  void drain_noexcept() noexcept {
    if (cudaSetDevice(ordinal) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      poisoned = true;
    }
  }
};

class CudaDevice final : public Device {
 public:
  explicit CudaDevice(std::shared_ptr<State> state)
      : state_(std::move(state)) {}
  CudaDevice(const CudaDevice&) = delete;
  CudaDevice& operator=(const CudaDevice&) = delete;
  CudaDevice(CudaDevice&&) = delete;
  CudaDevice& operator=(CudaDevice&&) = delete;
  ~CudaDevice() override { state_->drain_noexcept(); }
  const DeviceInfo& info() const noexcept override { return state_->info; }
  DeviceBuffer allocate(AllocationRequest r) override {
    diagnostics::record_allocation();
    return detail::BackendAccess::allocate(*this, r);
  }
  HostBuffer allocate_staging(AllocationRequest r) override {
    diagnostics::record_allocation();
    return detail::BackendAccess::allocate_staging(*this, r);
  }
  Queue create_queue() override {
    return detail::BackendAccess::create_queue(*this);
  }
  std::shared_ptr<State> state_;
};

std::shared_ptr<State> device_state(Device& device) {
  auto* native = dynamic_cast<CudaDevice*>(&device);
  require(native != nullptr, "expected CUDA device");
  native->state_->activate();
  return native->state_;
}

std::size_t allocation_bytes(AllocationRequest r) {
  require(r.alignment != 0 && (r.alignment & (r.alignment - 1)) == 0,
          "allocation alignment must be a power of two");
  if (r.bytes == 0) {
    return 0;
  }
  if (r.bytes > std::numeric_limits<std::size_t>::max() - (r.alignment - 1)) {
    throw std::length_error("aligned allocation size overflows");
  }
  return r.bytes + r.alignment - 1;
}

void* aligned_pointer(void* base, std::size_t alignment) {
  if (!base) {
    return nullptr;
  }
  // allocation_bytes reserves bytes + alignment - 1, so at least alignment
  // bytes remain available for this one-byte alignment probe.
  auto space = alignment;
  return std::align(alignment, 1, base, space);
}

void check_range(std::size_t capacity, std::size_t offset, std::size_t bytes) {
  if (offset > std::numeric_limits<std::size_t>::max() - bytes) {
    throw std::length_error("buffer range overflows");
  }
  require(offset <= capacity && bytes <= capacity - offset,
          "buffer range exceeds allocation");
}

std::uint64_t next_queue_identity() {
  // Process-wide monotonic IDs must be shared and mutable; all access is
  // atomic. NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::atomic<std::uint64_t> identity{1};
  auto value = identity.load(std::memory_order_relaxed);
  while (true) {
    if (value == std::numeric_limits<std::uint64_t>::max()) {
      throw std::length_error("queue identity exhausted");
    }
    if (identity.compare_exchange_weak(value, value + 1,
                                       std::memory_order_relaxed)) {
      break;
    }
  }
  return value;
}

}  // namespace

struct DeviceBuffer::Impl {
  std::shared_ptr<State> state;
  void* base = nullptr;
  void* pointer = nullptr;
  std::size_t bytes = 0;
  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    if (base) {
      state->drain_noexcept();
      if (cudaFree(base) != cudaSuccess) {
        state->poisoned = true;
      }
    }
  }
};
struct HostBuffer::Impl {
  std::shared_ptr<State> state;
  void* base = nullptr;
  std::byte* pointer = nullptr;
  std::size_t bytes = 0;
  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    if (base) {
      state->drain_noexcept();
      if (cudaFreeHost(base) != cudaSuccess) {
        state->poisoned = true;
      }
    }
  }
};
struct Queue::Impl {
  struct Mark {
    cudaEvent_t event = nullptr;
    std::uint64_t sequence = 0;
  };
  std::shared_ptr<State> state;
  cudaStream_t stream = nullptr;
  std::uint64_t identity = next_queue_identity();
  std::uint64_t issued = 0;
  std::uint64_t completed = 0;
  std::vector<Mark> marks;
  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    if (!stream) {
      return;
    }
    if (cudaSetDevice(state->ordinal) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      state->poisoned = true;
    }
    for (auto& mark : marks) {
      if (mark.event && cudaEventDestroy(mark.event) != cudaSuccess) {
        state->poisoned = true;
      }
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
      state->poisoned = true;
    }
  }
};

Device::Device() = default;
Device::~Device() = default;
DeviceBuffer::DeviceBuffer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DeviceBuffer::~DeviceBuffer() = default;
DeviceBuffer::DeviceBuffer(DeviceBuffer&&) noexcept = default;
DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&&) noexcept = default;
std::size_t DeviceBuffer::size_bytes() const noexcept { return impl_->bytes; }
HostBuffer::HostBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
HostBuffer::~HostBuffer() = default;
HostBuffer::HostBuffer(HostBuffer&&) noexcept = default;
HostBuffer& HostBuffer::operator=(HostBuffer&&) noexcept = default;
std::span<std::byte> HostBuffer::bytes() noexcept {
  return {impl_->pointer, impl_->bytes};
}
std::span<const std::byte> HostBuffer::bytes() const noexcept {
  return {impl_->pointer, impl_->bytes};
}
Queue::Queue(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Queue::~Queue() = default;
Queue::Queue(Queue&&) noexcept = default;
Queue& Queue::operator=(Queue&&) noexcept = default;

DeviceBuffer detail::BackendAccess::allocate(Device& device,
                                             AllocationRequest request) {
  const auto bytes = allocation_bytes(request);
  auto impl = std::make_unique<DeviceBuffer::Impl>();
  impl->state = device_state(device);
  if (bytes) {
    impl->state->check(cudaMalloc(&impl->base, bytes));
  }
  impl->pointer = aligned_pointer(impl->base, request.alignment);
  impl->bytes = request.bytes;
  return DeviceBuffer(std::move(impl));
}
HostBuffer detail::BackendAccess::allocate_staging(Device& device,
                                                   AllocationRequest request) {
  const auto bytes = allocation_bytes(request);
  auto impl = std::make_unique<HostBuffer::Impl>();
  impl->state = device_state(device);
  if (bytes) {
    impl->state->check(cudaHostAlloc(&impl->base, bytes, cudaHostAllocDefault));
  }
  impl->pointer =
      static_cast<std::byte*>(aligned_pointer(impl->base, request.alignment));
  impl->bytes = request.bytes;
  return HostBuffer(std::move(impl));
}
Queue detail::BackendAccess::create_queue(Device& device) {
  auto impl = std::make_unique<Queue::Impl>();
  impl->state = device_state(device);
  impl->state->check(
      cudaStreamCreateWithFlags(&impl->stream, cudaStreamNonBlocking));
  impl->marks.emplace_back();
  impl->state->check(cudaEventCreateWithFlags(&impl->marks.back().event,
                                              cudaEventDisableTiming));
  return Queue(std::move(impl));
}
cudaStream_t detail::BackendAccess::stream(Queue& queue, Device& device) {
  auto state = device_state(device);
  require(queue.impl_ && queue.impl_->state == state,
          "queue belongs to another device");
  return queue.impl_->stream;
}
void* detail::BackendAccess::pointer(DeviceBuffer& buffer, Device& device) {
  auto state = device_state(device);
  require(buffer.impl_ && buffer.impl_->state == state,
          "buffer belongs to another device");
  return buffer.impl_->pointer;
}
const void* detail::BackendAccess::pointer(const DeviceBuffer& buffer,
                                           Device& device) {
  auto state = device_state(device);
  require(buffer.impl_ && buffer.impl_->state == state,
          "buffer belongs to another device");
  return buffer.impl_->pointer;
}
const cudaDeviceProp& detail::BackendAccess::properties(Device& device) {
  return device_state(device)->properties;
}
int detail::BackendAccess::ordinal(Device& device) {
  return device_state(device)->ordinal;
}
void detail::BackendAccess::check(Device& device, cudaError_t error) {
  device_state(device)->check(error);
}
void detail::BackendAccess::check_launch(Device& device) {
  auto state = device_state(device);
  state->check(cudaGetLastError());
}
void detail::BackendAccess::drain_noexcept(Device& device) noexcept {
  auto* native = dynamic_cast<CudaDevice*>(&device);
  if (native) {
    native->state_->drain_noexcept();
  }
}

void Queue::upload(DeviceRegion destination, ConstHostRegion source) {
  impl_->state->activate();
  require(destination.owner && source.owner, "upload requires region owners");
  require(destination.owner->impl_ && source.owner->impl_ &&
              destination.owner->impl_->state == impl_->state &&
              source.owner->impl_->state == impl_->state,
          "upload device identity mismatch");
  check_range(destination.owner->impl_->bytes, destination.offset,
              destination.bytes);
  check_range(source.owner->impl_->bytes, source.offset, source.bytes);
  require(destination.bytes == source.bytes, "upload sizes differ");
  if (destination.bytes) {
    impl_->state->check(cudaMemcpyAsync(
        std::span(static_cast<std::byte*>(destination.owner->impl_->pointer),
                  destination.owner->impl_->bytes)
            .subspan(destination.offset)
            .data(),
        std::span(static_cast<const std::byte*>(source.owner->impl_->pointer),
                  source.owner->impl_->bytes)
            .subspan(source.offset)
            .data(),
        destination.bytes, cudaMemcpyHostToDevice, impl_->stream));
  }
}
void Queue::download(HostRegion destination, ConstDeviceRegion source) {
  impl_->state->activate();
  require(destination.owner && source.owner, "download requires region owners");
  require(destination.owner->impl_ && source.owner->impl_ &&
              destination.owner->impl_->state == impl_->state &&
              source.owner->impl_->state == impl_->state,
          "download device identity mismatch");
  check_range(destination.owner->impl_->bytes, destination.offset,
              destination.bytes);
  check_range(source.owner->impl_->bytes, source.offset, source.bytes);
  require(destination.bytes == source.bytes, "download sizes differ");
  if (destination.bytes) {
    impl_->state->check(cudaMemcpyAsync(
        std::span(static_cast<std::byte*>(destination.owner->impl_->pointer),
                  destination.owner->impl_->bytes)
            .subspan(destination.offset)
            .data(),
        std::span(static_cast<const std::byte*>(source.owner->impl_->pointer),
                  source.owner->impl_->bytes)
            .subspan(source.offset)
            .data(),
        destination.bytes, cudaMemcpyDeviceToHost, impl_->stream));
  }
}
void Queue::copy(DeviceRegion destination, ConstDeviceRegion source) {
  impl_->state->activate();
  require(destination.owner && source.owner, "copy requires region owners");
  require(destination.owner->impl_ && source.owner->impl_ &&
              destination.owner->impl_->state == impl_->state &&
              source.owner->impl_->state == impl_->state,
          "copy device identity mismatch");
  check_range(destination.owner->impl_->bytes, destination.offset,
              destination.bytes);
  check_range(source.owner->impl_->bytes, source.offset, source.bytes);
  require(destination.bytes == source.bytes, "copy sizes differ");
  if (destination.owner == source.owner && destination.bytes != 0) {
    require(destination.offset >= source.offset + source.bytes ||
                source.offset >= destination.offset + destination.bytes,
            "copy regions overlap");
  }
  if (destination.bytes) {
    impl_->state->check(cudaMemcpyAsync(
        std::span(static_cast<std::byte*>(destination.owner->impl_->pointer),
                  destination.owner->impl_->bytes)
            .subspan(destination.offset)
            .data(),
        std::span(static_cast<const std::byte*>(source.owner->impl_->pointer),
                  source.owner->impl_->bytes)
            .subspan(source.offset)
            .data(),
        destination.bytes, cudaMemcpyDeviceToDevice, impl_->stream));
  }
}
void Queue::reserve_completion_points(std::size_t count) {
  impl_->state->activate();
  if (impl_->issued != impl_->completed) {
    throw std::logic_error("reserve completion points on an idle queue");
  }
  impl_->marks.resize(std::max(count, impl_->marks.size()));
  for (auto& mark : impl_->marks) {
    if (!mark.event) {
      impl_->state->check(
          cudaEventCreateWithFlags(&mark.event, cudaEventDisableTiming));
    }
  }
}
CompletionPoint Queue::mark() {
  impl_->state->activate();
  if (impl_->issued == std::numeric_limits<std::uint64_t>::max()) {
    throw std::length_error("queue completion sequence exhausted");
  }
  auto found = std::find_if(impl_->marks.begin(), impl_->marks.end(),
                            [](const auto& m) { return m.sequence == 0; });
  if (found == impl_->marks.end()) {
    impl_->marks.emplace_back();
    found = std::prev(impl_->marks.end());
  }
  if (!found->event) {
    impl_->state->check(
        cudaEventCreateWithFlags(&found->event, cudaEventDisableTiming));
  }
  impl_->state->check(cudaEventRecord(found->event, impl_->stream));
  found->sequence = ++impl_->issued;
  return {impl_->identity, impl_->issued};
}
void Queue::wait(CompletionPoint point) {
  impl_->state->activate();
  require(point.queue_identity == impl_->identity && point.sequence != 0 &&
              point.sequence <= impl_->issued,
          "invalid queue completion point");
  if (point.sequence <= impl_->completed) {
    return;
  }
  auto found =
      std::find_if(impl_->marks.begin(), impl_->marks.end(),
                   [&](const auto& m) { return m.sequence == point.sequence; });
  require(found != impl_->marks.end(), "unknown queue completion point");
  impl_->state->check(cudaEventSynchronize(found->event), true);
  impl_->completed = point.sequence;
  for (auto& mark : impl_->marks) {
    if (mark.sequence <= impl_->completed) {
      mark.sequence = 0;
    }
  }
}
void Queue::drain() {
  impl_->state->activate();
  impl_->state->check(cudaStreamSynchronize(impl_->stream), true);
  impl_->completed = impl_->issued;
  for (auto& mark : impl_->marks) {
    mark.sequence = 0;
  }
}

std::unique_ptr<Device> cuda::open_device(int ordinal) {
  require(ordinal >= 0, "CUDA ordinal must be nonnegative");
  auto state = std::make_shared<State>();
  state->ordinal = ordinal;
  int count = 0;
  state->check(cudaGetDeviceCount(&count));
  if (count == 0) {
    throw std::runtime_error("no CUDA device is available");
  }
  require(ordinal < count, "CUDA ordinal is not available");
  state->activate();
  state->check(cudaGetDeviceProperties(&state->properties, ordinal));
  require(state->properties.major > minimum_compute_major ||
              (state->properties.major == minimum_compute_major &&
               state->properties.minor >= minimum_compute_minor),
          "CUDA build requires compute capability 8.6 or later");
  int driver = 0;
  state->check(cudaDriverGetVersion(&driver));
  state->info = {
      "cuda", std::span(state->properties.name).data(),
      "sm_" + std::to_string(state->properties.major) +
          std::to_string(state->properties.minor),
      "CUDA driver API " + std::to_string(driver / driver_major_scale) + "." +
          std::to_string((driver % driver_major_scale) / driver_minor_scale),
      state->properties.totalGlobalMem};
  return std::make_unique<CudaDevice>(std::move(state));
}
// Admission snapshot: current free memory includes other processes; it is not
// a reservation. Core I.7 (#ri-post), GPU.9
// (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md), queried
// and read for the Metal/shared-retention implementation, govern this boundary.
MemorySnapshot query_memory(Device& device) {
  const auto state = device_state(device);
  std::size_t available = 0, total = 0;
  state->check(cudaMemGetInfo(&available, &total));
  return {MemoryTopology::discrete, total, total, total - available, available};
}
}  // namespace dolittle::gpu
