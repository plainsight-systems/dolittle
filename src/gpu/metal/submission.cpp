/*
 * X4 prepared Metal timeline; implements submission.h. Every window owns its
 * allocator and command buffer until drain. Seal in order, commit every window
 * at start, signal an event after each and order subsequent work after it.
 * Collection waits only for an immutable prefix; it submits nothing. All
 * resource and feedback owners are allocated before start. Actual allocator
 * bytes are checked against admission before any work is submitted.
 * Overlay: metal-port.md, Metal scheduling. O(W) setup and native storage;
 * optimization: reuse encoded bodies, not CPU-dependent iteration. Errors
 * poison the device; cleanup drains before freeing native references.
 * Implementation corpora queried/read: Core R.1 (#rr-raii), I.7 (#ri-post),
 * ES.103 (#res-overflow); performance GPU.6/GPU.7/GPU.8, sources
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md,
 * guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md,
 * guidelines/gpu/GPU.8-make-barriers-describe-real-hazards.md.
 */
#include "gpu/metal/submission.h"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "gpu/metal/backend.h"
#include "gpu/metal/support.h"
namespace dolittle::gpu::metal {
using Access = detail::BackendAccess;
namespace {
void require(bool condition) {
  if (!condition) {
    throw std::invalid_argument("invalid Metal timeline lifecycle/index");
  }
}
struct Window {
  gpu::metal::NativePtr<MTL4::CommandAllocator> allocator;
  gpu::metal::NativePtr<MTL4::CommandBuffer> commands;
  std::shared_ptr<Feedback> feedback;
  gpu::metal::NativePtr<MTL4::CommitOptions> options;
  std::size_t charged_bytes{};
};
}  // namespace
struct Submission::Impl {
  std::reference_wrapper<Device> device;
  std::reference_wrapper<MTL4::CommandQueue> queue;
  TimelineLimits limits;
  gpu::metal::NativePtr<MTL::ResidencySet> residency;
  gpu::metal::NativePtr<MTL::SharedEvent> event;
  std::vector<Window> windows;
  std::vector<NativePtr<MTL::Allocation>> retained;
  std::size_t sealed{}, submitted{}, retired{}, command_storage_bytes{};
  bool open{}, started{}, failed{};
  Impl(Device& d, Queue& q, TimelineLimits l)
      : device(d),
        queue(Access::native_queue(q, d)),
        limits(l),
        residency(metal::residency(Access::native_device(d))),
        event(owned(Access::native_device(d).newSharedEvent())) {
    require(l.windows && l.max_command_bytes);
    q.drain();
    windows.reserve(l.windows);
    for (std::size_t i = 0; i < l.windows; ++i) {
      auto feedback = std::make_shared<Feedback>();
      windows.push_back({owned(Access::native_device(d).newCommandAllocator()),
                         owned(Access::native_device(d).newCommandBuffer()),
                         feedback, commit_options(feedback)});
      charge(windows.back());
    }
  }
  void charge(Window& window) {
    const auto allocated = window.allocator->allocatedSize();
    const auto others = command_storage_bytes - window.charged_bytes;
    if (allocated > limits.max_command_bytes ||
        others > limits.max_command_bytes - allocated) {
      failed = true;
      throw std::length_error("Metal prepared command budget exceeded");
    }
    command_storage_bytes = others + allocated;
    window.charged_bytes = allocated;
  }
  void wait(std::size_t count) {
    require(started && count <= submitted);
    if (count <= retired) return;
    try {
      if (!event->waitUntilSignaledValue(count, completion_timeout_ms)) {
        throw std::runtime_error("Metal timeline completion timed out");
      }
      for (; retired < count; ++retired) {
        wait_feedback(*windows.at(retired).feedback);
      }
    } catch (...) {
      Access::poison(device);
      throw;
    }
  }
};
Submission::Submission(Device& d, Queue& q, TimelineLimits limits) {
  const auto autorelease = pool();
  impl_ = std::make_unique<Impl>(d, q, limits);
}
Submission::~Submission() {
  if (!impl_) return;
  if (impl_->submitted) {
    try {
      impl_->wait(impl_->submitted);
    } catch (...) {
      Access::drain_noexcept(impl_->device);
    }
    impl_->queue.get().removeResidencySet(impl_->residency.get());
  }
}
void Submission::retain(std::span<const ConstDeviceRegion> regions) {
  require(!impl_->started);
  for (auto region : regions) {
    const auto binding = Access::binding(region, impl_->device);
    if (binding.buffer) retain_native(*binding.buffer);
  }
}
void Submission::retain(std::span<const ConstHostRegion> regions) {
  require(!impl_->started);
  for (auto region : regions) {
    const auto binding = Access::binding(region, impl_->device);
    if (binding.buffer) retain_native(*binding.buffer);
  }
}
void Submission::retain_native(MTL::Allocation& allocation) {
  require(!impl_->started);
  impl_->retained.push_back(metal::retain(&allocation));
  impl_->residency->addAllocation(&allocation);
}
MTL4::CommandBuffer& Submission::begin_window(std::size_t index) {
  require(!impl_->failed && !impl_->started && !impl_->open &&
          index == impl_->sealed && index < impl_->windows.size());
  auto& window = impl_->windows.at(index);
  window.commands->beginCommandBuffer(window.allocator.get());
  impl_->open = true;
  return *window.commands.get();
}
void Submission::end_window(std::size_t index) {
  require(!impl_->started && impl_->open && index == impl_->sealed);
  impl_->windows.at(index).commands->endCommandBuffer();
  impl_->open = false;
  ++impl_->sealed;
  impl_->charge(impl_->windows.at(index));
}
std::size_t Submission::command_bytes() const {
  return impl_->command_storage_bytes;
}
std::size_t Submission::host_bytes() const {
  return sizeof(Impl) + impl_->windows.capacity() * sizeof(Window) +
         impl_->retained.capacity() * sizeof(NativePtr<MTL::Allocation>) +
         impl_->windows.size() * sizeof(Feedback);
}
void Submission::start() {
  require(!impl_->failed && !impl_->started && !impl_->open &&
          impl_->sealed == impl_->windows.size());
  if (command_bytes() > impl_->limits.max_command_bytes) {
    throw std::length_error("Metal command budget exceeded");
  }
  impl_->residency->commit();
  impl_->queue.get().addResidencySet(impl_->residency.get());
  impl_->started = true;
  for (const auto& window : impl_->windows) {
    const std::array<const MTL4::CommandBuffer*, 1> commands{
        window.commands.get()};
    impl_->queue.get().commit(commands.data(), commands.size(),
                              window.options.get());
    impl_->queue.get().signalEvent(impl_->event.get(), ++impl_->submitted);
    impl_->queue.get().wait(impl_->event.get(), impl_->submitted);
  }
}
std::size_t Submission::completed_prefix() const {
  require(impl_->started);
  return std::min(static_cast<std::size_t>(impl_->event->signaledValue()),
                  impl_->submitted);
}
void Submission::wait_prefix(std::size_t count) { impl_->wait(count); }
void Submission::drain() {
  if (impl_->started) impl_->wait(impl_->submitted);
}
}  // namespace dolittle::gpu::metal
