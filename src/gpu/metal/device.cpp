/*
 * X4 native Metal resources; implements gpu/device.h and memory.h. Allocate
 * aligned arenas once, copy checked equal-size regions through ordered Metal4
 * encoders, signal prefix events, and acquire feedback before host access or
 * release. Devices outlive borrowed handles. Invalid identity/range/alignment
 * throws before submission; native failure poisons the device. Destruction
 * drains outstanding work. Overlay: metal-port.md, Memory and Metal scheduling.
 * Optimization: private tensor arenas and shared staging have separate
 * ownership; transfer packets survive through completion, with no per-token
 * transfers. Setup cost O(bytes) for copies, O(queued transfers) retained
 * command storage. Both implementation corpora queried/read: Core R.1
 * (#rr-raii), I.7 (#ri-post), ES.103 (#res-overflow); performance
 * GPU.7/GPU.8/GPU.9, sources guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md,
 * guidelines/gpu/GPU.8-make-barriers-describe-real-hazards.md,
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "gpu/metal/device.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#include "diagnostics/allocation.h"
#include "gpu/memory.h"
#include "gpu/metal/backend.h"
#include "gpu/metal/buffer_views.h"
#include "gpu/metal/commands.h"
#include "gpu/metal/support.h"

namespace dolittle::gpu {
namespace {
using metal::owned;
constexpr std::size_t max_dispatch_bindings = 31;
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
struct NativeQueue {
  gpu::metal::NativePtr<MTL4::CommandQueue> queue;
  gpu::metal::NativePtr<MTL::SharedEvent> drained;
  std::uint64_t sequence{};
  void drain() {
    if (sequence == std::numeric_limits<std::uint64_t>::max()) {
      throw std::length_error("Metal completion sequence exhausted");
    }
    queue->signalEvent(drained.get(), ++sequence);
    if (!drained->waitUntilSignaledValue(sequence,
                                         metal::completion_timeout_ms)) {
      throw std::runtime_error("Metal queue drain timed out");
    }
  }
};
struct State {
  gpu::metal::NativePtr<MTL::Device> native;
  DeviceInfo info;
  bool poisoned{};
  std::map<std::tuple<std::string, std::string, bool>,
           metal::NativePtr<MTL::ComputePipelineState>>
      pipelines;
  metal::NativePtr<MTL::ComputePipelineState> copy_pipeline;
  std::vector<std::weak_ptr<NativeQueue>> queues;
  void check() const {
    if (poisoned) throw std::runtime_error("Metal device is poisoned");
  }
  void drain_noexcept() noexcept {
    for (auto& weak : queues) {
      if (auto queue = weak.lock()) {
        try {
          queue->drain();
        } catch (...) {
          poisoned = true;
        }
      }
    }
  }
};
class MetalDevice final : public Device {
 public:
  explicit MetalDevice(std::shared_ptr<State> state)
      : state_(std::move(state)) {}
  const DeviceInfo& info() const noexcept override { return state_->info; }
  DeviceBuffer allocate(AllocationRequest request) override {
    return detail::BackendAccess::allocate(*this, request);
  }
  HostBuffer allocate_staging(AllocationRequest request) override {
    return detail::BackendAccess::allocate_staging(*this, request);
  }
  Queue create_queue() override {
    return detail::BackendAccess::create_queue(*this);
  }
  std::shared_ptr<State> state_;
};
std::shared_ptr<State> state_of(Device& device) {
  auto* native = dynamic_cast<MetalDevice*>(&device);
  require(native, "not a Metal device");
  native->state_->check();
  return native->state_;
}
std::size_t allocation_bytes(AllocationRequest request) {
  require(request.alignment && !(request.alignment & (request.alignment - 1)),
          "alignment must be a power of two");
  if (request.bytes >
      std::numeric_limits<std::size_t>::max() - (request.alignment - 1)) {
    throw std::length_error("allocation overflows");
  }
  return request.bytes ? request.bytes + request.alignment - 1 : 0;
}
struct Storage {
  std::shared_ptr<State> state;
  struct View {
    metal::BufferExtent extent;
    metal::NativePtr<MTL::Buffer> native;
  };
  std::shared_ptr<void> backing;
  std::size_t backing_bytes{}, view_limit{}, page_bytes{};
  mutable std::vector<View> views;
  gpu::metal::NativePtr<MTL::Buffer> buffer;
  std::size_t offset{}, bytes{};
  Storage() = default;
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;
  ~Storage() {
    if (state && buffer) state->drain_noexcept();
  }
  void allocate(std::shared_ptr<State> owner, AllocationRequest request,
                bool shared) {
    state = std::move(owner);
    const auto total = allocation_bytes(request);
    bytes = request.bytes;
    if (!total) return;
    if (total > state->native->maxBufferLength()) {
      if (shared) {
        throw std::length_error("staging exceeds Metal maxBufferLength");
      }
      allocate_views(state, request, state->native->maxBufferLength());
      return;
    }
    diagnostics::record_allocation();
    buffer = owned(state->native->newBuffer(
        total, shared ? MTL::ResourceStorageModeShared
                      : MTL::ResourceStorageModePrivate));
    // Host staging also promises C++ pointer alignment. Metal may use a
    // different GPU address; both addresses must satisfy the requested layout.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto address =
        shared ? reinterpret_cast<std::uintptr_t>(buffer->contents())
               : buffer->gpuAddress();
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
    offset =
        (request.alignment - address % request.alignment) % request.alignment;
    if ((buffer->gpuAddress() + offset) % request.alignment) {
      throw std::runtime_error("Metal cannot satisfy shared CPU/GPU alignment");
    }
  }
  void allocate_views(std::shared_ptr<State> owner, AllocationRequest request,
                      std::size_t maximum) {
    state = std::move(owner);
    (void)allocation_bytes(request);
    const auto native_page = sysconf(_SC_PAGESIZE);
    if (native_page <= 0) {
      throw std::runtime_error("cannot query native page size");
    }
    page_bytes = static_cast<std::size_t>(native_page);
    require(request.alignment <= page_bytes,
            "viewed allocation alignment exceeds page size");
    maximum -= maximum % page_bytes;
    require(maximum && maximum <= state->native->maxBufferLength(),
            "invalid native view limit");
    bytes = request.bytes;
    view_limit = maximum;
    if (bytes > std::numeric_limits<std::size_t>::max() - (page_bytes - 1)) {
      throw std::length_error("view backing overflow");
    }
    backing_bytes = bytes + (page_bytes - bytes % page_bytes) % page_bytes;
    if (!backing_bytes) return;
    void* mapping = mmap(nullptr, backing_bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping == MAP_FAILED) {
      throw std::runtime_error("Metal backing allocation failed");
    }
    diagnostics::record_allocation();
    backing = std::shared_ptr<void>(
        mapping, [count = backing_bytes](void* pointer) noexcept {
          (void)munmap(pointer, count);
        });
    const auto plan =
        metal::plan_buffer_views(backing_bytes, maximum, page_bytes, {});
    for (const auto extent : plan.views) add_view(extent);
  }
  void add_view(metal::BufferExtent extent) const {
    const auto storage =
        std::span(static_cast<std::byte*>(backing.get()), backing_bytes)
            .subspan(extent.offset, extent.bytes);
    auto native = owned(state->native->newBuffer(storage.data(), storage.size(),
                                                 MTL::ResourceStorageModeShared,
                                                 nullptr));
    views.push_back({extent, std::move(native)});
  }
  std::size_t transfer_limit(std::size_t begin) const {
    return backing ? view_limit - begin % page_bytes : bytes - begin;
  }
  metal::BufferBinding binding(std::size_t begin, std::size_t count) const {
    state->check();
    require(begin <= bytes && count <= bytes - begin, "region exceeds buffer");
    if (backing) {
      const auto contains = [&](const View& view) {
        return begin >= view.extent.offset &&
               begin - view.extent.offset <= view.extent.bytes &&
               count <= view.extent.bytes - (begin - view.extent.offset);
      };
      auto found = std::ranges::find_if(views, contains);
      if (found == views.end()) {
        const metal::BufferExtent required{begin, count};
        const auto plan = metal::plan_buffer_views(
            backing_bytes, view_limit, page_bytes, std::span(&required, 1));
        add_view(plan.views.at(plan.binding_views.front()));
        found = std::prev(views.end());
      }
      const auto within = begin - found->extent.offset;
      return {found->native.get(), within, count,
              found->native->gpuAddress() + within};
    }
    return {buffer.get(), offset + begin, count,
            buffer ? buffer->gpuAddress() + offset + begin : 0};
  }
};
struct Transfer {
  gpu::metal::NativePtr<MTL4::CommandAllocator> allocator;
  gpu::metal::NativePtr<MTL4::CommandBuffer> commands;
  gpu::metal::NativePtr<MTL::ResidencySet> residency;
  std::shared_ptr<metal::Feedback> feedback;
  std::shared_ptr<metal::CommandBody> body;
};
std::uint64_t next_identity() {
  static std::atomic<std::uint64_t> counter{1};
  const auto id = counter.fetch_add(1);
  if (!id) throw std::length_error("queue identities exhausted");
  return id;
}
}  // namespace
struct DeviceBuffer::Impl : Storage {};
struct HostBuffer::Impl : Storage {};
struct Queue::Impl {
  std::shared_ptr<State> state;
  std::shared_ptr<NativeQueue> native;
  metal::CommandBody* recording{};
  metal::NativePtr<MTL::ResidencySet> residency;
  gpu::metal::NativePtr<MTL::SharedEvent> completed;
  std::uint64_t identity = next_identity(), issued{}, retired{};
  std::deque<Transfer> transfers;
  std::size_t transfer_first{}, transfer_end{};
  std::vector<std::pair<std::uint64_t, std::size_t>> prefixes;
  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    if (!native) return;
    try {
      drain();
    } catch (...) {
      state->poisoned = true;
    }
  }
  void drain() {
    native->drain();
    for (const auto& packet : transfers) {
      metal::wait_feedback(*packet.feedback);
    }
    transfers.clear();
    residency->removeAllAllocations();
    residency->commit();
    transfer_first = transfer_end;
    prefixes.clear();
    retired = issued;
  }
  void transfer(metal::BufferBinding dst, metal::BufferBinding src) {
    state->check();
    require(dst.bytes == src.bytes, "transfer extents differ");
    if (!dst.bytes) return;
    const auto autorelease = metal::pool();
    Transfer packet{owned(state->native->newCommandAllocator()),
                    owned(state->native->newCommandBuffer()),
                    residency,
                    std::make_shared<metal::Feedback>(),
                    {}};
    packet.residency->addAllocation(src.buffer);
    packet.residency->addAllocation(dst.buffer);
    packet.residency->commit();
    packet.commands->beginCommandBuffer(packet.allocator.get());
    auto* encoder = packet.commands->computeCommandEncoder();
    if (!encoder) {
      throw std::runtime_error("Metal transfer encoder creation failed");
    }
    encoder->barrierAfterQueueStages(MTL::StageAll, MTL::StageBlit,
                                     MTL4::VisibilityOptionDevice);
    encoder->copyFromBuffer(src.buffer, src.offset, dst.buffer, dst.offset,
                            src.bytes);
    encoder->endEncoding();
    packet.commands->endCommandBuffer();
    const auto options = metal::commit_options(packet.feedback);
    // Store before commit: vector allocation failure cannot destroy live
    // commands.
    transfers.push_back(std::move(packet));
    ++transfer_end;
    auto& stored = transfers.back();

    const std::array<const MTL4::CommandBuffer*, 1> buffers{
        stored.commands.get()};
    native->queue->commit(buffers.data(), buffers.size(), options.get());
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
  if (!impl_->buffer) return {};
  // Metal exposes a mapped byte allocation. The checked offset stays in it.
  return std::span(static_cast<std::byte*>(impl_->buffer->contents()),
                   impl_->buffer->length())
      .subspan(impl_->offset, impl_->bytes);
}
std::span<const std::byte> HostBuffer::bytes() const noexcept {
  if (!impl_->buffer) return {};
  return std::span(static_cast<const std::byte*>(impl_->buffer->contents()),
                   impl_->buffer->length())
      .subspan(impl_->offset, impl_->bytes);
}
Queue::Queue(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Queue::~Queue() = default;
Queue::Queue(Queue&&) noexcept = default;
Queue& Queue::operator=(Queue&&) noexcept = default;
DeviceBuffer detail::BackendAccess::allocate(Device& device,
                                             AllocationRequest request) {
  auto impl = std::make_unique<DeviceBuffer::Impl>();
  impl->allocate(state_of(device), request, false);
  return DeviceBuffer(std::move(impl));
}
DeviceBuffer detail::BackendAccess::allocate_viewed(Device& device,
                                                    AllocationRequest request,
                                                    std::size_t maximum) {
  auto impl = std::make_unique<DeviceBuffer::Impl>();
  impl->allocate_views(state_of(device), request, maximum);
  return DeviceBuffer(std::move(impl));
}
HostBuffer detail::BackendAccess::allocate_staging(Device& device,
                                                   AllocationRequest request) {
  auto impl = std::make_unique<HostBuffer::Impl>();
  impl->allocate(state_of(device), request, true);
  return HostBuffer(std::move(impl));
}
Queue detail::BackendAccess::create_queue(Device& device) {
  const auto autorelease = metal::pool();
  auto impl = std::make_unique<Queue::Impl>();
  impl->state = state_of(device);
  auto& native = *impl->state->native.get();
  impl->native = std::make_shared<NativeQueue>();
  impl->native->queue = owned(native.newMTL4CommandQueue());
  impl->native->drained = owned(native.newSharedEvent());
  impl->completed = owned(native.newSharedEvent());
  impl->residency = metal::residency(native);
  impl->native->queue->addResidencySet(impl->residency.get());
  std::erase_if(impl->state->queues,
                [](const auto& weak) { return weak.expired(); });
  impl->state->queues.push_back(impl->native);
  return Queue(std::move(impl));
}
MTL::Device& detail::BackendAccess::native_device(Device& device) {
  return *state_of(device)->native.get();
}
MTL4::CommandQueue& detail::BackendAccess::native_queue(Queue& queue,
                                                        Device& device) {
  require(queue.impl_ && queue.impl_->state == state_of(device),
          "foreign queue");
  return *queue.impl_->native->queue.get();
}
metal::BufferBinding detail::BackendAccess::binding(ConstDeviceRegion region,
                                                    Device& device) {
  require(region.owner && region.owner->impl_ &&
              region.owner->impl_->state == state_of(device),
          "foreign device region");
  return region.owner->impl_->binding(region.offset, region.bytes);
}
metal::BufferBinding detail::BackendAccess::binding(DeviceRegion region,
                                                    Device& device) {
  return binding(ConstDeviceRegion{region.owner, region.offset, region.bytes},
                 device);
}
metal::BufferBinding detail::BackendAccess::binding(HostRegion region,
                                                    Device& device) {
  require(region.owner && region.owner->impl_ &&
              region.owner->impl_->state == state_of(device),
          "foreign host region");
  return region.owner->impl_->binding(region.offset, region.bytes);
}
metal::BufferBinding detail::BackendAccess::binding(ConstHostRegion region,
                                                    Device& device) {
  require(region.owner && region.owner->impl_ &&
              region.owner->impl_->state == state_of(device),
          "foreign host region");
  return region.owner->impl_->binding(region.offset, region.bytes);
}
void detail::BackendAccess::record(Queue& queue, Device& device,
                                   metal::CommandBody* body) {
  (void)native_queue(queue, device);
  require(!body || !queue.impl_->recording, "nested command recording");
  queue.impl_->recording = body;
}
std::shared_ptr<MTL::ComputePipelineState> detail::BackendAccess::pipeline(
    Device& device, const char* path, const char* name, bool indirect) {
  const auto state = state_of(device);
  const auto key =
      std::make_tuple(std::string(path), std::string(name), indirect);
  if (auto found = state->pipelines.find(key);
      found != state->pipelines.end()) {
    return found->second;
  }
  const auto autorelease = metal::pool();
  NS::Error* error = nullptr;
  auto library = metal::transfer(state->native->newLibrary(
      NS::String::string(path, NS::UTF8StringEncoding), &error));
  if (!library) {
    throw std::runtime_error("Metal library: " + metal::describe(error));
  }
  auto function = owned(
      library->newFunction(NS::String::string(name, NS::UTF8StringEncoding)));
  auto descriptor = owned(MTL::ComputePipelineDescriptor::alloc()->init());
  descriptor->setComputeFunction(function.get());
  descriptor->setSupportIndirectCommandBuffers(indirect);
  descriptor->setLabel(NS::String::string(name, NS::UTF8StringEncoding));
  // Apple's validator cannot instrument ICBs with their own pipeline/buffers.
  // Direct control shaders remain instrumented; API validation covers both.
  if (indirect) descriptor->setShaderValidation(MTL::ShaderValidationDisabled);
  auto result = metal::transfer(state->native->newComputePipelineState(
      descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  if (!result) {
    throw std::runtime_error("Metal pipeline: " + metal::describe(error));
  }
  state->pipelines.emplace(key, result);
  return result;
}
void detail::BackendAccess::dispatch(
    Queue& queue, Device& device, MTL::ComputePipelineState& pipeline,
    std::span<const metal::BufferBinding> bindings, MTL::Size grid,
    MTL::Size group, std::span<const std::byte> parameters,
    std::size_t threadgroup_bytes) {
  (void)native_queue(queue, device);
  require(
      threadgroup_bytes <= native_device(device).maxThreadgroupMemoryLength(),
      "dispatch exceeds threadgroup memory");
  require(bindings.size() < max_dispatch_bindings,
          "too many dispatch bindings");
  const auto direct = queue.impl_->recording == nullptr;
  auto owned_body = direct ? std::make_shared<metal::CommandBody>() : nullptr;
  auto& body = direct ? *owned_body : *queue.impl_->recording;
  std::array<metal::BufferBinding, max_dispatch_bindings> inputs{};
  std::size_t count = 0;
  if (!parameters.empty()) {
    inputs.at(count++) = body.parameter(device, parameters);
  }
  for (auto binding : bindings) inputs.at(count++) = binding;
  body.append(pipeline, std::span(inputs).first(count), grid, group,
              threadgroup_bytes);
  if (!direct) return;
  body.seal(device);
  const auto autorelease = metal::pool();
  auto& q = *queue.impl_;
  Transfer packet{owned(q.state->native->newCommandAllocator()),
                  owned(q.state->native->newCommandBuffer()), q.residency,
                  std::make_shared<metal::Feedback>(), std::move(owned_body)};
  body.residency(*packet.residency);
  packet.residency->commit();
  packet.commands->beginCommandBuffer(packet.allocator.get());
  auto* encoder = packet.commands->computeCommandEncoder();
  if (!encoder) {
    throw std::runtime_error("Metal compute encoder creation failed");
  }
  encoder->barrierAfterQueueStages(MTL::StageAll, MTL::StageDispatch,
                                   MTL4::VisibilityOptionDevice);
  encoder->executeCommandsInBuffer(body.native(), NS::Range{0, body.count()});
  encoder->endEncoding();
  packet.commands->endCommandBuffer();
  const auto options = metal::commit_options(packet.feedback);
  q.transfers.push_back(std::move(packet));
  ++q.transfer_end;
  auto& stored = q.transfers.back();

  const std::array<const MTL4::CommandBuffer*, 1> commands{
      stored.commands.get()};
  q.native->queue->commit(commands.data(), commands.size(), options.get());
}
void detail::BackendAccess::poison(Device& device) noexcept {
  if (auto* native = dynamic_cast<MetalDevice*>(&device)) {
    native->state_->poisoned = true;
  }
}
void detail::BackendAccess::drain_noexcept(Device& device) noexcept {
  if (auto* native = dynamic_cast<MetalDevice*>(&device)) {
    native->state_->drain_noexcept();
  }
}
void Queue::upload(DeviceRegion dst, ConstHostRegion src) {
  require(dst.owner && src.owner && dst.owner->impl_ && src.owner->impl_ &&
              dst.owner->impl_->state == impl_->state &&
              src.owner->impl_->state == impl_->state,
          "foreign upload region");
  if (src.offset > std::numeric_limits<std::size_t>::max() - src.bytes ||
      dst.offset > std::numeric_limits<std::size_t>::max() - dst.bytes) {
    throw std::length_error("transfer extent overflows");
  }
  const auto& source = *src.owner->impl_;
  const auto& destination = *dst.owner->impl_;
  require(src.bytes == dst.bytes && src.offset <= source.bytes &&
              src.bytes <= source.bytes - src.offset &&
              dst.offset <= destination.bytes &&
              dst.bytes <= destination.bytes - dst.offset,
          "transfer exceeds allocation");
  for (std::size_t done = 0; done < src.bytes;) {
    const auto count =
        std::min({src.bytes - done, source.transfer_limit(src.offset + done),
                  destination.transfer_limit(dst.offset + done)});
    impl_->transfer(destination.binding(dst.offset + done, count),
                    source.binding(src.offset + done, count));
    done += count;
  }
}
void Queue::download(HostRegion dst, ConstDeviceRegion src) {
  require(dst.owner && src.owner && dst.owner->impl_ && src.owner->impl_ &&
              dst.owner->impl_->state == impl_->state &&
              src.owner->impl_->state == impl_->state,
          "foreign download region");
  if (src.offset > std::numeric_limits<std::size_t>::max() - src.bytes ||
      dst.offset > std::numeric_limits<std::size_t>::max() - dst.bytes) {
    throw std::length_error("transfer extent overflows");
  }
  const auto& source = *src.owner->impl_;
  const auto& destination = *dst.owner->impl_;
  require(src.bytes == dst.bytes && src.offset <= source.bytes &&
              src.bytes <= source.bytes - src.offset &&
              dst.offset <= destination.bytes &&
              dst.bytes <= destination.bytes - dst.offset,
          "transfer exceeds allocation");
  for (std::size_t done = 0; done < src.bytes;) {
    const auto count =
        std::min({src.bytes - done, source.transfer_limit(src.offset + done),
                  destination.transfer_limit(dst.offset + done)});
    impl_->transfer(destination.binding(dst.offset + done, count),
                    source.binding(src.offset + done, count));
    done += count;
  }
}
void Queue::copy(DeviceRegion dst, ConstDeviceRegion src) {
  require(dst.owner && src.owner && dst.owner->impl_ && src.owner->impl_ &&
              dst.owner->impl_->state == impl_->state &&
              src.owner->impl_->state == impl_->state,
          "foreign copy region");
  if (src.offset > std::numeric_limits<std::size_t>::max() - src.bytes ||
      dst.offset > std::numeric_limits<std::size_t>::max() - dst.bytes) {
    throw std::length_error("transfer extent overflows");
  }
  const auto& source = *src.owner->impl_;
  const auto& destination = *dst.owner->impl_;
  require(src.offset <= source.bytes &&
              src.bytes <= source.bytes - src.offset &&
              dst.offset <= destination.bytes &&
              dst.bytes <= destination.bytes - dst.offset,
          "copy exceeds allocation");
  require(dst.bytes == src.bytes, "copy extents differ");
  require(dst.owner != src.owner || dst.offset >= src.offset + src.bytes ||
              src.offset >= dst.offset + dst.bytes,
          "overlapping copy");
  if (impl_->recording) {
    require(src.bytes <= std::numeric_limits<std::uint32_t>::max(),
            "prepared copy exceeds dispatch size");
    if (!impl_->state->copy_pipeline) {
      MetalDevice borrowed(impl_->state);
      impl_->state->copy_pipeline =
          metal::pipeline(borrowed, DOLITTLE_METAL_LIBRARY, "sequence_copy");
    }
    const std::array bindings{source.binding(src.offset, src.bytes),
                              destination.binding(dst.offset, dst.bytes)};
    constexpr std::size_t copy_threads = 256;
    impl_->recording->append(*impl_->state->copy_pipeline.get(), bindings,
                             {src.bytes, 1, 1},
                             {std::min(src.bytes, copy_threads), 1, 1});
    return;
  }
  for (std::size_t done = 0; done < src.bytes;) {
    const auto count =
        std::min({src.bytes - done, source.transfer_limit(src.offset + done),
                  destination.transfer_limit(dst.offset + done)});
    impl_->transfer(destination.binding(dst.offset + done, count),
                    source.binding(src.offset + done, count));
    done += count;
  }
}
void Queue::reserve_completion_points(std::size_t count) {
  require(impl_->transfers.empty(), "reserve requires an idle queue");
  impl_->prefixes.reserve(count);
}
CompletionPoint Queue::mark() {
  impl_->state->check();
  if (impl_->issued == std::numeric_limits<std::uint64_t>::max()) {
    throw std::length_error("completion exhausted");
  }
  const auto sequence = impl_->issued + 1;
  impl_->prefixes.emplace_back(sequence, impl_->transfer_end);
  impl_->issued = sequence;
  impl_->native->queue->signalEvent(impl_->completed.get(), sequence);
  impl_->native->queue->wait(impl_->completed.get(), sequence);
  return {impl_->identity, sequence};
}
void Queue::wait(CompletionPoint point) {
  impl_->state->check();
  require(point.queue_identity == impl_->identity && point.sequence &&
              point.sequence <= impl_->issued,
          "invalid completion point");
  if (point.sequence <= impl_->retired) return;
  try {
    if (!impl_->completed->waitUntilSignaledValue(
            point.sequence, metal::completion_timeout_ms)) {
      throw std::runtime_error("Metal prefix timed out");
    }
    const auto found =
        std::ranges::find(impl_->prefixes, point.sequence,
                          &std::pair<std::uint64_t, std::size_t>::first);
    require(found != impl_->prefixes.end(), "unknown completion point");
    const auto prefix_end = found->second;
    while (impl_->transfer_first < prefix_end) {
      auto& packet = impl_->transfers.front();
      metal::wait_feedback(*packet.feedback);

      impl_->transfers.pop_front();
      ++impl_->transfer_first;
    }
    if (impl_->transfers.empty()) {
      impl_->residency->removeAllAllocations();
      impl_->residency->commit();
    }
    std::erase_if(impl_->prefixes, [&](const auto& prefix) {
      return prefix.first <= point.sequence;
    });
    impl_->retired = point.sequence;

  } catch (...) {
    impl_->state->poisoned = true;
    throw;
  }
}
void Queue::drain() {
  impl_->state->check();
  try {
    impl_->drain();
  } catch (...) {
    impl_->state->poisoned = true;
    throw;
  }
}
MemorySnapshot query_memory(Device& device) {
  const auto state = state_of(device);
  const auto limit = state->native->recommendedMaxWorkingSetSize();
  const auto allocated = state->native->currentAllocatedSize();
  return {MemoryTopology::unified, state->native->maxBufferLength(), limit,
          allocated, allocated < limit ? limit - allocated : 0};
}
namespace metal {
std::unique_ptr<Device> open_device() {
  const auto autorelease = pool();
  auto state = std::make_shared<State>();
  state->native = owned(MTL::CreateSystemDefaultDevice());
  require(state->native->hasUnifiedMemory() &&
              state->native->supportsFamily(MTL::GPUFamilyMetal4),
          "Metal4 unified-memory device required");
  state->info = {"metal", state->native->name()->utf8String(),
                 state->native->architecture()->name()->utf8String(), "Metal 4",
                 state->native->recommendedMaxWorkingSetSize()};
  return std::make_unique<MetalDevice>(std::move(state));
}
}  // namespace metal
}  // namespace dolittle::gpu
