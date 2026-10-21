/*
 * X4 command-body implementation; contract and algorithm: commands.h. Resolve
 * pipelines at preparation, retain and register every indirect buffer/pipeline
 * allocation for residency, encode static grids
 * and dependencies once. No sampled value reaches host dispatch. O(recipe)
 * storage. Implementation corpora: Core R.1 (#rr-raii), I.7 (#ri-post);
 * performance GPU.6/GPU.8, guidelines/gpu/GPU.6-batch-tiny-gpu-work.md and
 * guidelines/gpu/GPU.8-make-barriers-describe-real-hazards.md.
 */
#include "gpu/metal/commands.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "gpu/metal/support.h"
namespace dolittle::gpu::metal {
namespace {
constexpr std::size_t max_buffer_bindings = 31;
}
void CommandBody::append(MTL::ComputePipelineState& state,
                         std::span<const BufferBinding> bindings,
                         MTL::Size grid, MTL::Size group,
                         std::size_t threadgroup_bytes) {
  if (commands_ || !grid.width || !grid.height || !grid.depth || !group.width ||
      !group.height || !group.depth || bindings.size() > max_buffer_bindings) {
    throw std::invalid_argument("invalid Metal recorded dispatch");
  }
  for (const auto binding : bindings) {
    if (!binding.buffer || !binding.bytes) {
      throw std::invalid_argument("empty dispatch operand");
    }
    buffers_.push_back(gpu::metal::retain(binding.buffer));
  }
  dispatches_.push_back({gpu::metal::retain(&state),
                         {bindings.begin(), bindings.end()},
                         grid,
                         group,
                         threadgroup_bytes});
}
void CommandBody::seal(Device& device) {
  if (commands_ || dispatches_.empty()) {
    throw std::invalid_argument("empty/already sealed Metal body");
  }
  const auto autorelease = pool();
  auto descriptor =
      owned(MTL::IndirectCommandBufferDescriptor::alloc()->init());
  descriptor->setCommandTypes(
      MTL::IndirectCommandTypeConcurrentDispatchThreads);
  descriptor->setInheritBuffers(false);
  descriptor->setInheritPipelineState(false);
  descriptor->setMaxKernelBufferBindCount(max_buffer_bindings);
  commands_ = owned(
      detail::BackendAccess::native_device(device).newIndirectCommandBuffer(
          descriptor.get(), dispatches_.size(),
          MTL::ResourceStorageModeShared));
  for (std::size_t i = 0; i < dispatches_.size(); ++i) {
    const auto& dispatch = dispatches_.at(i);
    auto* command = commands_->indirectComputeCommand(i);
    command->setComputePipelineState(dispatch.pipeline.get());
    for (std::size_t j = 0; j < dispatch.bindings.size(); ++j) {
      const auto binding = dispatch.bindings.at(j);
      command->setKernelBuffer(binding.buffer, binding.offset, j);
    }
    if (dispatch.threadgroup_bytes) {
      command->setThreadgroupMemoryLength(dispatch.threadgroup_bytes, 0);
    }
    command->concurrentDispatchThreads(dispatch.grid, dispatch.group);
    command->setBarrier();
  }
}
void CommandBody::retain(Submission& submission) const {
  if (!commands_) throw std::invalid_argument("unsealed Metal body");
  submission.retain_native(*commands_.get());
  for (const auto& dispatch : dispatches_) {
    submission.retain_native(*dispatch.pipeline);
  }
  for (const auto& buffer : buffers_) submission.retain_native(*buffer.get());
}
std::size_t CommandBody::command_bytes() const {
  return commands_ ? commands_->allocatedSize() : 0;
}
std::size_t CommandBody::host_bytes() const {
  std::size_t result = dispatches_.capacity() * sizeof(Dispatch) +
                       (buffers_.capacity() + parameters_.capacity()) *
                           sizeof(gpu::metal::NativePtr<MTL::Buffer>);
  for (const auto& dispatch : dispatches_) {
    result += dispatch.bindings.capacity() * sizeof(BufferBinding);
  }
  return result;
}
BufferBinding CommandBody::parameter(Device& device,
                                     std::span<const std::byte> bytes) {
  constexpr std::size_t page_bytes = 65536;
  constexpr std::size_t alignment = 16;
  if (commands_ || bytes.empty() || bytes.size() > page_bytes) {
    throw std::invalid_argument("invalid Metal parameter snapshot");
  }
  auto offset = (parameter_offset_ + alignment - 1) & ~(alignment - 1);
  if (parameters_.empty() || bytes.size() > page_bytes - offset) {
    parameters_.push_back(
        owned(detail::BackendAccess::native_device(device).newBuffer(
            page_bytes, MTL::ResourceStorageModeShared)));
    offset = 0;
  }
  auto* buffer = parameters_.back().get();
  auto destination =
      std::span(static_cast<std::byte*>(buffer->contents()), page_bytes)
          .subspan(offset, bytes.size());
  std::ranges::copy(bytes, destination.begin());
  parameter_offset_ = offset + bytes.size();
  return {buffer, offset, bytes.size(), buffer->gpuAddress() + offset};
}
void CommandBody::residency(MTL::ResidencySet& set) const {
  if (!commands_) throw std::invalid_argument("unsealed Metal body");
  set.addAllocation(commands_.get());
  for (const auto& dispatch : dispatches_) {
    set.addAllocation(dispatch.pipeline.get());
  }
  for (const auto& buffer : buffers_) set.addAllocation(buffer.get());
}
std::size_t CommandBody::parameter_bytes() const {
  std::size_t bytes = 0;
  for (const auto& buffer : parameters_) bytes += buffer->allocatedSize();
  return bytes;
}
gpu::metal::NativePtr<MTL::ComputePipelineState> pipeline(Device& device,
                                                          const char* path,
                                                          const char* name,
                                                          PipelineUse use) {
  return detail::BackendAccess::pipeline(device, path, name,
                                         use == PipelineUse::indirect);
}
}  // namespace dolittle::gpu::metal
