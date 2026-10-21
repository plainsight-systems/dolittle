/*
 * X4 command-body implementation; contract and algorithm: commands.h. Resolve
 * pipelines at preparation, retain all indirect operands, encode static grids
 * and dependencies once. No sampled value reaches host dispatch. O(recipe)
 * storage. Implementation corpora: Core R.1 (#rr-raii), I.7 (#ri-post);
 * performance GPU.6/GPU.8, guidelines/gpu/GPU.6-batch-tiny-gpu-work.md and
 * guidelines/gpu/GPU.8-make-barriers-describe-real-hazards.md.
 */
#include "gpu/metal/commands.h"

#include <algorithm>
#include <limits>

#include "gpu/metal/support.h"
namespace dolittle::gpu::metal {
namespace {
constexpr std::size_t max_buffer_bindings = 31;
}
void CommandBody::append(MTL::ComputePipelineState& state,
                         std::span<const BufferBinding> bindings,
                         MTL::Size grid, MTL::Size group) {
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
                         group});
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
    command->concurrentDispatchThreads(dispatch.grid, dispatch.group);
    command->setBarrier();
  }
}
void CommandBody::retain(Submission& submission) const {
  if (!commands_) throw std::invalid_argument("unsealed Metal body");
  submission.retain_native(*commands_.get());
  for (const auto& buffer : buffers_) submission.retain_native(*buffer.get());
}
std::size_t CommandBody::command_bytes() const {
  return commands_ ? commands_->allocatedSize() : 0;
}
std::size_t CommandBody::host_bytes() const {
  std::size_t result =
      dispatches_.capacity() * sizeof(Dispatch) +
      buffers_.capacity() * sizeof(gpu::metal::NativePtr<MTL::Buffer>);
  for (const auto& dispatch : dispatches_) {
    result += dispatch.bindings.capacity() * sizeof(BufferBinding);
  }
  return result;
}
gpu::metal::NativePtr<MTL::ComputePipelineState> pipeline(Device& device,
                                                          const char* path,
                                                          const char* name,
                                                          PipelineUse use) {
  const auto autorelease = pool();
  NS::Error* error = nullptr;
  auto library = gpu::metal::transfer(
      detail::BackendAccess::native_device(device).newLibrary(
          NS::String::string(path, NS::UTF8StringEncoding), &error));
  if (!library) throw std::runtime_error("Metal library: " + describe(error));
  auto function = owned(
      library->newFunction(NS::String::string(name, NS::UTF8StringEncoding)));
  auto descriptor = owned(MTL::ComputePipelineDescriptor::alloc()->init());
  descriptor->setComputeFunction(function.get());
  descriptor->setSupportIndirectCommandBuffers(use == PipelineUse::indirect);
  descriptor->setLabel(NS::String::string(name, NS::UTF8StringEncoding));
  // Apple's shader validator requires inherited pipeline/buffer bindings for
  // ICBs. These reusable bodies intentionally own their bindings, so that
  // instrumentation is unsupported. Keep API validation and direct control
  // shader validation enabled. Numerical/device tests exercise body results.
  // https://developer.apple.com/documentation/xcode/validating-your-apps-metal-shader-usage
  if (use == PipelineUse::indirect) {
    descriptor->setShaderValidation(MTL::ShaderValidationDisabled);
  }
  auto result = gpu::metal::transfer(
      detail::BackendAccess::native_device(device).newComputePipelineState(
          descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  if (!result) throw std::runtime_error("Metal pipeline: " + describe(error));
  return result;
}
}  // namespace dolittle::gpu::metal
