/*
 * X4 reusable compute command bodies. Contract: append only during preparation,
 * seal once, then execute without rebinding. Each dispatch retains its pipeline
 * and buffer references; callers keep logical owners stable through drain.
 * Algorithm (metal-port.md, Metal scheduling): record fixed grid/bindings,
 * populate compute ICB commands with a barrier between dependent dispatches,
 * register every indirect buffer and pipeline allocation with the timeline.
 * Metal 4 requires pipeline residency even when the CPU retains the object.
 * O(recipe) metadata and command storage, independent of generated length. Core
 * R.1/I.7
 * (#rr-raii/#ri-post); performance GPU.6/GPU.8/GPU.9, retrieved for
 * implementation: guidelines/gpu/GPU.6-batch-tiny-gpu-work.md,
 * guidelines/gpu/GPU.8-make-barriers-describe-real-hazards.md,
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#pragma once
#include <Metal/Metal.hpp>
#include <vector>

#include "gpu/metal/backend.h"
#include "gpu/metal/submission.h"
#include "gpu/metal/support.h"
namespace dolittle::gpu::metal {
struct Dispatch {
  gpu::metal::NativePtr<MTL::ComputePipelineState> pipeline;
  std::vector<BufferBinding> bindings;
  MTL::Size grid, group;
  std::size_t threadgroup_bytes{};
};
class CommandBody {
 public:
  void append(MTL::ComputePipelineState&, std::span<const BufferBinding>,
              MTL::Size grid, MTL::Size group,
              std::size_t threadgroup_bytes = 0);
  [[nodiscard]] BufferBinding parameter(Device&, std::span<const std::byte>);
  void residency(MTL::ResidencySet&) const;
  [[nodiscard]] std::size_t parameter_bytes() const;
  void seal(Device&);
  void retain(Submission&) const;
  [[nodiscard]] MTL::IndirectCommandBuffer* native() const {
    return commands_.get();
  }
  [[nodiscard]] std::size_t count() const { return dispatches_.size(); }
  [[nodiscard]] std::size_t command_bytes() const;
  [[nodiscard]] std::size_t host_bytes() const;

 private:
  std::vector<Dispatch> dispatches_;
  std::vector<NativePtr<MTL::Buffer>> parameters_;
  std::size_t parameter_offset_{};
  std::vector<gpu::metal::NativePtr<MTL::Buffer>> buffers_;
  gpu::metal::NativePtr<MTL::IndirectCommandBuffer> commands_;
};
enum class PipelineUse : std::uint8_t { direct, indirect };
[[nodiscard]] gpu::metal::NativePtr<MTL::ComputePipelineState> pipeline(
    Device&, const char* library, const char* name,
    PipelineUse use = PipelineUse::indirect);
}  // namespace dolittle::gpu::metal
