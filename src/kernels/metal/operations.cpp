/*
 * Metal implementation of the shared target-generation operations.
 *
 * Axis X5: implement the existing PreparedOperation interface for Metal.
 * Contract: same shapes, alias rules, row policies and errors as operations.h.
 * This owner borrows Device and retains its pipelines, metadata and immutable
 * parameter storage until all encoded readers finish. A body reads the device
 * position supplied by DeviceSequence; direct diagnostics may use a checked
 * fixed position. Submission must not discover layouts, compile a pipeline,
 * allocate tensor storage, wait for results or read device control words.
 *
 * Algorithm: transformer-execution.md, Preparing the first execution recipe.
 * 1. Run shared operation validation. Select BF16/FP32 pipeline variants and
 *    geometry from feature dimensions and native limits; report all workspace.
 * 2. Resolve each tensor to a checked native view that contains the complete
 *    operand. Validate device identity, byte extent, alignment and overlap.
 * 3. Prepare immutable arguments for each recorded prefill/tail/decode binding
 *    variant. Copy constants into a bounded shared arena before start; retain
 *    each snapshot until its body retires. Do not rewrite one uniform buffer
 *    after encoding several bodies that reference it. GPU-derived values are
 *    device bindings, never copied into these constants.
 * 4. Encode dispatches into the active CommandBody using BackendAccess. Record
 *    exact threadgroup memory lengths alongside grid/pipeline/buffer bindings.
 *    For a direct diagnostic submission, retain constants until its completion
 *    fence before recycling them. That path is outside live generation.
 * 5. Queue barriers order writes before consumers. GatherRows uses D2D copies;
 *    K projection/RoPE and V projection may write their own cache append range
 *    directly. Q and ordinary outputs use local row offsets. Every dynamic
 *    store computes S from the immutable-for-this-body device position.
 * 6. Kernel return leaves output device-owned. Existing hook insertion copies
 *    only advertised producer values. Do not fuse across a requested producer
 *    or materialize a full vocabulary row on the host during generation.
 *
 * Workspace: attention scale 4 bytes; RoPE frequency table (d/2)*4 bytes;
 * kernel parameters and pipeline objects prepared separately and reported to
 * admission. These constants are initialized by GPU setup dispatches before
 * graph readers. Projection/RMS/attention intermediates remain registers or
 * bounded threadgroup storage. Sampler owns its separate admitted arena.
 *
 * Optimization: pipeline cache keys include entry point, storage type and
 * arithmetic specialization; cache ownership belongs to Device, not each
 * token/operation. Parameters may be shared only when byte-identical and
 * immutable. X4 CommandBody retains threadgroup-length metadata and replays
 * it via setThreadgroupMemoryLength alongside buffers and grids.
 * No command/resource creation in the running token sequence.
 * Shared PreparedOperation validation and executor hooks remain
 * backend-neutral.
 *
 * Header-design consultation, 2026-10-21: both local corpora searched and
 * cited rules read. Core I.7 (#ri-post) fixes numerical/completion results;
 * SL.con.3 (#rsl-bounds) requires checked extents before native indexing;
 * R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.1: device position and outputs never return to host dispatch.
 * guidelines/gpu/GPU.1-keep-data-on-device.md
 * GPU.6: cache pipelines and replay prepared bodies.
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md
 * GPU.9: admit constants/workspace and retain aliases until last use.
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md
 */

#include "kernels/operations.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "bounds.h"
#include "gpu/metal/backend.h"
#include "gpu/metal/commands.h"
#include "inference/cache/contiguous.h"
#include "kernels/metal/launch.h"
namespace dolittle::kernels {
namespace {
using Bridge = gpu::detail::BackendAccess;
constexpr std::size_t simd_lanes = 32;
constexpr std::size_t projection_threads = 128;
constexpr std::size_t compute_threads = 256;
constexpr std::size_t retained_limit = 4096;
constexpr std::size_t narrow_output_limit = 1024;
constexpr std::size_t narrow_partials = 16;
constexpr std::size_t prompt_tile = 8;
constexpr std::size_t query_rows = 4;

void require(bool value, const char* message) {
  if (!value) throw std::invalid_argument(message);
}
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::length_error("prepared dimensions overflow");
  }
  return a * b;
}
std::size_t sum(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b) {
    throw std::length_error("prepared dimensions overflow");
  }
  return a + b;
}
std::size_t ceil_div(std::size_t a, std::size_t b) {
  return a / b + (a % b != 0);
}
struct BoundOperand {
  const gpu::DeviceBuffer* owner{};
  gpu::metal::BufferBinding native{};
  std::size_t offset{}, bytes{};
};
void dispatch(gpu::Device& device, gpu::Queue& queue, const char* name,
              const metal::OperationArgs& args,
              std::span<const gpu::metal::BufferBinding> bindings,
              std::size_t groups, std::size_t threads, std::size_t shared = 0) {
  require(groups && product(groups, threads) <=
                        std::numeric_limits<std::uint32_t>::max(),
          "Metal grid exceeds uint32 coordinates");
  auto pipeline = gpu::metal::pipeline(device, DOLITTLE_METAL_LIBRARY, name);
  require(pipeline->threadExecutionWidth() == simd_lanes &&
              threads <= pipeline->maxTotalThreadsPerThreadgroup(),
          "unsupported Metal pipeline geometry");
  Bridge::dispatch(queue, device, *pipeline, bindings, {groups * threads, 1, 1},
                   {threads, 1, 1}, std::as_bytes(std::span(&args, 1)), shared);
}
}  // namespace
namespace metal {
ProjectionPlan plan_projection(ProjectionShape s, tensors::ElementType,
                               KernelLimits limits) {
  require(s.rows && s.input_width && s.output_width &&
              limits.simd_width == simd_lanes &&
              limits.max_threads >= projection_threads,
          "unsupported Metal projection shape");
  const std::size_t partials =
      s.input_width <= retained_limit && s.output_width > narrow_output_limit
          ? narrow_partials
          : simd_lanes;
  const std::size_t rows = s.rows == 1 ? 1 : prompt_tile;
  const std::size_t shared =
      rows == 1 ? sizeof(float) : rows * projection_threads * sizeof(float);
  require(shared <= limits.max_threadgroup_bytes,
          "projection stage exceeds Metal threadgroup memory");
  return {partials, rows, projection_threads / partials, projection_threads,
          shared};
}
AttentionPlan plan_attention(AttentionShape s, tensors::ElementType,
                             KernelLimits limits) {
  require(s.rows && s.query_heads && s.kv_heads && s.width && s.capacity &&
              limits.simd_width == simd_lanes &&
              limits.max_threads >= compute_threads,
          "unsupported Metal attention shape");
  const auto retained =
      product(sum(sum(s.capacity, s.width), compute_threads), sizeof(float));
  if (s.rows == 1 && s.capacity <= retained_limit &&
      retained <= limits.max_threadgroup_bytes) {
    return {1, simd_lanes, s.capacity, compute_threads, retained};
  }
  const auto tiled = product(sum(product(query_rows + simd_lanes, s.width),
                                 2 * query_rows * simd_lanes + 2 * query_rows),
                             sizeof(float));
  require(s.width <= compute_threads && tiled <= limits.max_threadgroup_bytes,
          "attention tile exceeds Metal threadgroup memory");
  return {4, simd_lanes, 0, compute_threads, tiled};
}
}  // namespace metal
struct PreparedOperation::Impl {
  gpu::Device* device;
  Operation operation;
  tensors::ElementType type;
  std::vector<OperandRequirement> requirements;
  std::array<std::size_t, 4> extents{}, dimensions{};
  RowPolicy rows;
  WorkspaceRequirement scratch{0, 1};
  metal::KernelLimits limits;
  Impl(gpu::Device& d, const Operation& op, tensors::ElementType t)
      : device(&d),
        operation(op),
        type(t),
        limits{simd_lanes, compute_threads,
               Bridge::native_device(d).maxThreadgroupMemoryLength()} {}
  ~Impl() { Bridge::drain_noexcept(*device); }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
};
PreparedOperation::PreparedOperation(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedOperation::~PreparedOperation() = default;
PreparedOperation::PreparedOperation(PreparedOperation&&) noexcept = default;
PreparedOperation& PreparedOperation::operator=(PreparedOperation&&) noexcept =
    default;
WorkspaceRequirement PreparedOperation::workspace() const noexcept {
  return impl_->scratch;
}
void PreparedOperation::initialize(gpu::Queue& queue,
                                   gpu::DeviceRegion region) const {
  require(region.bytes == workspace().bytes, "constant region size mismatch");
  if (!region.bytes) return;
  require(region.offset % alignof(float) == 0, "unaligned operation constants");
  const auto binding = Bridge::binding(region, *impl_->device);
  metal::OperationArgs args;
  args.address.at(4) = binding.address;
  std::ranges::copy(impl_->dimensions, args.dimensions.begin());
  if (const auto* op = std::get_if<Rotary>(&impl_->operation)) {
    args.address.at(metal::constant_bits_slot) =
        std::bit_cast<std::uint64_t>(op->theta);
    dispatch(*impl_->device, queue, "rotary_frequencies", args,
             std::span(&binding, 1),
             ceil_div(args.dimensions.at(2) / 2, simd_lanes), simd_lanes);
  } else {
    dispatch(*impl_->device, queue, "attention_scale", args,
             std::span(&binding, 1), 1, 1);
  }
}
std::span<const OperandRequirement> PreparedOperation::operands()
    const noexcept {
  return impl_->requirements;
}
std::string_view PreparedOperation::implementation_name() const noexcept {
  return "metal-direct-fp32-accumulation";
}
PreparedOperation prepare(gpu::Device& device, const Operation& operation,
                          std::span<const tensors::Layout> layouts,
                          tensors::ComputationMode mode, RowPolicy rows) {
  const auto uses = validate_operation(operation, layouts, mode);
  auto impl = std::make_unique<PreparedOperation::Impl>(
      device, operation, mode.activation_storage);
  impl->rows = rows;
  impl->requirements.reserve(uses.size());
  for (std::size_t i = 0; i < uses.size(); ++i) {
    const auto& layout = dolittle::at(layouts, uses.at(i).tensor.value);
    impl->requirements.push_back({uses.at(i).tensor, uses.at(i).access,
                                  tensors::element_bytes(layout.element_type)});
    impl->extents.at(i) = tensors::required_bytes(layout);
  }
  auto& d = impl->dimensions;
  const auto layout = [&](TensorId id) -> const tensors::Layout& {
    return dolittle::at(layouts, id.value);
  };
  const auto count = [&](TensorId id) {
    const auto& l = layout(id);
    return tensors::required_bytes(l) / tensors::element_bytes(l.element_type);
  };

  std::visit(
      [&](const auto& op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, Embedding>) {
          d = {layout(op.tokens).shape.at(0), layout(op.weight).shape.at(1), 0,
               0};

        } else if constexpr (std::is_same_v<T, Projection>) {
          d = {layout(op.input).shape.at(0), layout(op.weight).shape.at(1),
               layout(op.weight).shape.at(0), 0};
        } else if constexpr (std::is_same_v<T, RmsNorm>) {
          d.at(1) = layout(op.input).shape.back();
          d.at(0) = count(op.input) / d.at(1);
        } else if constexpr (std::is_same_v<T, Rotary>) {
          const auto& shape = layout(op.input).shape;
          d = {shape.at(0), shape.at(1), shape.at(2), 0};
          require(std::isfinite(static_cast<float>(op.theta)) &&
                      static_cast<float>(op.theta) > 0,
                  "Metal RoPE theta must fit positive FP32");
        } else if constexpr (std::is_same_v<T, CausalAttention>) {
          const auto& shape = layout(op.query).shape;
          d = {shape.at(0), shape.at(1), layout(op.key).shape.at(1),
               shape.at(2)};
        } else if constexpr (std::is_same_v<T, Add>) {
          d.at(0) = count(op.left);
        } else if constexpr (std::is_same_v<T, SiluMultiply>) {
          d.at(0) = count(op.gate);
        } else if constexpr (std::is_same_v<T, GatherRows>) {
          d.at(0) = tensors::required_bytes(layout(op.input)) /
                    layout(op.input).shape.at(0);
        }
      },
      operation);

  if (rows.last_row) {
    const auto* gather = std::get_if<GatherRows>(&operation);
    require(rows.max_rows && gather &&
                gather->rows == std::vector<std::size_t>{rows.max_rows - 1},
            "last-row policy requires a final-row gather");
  } else if (rows.max_rows) {
    require(!std::holds_alternative<GatherRows>(operation),
            "dynamic gather requires last-row policy");
    // Every non-weight input/output has the admitted leading row count.
    for (const auto& use : uses) {
      bool weight = false;
      std::visit(
          [&](const auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, Embedding> ||
                          std::is_same_v<T, Projection> ||
                          std::is_same_v<T, RmsNorm>) {
              weight = use.tensor.value == op.weight.value;
            }
          },
          operation);
      require(weight || dolittle::at(layouts, use.tensor.value).shape.at(0) ==
                            rows.max_rows,
              "dynamic operation leading dimension mismatch");
    }
  }
  std::visit(
      [&](const auto& op) {
        using T = std::decay_t<decltype(op)>;
        const char* name = nullptr;
        if constexpr (std::is_same_v<T, Embedding>) {
          name = "embedding";
        } else if constexpr (std::is_same_v<T, RmsNorm>) {
          name = "rms_norm";
        } else if constexpr (std::is_same_v<T, Rotary>) {
          name = "rotary";
        } else if constexpr (std::is_same_v<T, Add>) {
          name = "elementwise_add";
        } else if constexpr (std::is_same_v<T, SiluMultiply>) {
          name = "silu_multiply";
        } else if constexpr (std::is_same_v<T, Projection>) {
          for (auto variant : {"projection_16_1", "projection_32_1",
                               "projection_16_8", "projection_32_8"}) {
            (void)gpu::metal::pipeline(device, DOLITTLE_METAL_LIBRARY, variant);
          }
          (void)metal::plan_projection({d.at(0), d.at(1), d.at(2)}, impl->type,
                                       impl->limits);
        } else if constexpr (std::is_same_v<T, CausalAttention>) {
          (void)gpu::metal::pipeline(device, DOLITTLE_METAL_LIBRARY,
                                     "attention_retained");
          name = "attention_tiled";
        }
        if (name) {
          (void)gpu::metal::pipeline(device, DOLITTLE_METAL_LIBRARY, name);
        }
      },
      operation);
  if (std::holds_alternative<Rotary>(operation)) {
    impl->scratch = {product(d.at(2) / 2, sizeof(float)), alignof(float)};
  } else if (std::holds_alternative<CausalAttention>(operation)) {
    impl->scratch = {sizeof(float), alignof(float)};
  }
  return PreparedOperation(std::move(impl));
}
void PreparedOperation::submit(
    gpu::Queue& queue, std::span<const BufferBinding> bindings,
    gpu::DeviceRegion workspace_region, const ExecutionExtent* extent,
    const inference::cache::ContiguousLayerView* cache) const {
  auto& device = *impl_->device;
  (void)Bridge::native_queue(queue, device);
  metal::OperationArgs args;
  std::array<gpu::metal::BufferBinding, metal::address_slots> native{};
  std::size_t native_count = 0;
  const bool dynamic = impl_->rows.max_rows != 0;
  require(dynamic == (extent != nullptr),
          "runtime extent/preparation mismatch");
  if (extent) {
    require(extent->active_rows && extent->active_rows <= impl_->rows.max_rows,
            "active rows outside prepared capacity");
    require(extent->first_position <=
                std::numeric_limits<std::size_t>::max() - extent->active_rows,
            "runtime positions overflow");
  }
  require(!cache || (dynamic &&
                     std::holds_alternative<CausalAttention>(impl_->operation)),
          "cache binding requires dynamic attention");
  require(!dynamic ||
              !std::holds_alternative<CausalAttention>(impl_->operation) ||
              cache,
          "dynamic attention requires an initialized cache prefix");
  require(bindings.size() == impl_->requirements.size(),
          "binding count differs from operation");
  require(workspace_region.bytes == workspace().bytes,
          "operation constant size mismatch");
  if (workspace_region.bytes) {
    require(workspace_region.offset % alignof(float) == 0,
            "unaligned operation constants");
    auto binding = Bridge::binding(workspace_region, device);
    args.address.at(4) = binding.address;
    native.at(native_count++) = binding;
  }
  bool device_position = false;
  if (extent && extent->device_position) {
    const auto region = *extent->device_position;
    require(extent->device_capacity &&
                extent->active_rows <= extent->device_capacity &&
                region.bytes == sizeof(std::uint64_t) &&
                region.offset % alignof(std::uint64_t) == 0,
            "invalid device position binding");
    auto binding = Bridge::binding(region, device);
    args.address.at(metal::position_slot) = binding.address;
    native.at(native_count++) = binding;
    device_position = true;
  }
  const auto positioned = [&](TensorId id) {
    for (const auto& binding : bindings) {
      if (binding.tensor.value == id.value) return binding.positioned_output;
    }
    return false;
  };
  std::array<BoundOperand, 4> bound{};
  for (std::size_t i = 0; i < impl_->requirements.size(); ++i) {
    const auto& required = impl_->requirements.at(i);
    const BufferBinding* match = nullptr;
    for (const auto& candidate : bindings) {
      if (candidate.tensor.value == required.tensor.value) {
        require(match == nullptr, "duplicate tensor binding");
        match = &candidate;
      }
    }
    require(match != nullptr, "missing tensor binding");
    const auto* owner = std::visit(
        [](auto* p) -> const gpu::DeviceBuffer* { return p; }, match->storage);
    require(owner != nullptr, "null tensor storage");
    if (required.access == OperandAccess::write) {
      require(std::holds_alternative<gpu::DeviceBuffer*>(match->storage),
              "write operand has immutable storage");
    }

    if (match->offset >
        std::numeric_limits<std::size_t>::max() - match->bytes) {
      throw std::length_error("binding extent overflows");
    }
    require(match->bytes == impl_->extents.at(i) &&
                match->offset <= owner->size_bytes() &&
                match->bytes <= owner->size_bytes() - match->offset,
            "tensor binding extent mismatch");
    auto bound_bytes = match->bytes;
    if (match->positioned_output) {
      require(device_position && required.access == OperandAccess::write &&
                  (std::holds_alternative<Projection>(impl_->operation) ||
                   std::holds_alternative<Rotary>(impl_->operation)),
              "positioned binding requires a cache producer");
      bound_bytes =
          product(match->bytes / impl_->rows.max_rows, extent->device_capacity);
      require(bound_bytes <= owner->size_bytes() - match->offset,
              "positioned output capacity exceeds allocation");
    }
    const auto view = Bridge::binding(
        gpu::ConstDeviceRegion{owner, match->offset, bound_bytes}, device);
    require(view.address % required.alignment == 0,
            "tensor binding alignment mismatch");
    bound.at(i) = {owner, view, match->offset, bound_bytes};
    native.at(native_count++) = view;
  }
  for (std::size_t i = 0; i < impl_->requirements.size(); ++i) {
    for (std::size_t j = i + 1; j < impl_->requirements.size(); ++j) {
      if (bound.at(i).owner == bound.at(j).owner &&
          (impl_->requirements.at(i).access == OperandAccess::write ||
           impl_->requirements.at(j).access == OperandAccess::write)) {
        require(
            bound.at(i).offset >= bound.at(j).offset + bound.at(j).bytes ||
                bound.at(j).offset >= bound.at(i).offset + bound.at(i).bytes,
            "write operand overlaps another operand");
      }
    }
  }

  const auto input = [&](TensorId id) {
    for (std::size_t i = 0; i < impl_->requirements.size(); ++i) {
      if (impl_->requirements.at(i).tensor.value == id.value) {
        return bound.at(i);
      }
    }
    throw std::logic_error("prepared operand absent");
  };
  auto d = impl_->dimensions;
  if (dynamic && !impl_->rows.last_row) {
    d.at(0) = (d.at(0) / impl_->rows.max_rows) * extent->active_rows;
  }
  std::ranges::copy(d, args.dimensions.begin());
  args.type = impl_->type == tensors::ElementType::bfloat16 ? 1 : 0;
  args.first = extent ? extent->first_position : 0;
  args.capacity = cache ? cache->capacity : d.at(0);
  const auto cache_binding = [&](gpu::ConstDeviceRegion region) {
    const auto bytes =
        product(product(product(cache->capacity, d.at(2)), d.at(3)),
                tensors::element_bytes(impl_->type));
    require(region.bytes == bytes, "cache region extent mismatch");
    const auto view = Bridge::binding(region, device);
    require(view.address % tensors::element_bytes(impl_->type) == 0,
            "cache alignment mismatch");
    for (std::size_t i = 0; i < impl_->requirements.size(); ++i) {
      if (impl_->requirements.at(i).access == OperandAccess::write &&
          bound.at(i).owner == region.owner) {
        require(bound.at(i).offset >= region.offset + bytes ||
                    region.offset >= bound.at(i).offset + bound.at(i).bytes,
                "attention output overlaps cache");
      }
    }
    native.at(native_count++) = view;
    return view.address;
  };
  if (cache) {
    require(
        (!device_position || extent->device_capacity == cache->capacity) &&
            extent->first_position <= cache->capacity &&
            extent->active_rows <= cache->capacity - extent->first_position &&
            cache->shape.kv_heads == d.at(2) &&
            cache->shape.head_width == d.at(3),
        "cached attention shape mismatch");
  }
  for (std::size_t i = 0; i < impl_->requirements.size(); ++i) {
    if (workspace_region.bytes && bound.at(i).owner == workspace_region.owner) {
      require(
          bound.at(i).offset >=
                  workspace_region.offset + workspace_region.bytes ||
              workspace_region.offset >= bound.at(i).offset + bound.at(i).bytes,
          "operand overlaps operation constants");
    }
  }
  const char* name = nullptr;
  std::string projection_name;
  std::size_t groups = 0, threads = compute_threads, shared = 0;
  std::visit(
      [&](const auto& op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, Embedding> ||
                      std::is_same_v<T, Projection> ||
                      std::is_same_v<T, RmsNorm>) {
          if constexpr (std::is_same_v<T, Embedding>) {
            args.address.at(0) = input(op.tokens).native.address;
          } else {
            args.address.at(0) = input(op.input).native.address;
          }
          args.address.at(1) = input(op.weight).native.address;
          args.address.at(2) = input(op.output).native.address;
          if constexpr (std::is_same_v<T, Embedding>) {
            name = "embedding";
            groups = ceil_div(product(d.at(0), d.at(1)), threads);
          }
          if constexpr (std::is_same_v<T, Projection>) {
            const auto plan = metal::plan_projection(
                {d.at(0), d.at(1), d.at(2)}, impl_->type, impl_->limits);
            threads = plan.threads;
            shared = plan.threadgroup_bytes;
            groups = product(ceil_div(d.at(0), plan.row_tile),
                             ceil_div(d.at(2), plan.outputs_per_group));
            projection_name = "projection_" + std::to_string(plan.partials) +
                              "_" + std::to_string(plan.row_tile);
            name = projection_name.c_str();
            args.flags = positioned(op.output) ? 1 : 0;
          }
          if constexpr (std::is_same_v<T, RmsNorm>) {
            name = "rms_norm";
            groups = d.at(0);
            threads = simd_lanes;
            args.scalar = op.epsilon;
            shared = product(sum(d.at(1), simd_lanes), sizeof(float));
            if (d.at(1) > projection_threads &&
                shared <= impl_->limits.max_threadgroup_bytes) {
              threads = compute_threads;
              args.flags = 2;
            } else {
              shared = sizeof(float);
            }
          }
        } else if constexpr (std::is_same_v<T, Rotary>) {
          name = "rotary";
          args.address.at(0) = input(op.input).native.address;
          args.address.at(2) = input(op.output).native.address;
          args.first = extent ? extent->first_position : op.first_position;
          args.flags = positioned(op.output) ? 1 : 0;
          groups = ceil_div(product(product(d.at(0), d.at(1)), d.at(2) / 2),
                            threads);
        } else if constexpr (std::is_same_v<T, CausalAttention>) {
          args.address.at(0) = input(op.query).native.address;
          args.address.at(1) =
              cache ? cache_binding(cache->keys) : input(op.key).native.address;
          args.address.at(2) = cache ? cache_binding(cache->values)
                                     : input(op.value).native.address;
          args.address.at(3) = input(op.output).native.address;
          const auto plan = metal::plan_attention(
              {d.at(0), d.at(1), d.at(2), d.at(3), args.capacity}, impl_->type,
              impl_->limits);
          name = plan.retained_keys ? "attention_retained" : "attention_tiled";
          threads = plan.threads;
          shared = plan.threadgroup_bytes;
          groups = product(ceil_div(d.at(0), plan.query_tile), d.at(1));
        } else if constexpr (std::is_same_v<T, Add> ||
                             std::is_same_v<T, SiluMultiply>) {
          if constexpr (std::is_same_v<T, Add>) {
            name = "elementwise_add";
            args.address.at(0) = input(op.left).native.address;
            args.address.at(1) = input(op.right).native.address;
          } else {
            name = "silu_multiply";
            args.address.at(0) = input(op.gate).native.address;
            args.address.at(1) = input(op.up).native.address;
          }
          args.address.at(2) = input(op.output).native.address;
          groups = ceil_div(d.at(0), threads);
        } else if constexpr (std::is_same_v<T, GatherRows>) {
          const auto source = input(op.input), destination = input(op.output);
          gpu::DeviceBuffer* target = nullptr;
          for (const auto& binding : bindings) {
            if (binding.tensor.value == op.output.value) {
              target = std::get<gpu::DeviceBuffer*>(binding.storage);
            }
          }
          require(target, "gather output absent");
          const auto copy = [&](std::size_t row, std::size_t first,
                                std::size_t rows) {
            queue.copy(
                {target, destination.offset + first * d.at(0), rows * d.at(0)},
                {source.owner, source.offset + row * d.at(0), rows * d.at(0)});
          };
          if (impl_->rows.last_row) {
            copy(extent->active_rows - 1, 0, 1);
          } else {
            for (std::size_t first = 0; first < op.rows.size();) {
              auto end = first + 1;
              while (end < op.rows.size() &&
                     op.rows.at(end) == op.rows.at(end - 1) + 1) {
                ++end;
              }
              copy(op.rows.at(first), first, end - first);
              first = end;
            }
          }
        }
      },
      impl_->operation);
  if (name) {
    dispatch(device, queue, name, args, std::span(native).first(native_count),
             groups, threads, shared);
  }
}
}  // namespace dolittle::kernels
