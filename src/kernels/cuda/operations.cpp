/*
 * Axis X5, CUDA kind: prepared dispatch for the validated operation catalog.
 * Contract: preparation copies descriptors, borrows a stable CUDA device,
 * checks capability/shape/numerical support and reports natural operand
 * alignment plus one persistent FP32 attention scale. initialize writes that
 * scalar on GPU before graph readers; other operations have zero workspace.
 * Submit requires one binding per
 * distinct ID, exact prepared byte extents and mutable outputs on the same
 * device/queue. Overlaps involving writes are rejected; read/read aliases are
 * allowed. Resources stay unmoved/alive until completion. Invalid bindings ->
 * invalid_argument; checked overflow -> length_error; CUDA errors ->
 * runtime_error; bad_alloc propagates. Algorithm: transformer overlay
 * "Preparing the first execution recipe" and "One Qwen3 transformer block":
 * validate descriptors, flatten dimensions once, check capabilities,
 * bind/validate existing regions, enqueue the concrete kernel or contiguous
 * gathered-row copies, check launch status, defer completion to Queue. Dynamic
 * row policy keeps maximum binding extents unchanged. At submit, validate
 * 0<B<=Bmax and S+B<=cache capacity, then scale active launch rows (including
 * flattened head/feature counts). RoPE receives S. Attention uses the declared
 * contiguous K/V regions; last-row gather copies local row B-1. All cache
 * extents/device identities and output overlap are checked before launch; the
 * owning executor guarantees earlier queue-ordered append writes. Design/cost:
 * O(operand ranks + gathered rows) preparation and bounded metadata; submit
 * uses fixed stack bindings, no allocation or waits. Gather copies cost one
 * enqueue per maximal consecutive selected range. Teardown conservatively
 * drains the device, including partially submitted work. No TF32/CPU fallback.
 * Optimization: cached dimensions and owned gather metadata avoid hot-path
 * layout discovery, GPU index uploads and hidden workspace. Numerical work
 * stays on GPU. Attention admission accounts for three width-sized FP32 arrays
 * (query and two value sums), 256 reduction values and three scalar slots.
 * Attention also admits the shared retained-score bound from launch.h; runtime
 * selects it for supported context/head widths and the recomputation path
 * otherwise. This bounds both paths even when dynamic cached rows grow.
 * Optimization consultation, header design and implementation, 2026-10-19:
 * cpp-guidelines I.7 (#ri-post): admitted bytes cover either attention path.
 * cpp-perf-guidelines GPU.5: budget retained-score shared storage explicitly.
 * Source:
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md.
 * RMSNorm admits (width+32) shared floats for the bounded cooperative path,
 * and uses warp registers otherwise. Consultation, design/implementation,
 * 2026-10-20: Core I.7 and performance GPU.5 require this admission to match
 * the staged input and 32 partial sums in normalization.cu.
 * Overflow checks precede
 * byte scaling. Corpus consultation, storage-bound design/implementation,
 * 2026-10-19: cpp-guidelines I.7 (#ri-post): admitted shared bytes cover the
 * launched kernel. cpp-perf-guidelines GDSA.16: count the actual per-output
 * on-chip state. Source:
 * guidelines/gpu-dsa/GDSA.16-tile-through-on-chip-memory.md. Corpus
 * consultation, design/implementation, 2026-10-18: cpp-guidelines I.1
 * (#ri-explicit): explicit bindings, numerical mode and device. cpp-guidelines
 * I.7 (#ri-post): exact region/access/completion contracts. cpp-guidelines R.1
 * (#rr-raii): owned descriptors survive exceptions and teardown.
 * cpp-perf-guidelines GPU.1: keep intermediate/gather values on device. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.9: all tensor/workspace allocation remains in
 * planning. Source:
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 *
 * Static-analysis design/implementation, 2026-10-19:
 * Binding retains byte extents through native CUDA submission. Mutable outputs
 * require the mutable buffer alternative; no const_cast manufactures write
 * access. Validated subspans select byte offsets without changing kernel
 * arithmetic. cpp-guidelines SL.con.3 (#rsl-bounds), I.30 (#ri-encapsulate):
 * make checked host access explicit and confine necessary native API
 * operations. cpp-perf-guidelines COPY.7: borrow spans/references without
 * hidden payload copies. Source:
 * guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 * Device-loop extension (design/implementation, 2026-10-20): a prepared
 * prefill/decode body borrows an immutable-for-this-iteration device position.
 * The graph advances it only after every reader completes. RoPE/attention use
 * that absolute position; positioned outputs address cache rows S..S+B-1.
 * Arithmetic and reduction order are unchanged. Graph storage is admitted for
 * the maximum position; the device loop enforces S+B <= capacity. No per-token
 * host argument update, copy of position, or host launch is required.
 * Core I.7/CP.3: explicit bounds and ordered borrowing. Performance
 * GPU.1/GPU.6: resident control and stable graph arguments
 * (guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md). Both corpora queried and rules
 * read during header design and implementation.
 */
#include "kernels/operations.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "bounds.h"
#include "gpu/cuda/backend.h"
#include "inference/cache/contiguous.h"
#include "kernels/cuda/launch.h"

namespace dolittle::kernels {
namespace {
inline constexpr int block_threads = 256;
inline constexpr int max_grid_blocks = 65535;
inline constexpr int minimum_compute_major = 8;
inline constexpr int minimum_compute_minor = 6;
inline constexpr int projection_tile_width = 16;
inline constexpr std::size_t projection_shared_floats = 528;
inline constexpr std::size_t attention_fixed_shared_floats = 259;

using Bridge = gpu::detail::BackendAccess;

void require(bool value, const char* message) {
  if (!value) {
    throw std::invalid_argument(message);
  }
}
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::length_error("prepared dimension product overflows");
  }
  return a * b;
}
struct BoundOperand {
  const gpu::DeviceBuffer* owner = nullptr;
  const std::byte* pointer = nullptr;
  std::size_t offset = 0;
  std::size_t bytes = 0;
};
}  // namespace

struct PreparedOperation::Impl {
  gpu::Device* device;
  Operation operation;
  tensors::ElementType type;
  std::vector<OperandRequirement> requirements;
  std::array<std::size_t, 4> extents{};
  std::array<std::size_t, 4> dimensions{};
  RowPolicy rows;

  Impl(gpu::Device& owner, const Operation& op, tensors::ElementType storage)
      : device(&owner), operation(op), type(storage) {}
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() { Bridge::drain_noexcept(*device); }
};

PreparedOperation::PreparedOperation(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedOperation::~PreparedOperation() = default;
PreparedOperation::PreparedOperation(PreparedOperation&&) noexcept = default;
PreparedOperation& PreparedOperation::operator=(PreparedOperation&&) noexcept =
    default;
WorkspaceRequirement PreparedOperation::workspace() const noexcept {
  return std::holds_alternative<CausalAttention>(impl_->operation)
             ? WorkspaceRequirement{sizeof(float), alignof(float)}
             : WorkspaceRequirement{0, 1};
}
void PreparedOperation::initialize(gpu::Queue& queue,
                                   gpu::DeviceRegion region) const {
  const auto requirement = workspace();
  require(region.bytes == requirement.bytes, "constant region size mismatch");
  if (!region.bytes) return;
  auto& device = *impl_->device;
  require(region.owner && region.offset % requirement.alignment == 0 &&
              region.offset <= region.owner->size_bytes() &&
              region.bytes <= region.owner->size_bytes() - region.offset,
          "invalid constant region");
  auto* base = static_cast<std::byte*>(Bridge::pointer(*region.owner, device));
  // Native typed boundary after extent/alignment validation.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* value =
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      reinterpret_cast<float*>(std::span(base, region.owner->size_bytes())
                                   .subspan(region.offset)
                                   .data());
  cuda::launch_attention_scale(value, impl_->dimensions.at(3),
                               Bridge::stream(queue, device));
  Bridge::check_launch(device);
}
std::span<const OperandRequirement> PreparedOperation::operands()
    const noexcept {
  return impl_->requirements;
}
std::string_view PreparedOperation::implementation_name() const noexcept {
  return "cuda-direct-fp32-accumulation";
}

PreparedOperation prepare(gpu::Device& device, const Operation& operation,
                          std::span<const tensors::Layout> layouts,
                          tensors::ComputationMode mode, RowPolicy rows) {
  const auto uses = validate_operation(operation, layouts, mode);
  const auto& properties = Bridge::properties(device);
  require(properties.maxThreadsPerBlock >= block_threads &&
              properties.maxThreadsDim[0] >= block_threads &&
              properties.maxGridSize[0] >= max_grid_blocks,
          "CUDA launch limits are unsupported");
  require(properties.major > minimum_compute_major ||
              (properties.major == minimum_compute_major &&
               properties.minor >= minimum_compute_minor),
          "CUDA build requires compute capability 8.6 or later");
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
  std::size_t shared_bytes = 0;
  std::visit(
      [&](const auto& op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, Embedding>) {
          d = {layout(op.tokens).shape.at(0), layout(op.weight).shape.at(1), 0,
               0};
          shared_bytes = sizeof(std::uint32_t);
        } else if constexpr (std::is_same_v<T, Projection>) {
          d = {layout(op.input).shape.at(0), layout(op.weight).shape.at(1),
               layout(op.weight).shape.at(0), 0};
          require(properties.maxThreadsDim[1] >= projection_tile_width,
                  "CUDA projection block is unsupported");
          shared_bytes = projection_shared_floats * sizeof(float);
        } else if constexpr (std::is_same_v<T, RmsNorm>) {
          d.at(1) = layout(op.input).shape.back();
          d.at(0) = count(op.input) / d.at(1);
          if (d.at(1) > cuda::rms_warp_width &&
              d.at(1) <= cuda::rms_staged_width) {
            shared_bytes =
                (d.at(1) + cuda::rms_staged_partials) * sizeof(float);
          }
        } else if constexpr (std::is_same_v<T, Rotary>) {
          const auto& shape = layout(op.input).shape;
          d = {shape.at(0), shape.at(1), shape.at(2), 0};
          require(std::isfinite(static_cast<float>(op.theta)) &&
                      static_cast<float>(op.theta) > 0,
                  "CUDA RoPE theta must fit positive FP32");
        } else if constexpr (std::is_same_v<T, CausalAttention>) {
          const auto& shape = layout(op.query).shape;
          d = {shape.at(0), shape.at(1), layout(op.key).shape.at(1),
               shape.at(2)};
          const auto shared_values = product(d.at(3), 3);
          if (shared_values > std::numeric_limits<std::size_t>::max() -
                                  attention_fixed_shared_floats) {
            throw std::length_error("attention shared storage overflows");
          }
          shared_bytes = product(shared_values + attention_fixed_shared_floats,
                                 sizeof(float));
          if (d.at(3) <= cuda::attention_retained_width) {
            shared_bytes =
                std::max(shared_bytes, cuda::attention_retained_shared_bytes);
          }
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
  require(shared_bytes <= properties.sharedMemPerBlock,
          "operation exceeds CUDA shared memory");
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
  return PreparedOperation(std::move(impl));
}

void PreparedOperation::submit(
    gpu::Queue& queue, std::span<const BufferBinding> bindings,
    gpu::DeviceRegion workspace_region, const ExecutionExtent* extent,
    const inference::cache::ContiguousLayerView* cache) const {
  auto& device = *impl_->device;
  const auto stream = Bridge::stream(queue, device);
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
  const float* attention_scale = nullptr;
  require(workspace_region.bytes == workspace().bytes,
          "operation constant size mismatch");
  if (workspace_region.bytes) {
    require(
        workspace_region.owner &&
            workspace_region.offset % alignof(float) == 0 &&
            workspace_region.offset <= workspace_region.owner->size_bytes() &&
            workspace_region.bytes <=
                workspace_region.owner->size_bytes() - workspace_region.offset,
        "invalid operation constant region");
    const auto* base = static_cast<const std::byte*>(
        Bridge::pointer(*workspace_region.owner, device));
    // Prepared typed device constant, never read on the host.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    attention_scale = reinterpret_cast<const float*>(
        std::span(base, workspace_region.owner->size_bytes())
            .subspan(workspace_region.offset)
            .data());
  }

  const std::size_t* device_position = nullptr;
  if (extent && extent->device_position) {
    const auto region = *extent->device_position;
    require(extent->active_rows > 0 &&
                extent->active_rows <= extent->device_capacity &&
                extent->device_capacity && region.owner &&
                region.bytes == sizeof(std::size_t) &&
                region.offset % alignof(std::size_t) == 0 &&
                region.offset <= region.owner->size_bytes() &&
                region.bytes <= region.owner->size_bytes() - region.offset,
            "invalid device position binding");
    const auto* base =
        static_cast<const std::byte*>(Bridge::pointer(*region.owner, device));
    // Native typed boundary; preparation proves size/alignment and lifetime.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    device_position = reinterpret_cast<const std::size_t*>(
        std::span(base, region.owner->size_bytes())
            .subspan(region.offset)
            .data());
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
    const auto* base =
        static_cast<const std::byte*>(Bridge::pointer(*owner, device));
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
    const auto* pointer =
        std::span(base, owner->size_bytes()).subspan(match->offset).data();
    // Native API byte/address boundary; no unrelated typed object is
    // dereferenced. NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    require(reinterpret_cast<std::uintptr_t>(pointer) % required.alignment == 0,
            "tensor binding alignment mismatch");
    bound.at(i) = {owner, pointer, match->offset, bound_bytes};
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
  const auto input_span = [&](TensorId id) -> std::span<const std::byte> {
    for (std::size_t i = 0; i < impl_->requirements.size(); ++i) {
      if (impl_->requirements.at(i).tensor.value == id.value) {
        return {static_cast<const std::byte*>(bound.at(i).pointer),
                bound.at(i).bytes};
      }
    }
    throw std::logic_error("prepared operand is absent");
  };
  const auto input = [&](TensorId id) -> const void* {
    return input_span(id).data();
  };
  const auto output_span = [&](TensorId id) -> std::span<std::byte> {
    for (const auto& binding : bindings) {
      if (binding.tensor.value == id.value) {
        auto* const* owner = std::get_if<gpu::DeviceBuffer*>(&binding.storage);
        if (!owner || !*owner) {
          throw std::invalid_argument("write operand has immutable storage");
        }
        auto* base = static_cast<std::byte*>(Bridge::pointer(**owner, device));
        return std::span(base, (*owner)->size_bytes())
            .subspan(binding.offset, binding.bytes);
      }
    }
    throw std::logic_error("prepared output is absent");
  };
  const auto output = [&](TensorId id) -> void* {
    return output_span(id).data();
  };
  auto d = impl_->dimensions;
  if (dynamic && !impl_->rows.last_row) {
    d.at(0) = (d.at(0) / impl_->rows.max_rows) * extent->active_rows;
  }
  const void* cached_key = nullptr;
  const void* cached_value = nullptr;
  if (cache) {
    require(
        (!device_position || extent->device_capacity == cache->capacity) &&
            extent->first_position <= cache->capacity &&
            extent->active_rows <= cache->capacity - extent->first_position &&
            cache->shape.kv_heads == d.at(2) &&
            cache->shape.head_width == d.at(3),
        "cached attention shape mismatch");
    const auto bytes =
        product(product(product(cache->capacity, d.at(2)), d.at(3)),
                tensors::element_bytes(impl_->type));
    const auto pointer = [&](gpu::ConstDeviceRegion region) -> const void* {
      require(region.owner && region.bytes == bytes &&
                  region.offset <= region.owner->size_bytes() &&
                  bytes <= region.owner->size_bytes() - region.offset,
              "cache region extent mismatch");
      const auto* base =
          static_cast<const std::byte*>(Bridge::pointer(*region.owner, device));
      // Native API byte/address boundary; no unrelated typed object is
      // dereferenced.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      require(reinterpret_cast<std::uintptr_t>(
                  std::span(base, region.owner->size_bytes())
                      .subspan(region.offset)
                      .data()) %
                      tensors::element_bytes(impl_->type) ==
                  0,
              "cache alignment mismatch");
      for (std::size_t i = 0; i < impl_->requirements.size(); ++i) {
        if (impl_->requirements.at(i).access == OperandAccess::write &&
            bound.at(i).owner == region.owner) {
          require(bound.at(i).offset >= region.offset + bytes ||
                      region.offset >= bound.at(i).offset + bound.at(i).bytes,
                  "attention output overlaps cache");
        }
      }
      return std::span(base, region.owner->size_bytes())
          .subspan(region.offset)
          .data();
    };
    cached_key = pointer(cache->keys);
    cached_value = pointer(cache->values);
  }
  std::visit(
      [&](const auto& op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, Embedding>) {
          cuda::launch_embedding(
              impl_->type, static_cast<const std::uint32_t*>(input(op.tokens)),
              input(op.weight), output(op.output), d.at(0), d.at(1), stream);
        } else if constexpr (std::is_same_v<T, Projection>) {
          cuda::launch_projection(
              impl_->type, input(op.input), input(op.weight), output(op.output),
              d.at(0), d.at(1), d.at(2), stream,
              positioned(op.output) ? device_position : nullptr);
        } else if constexpr (std::is_same_v<T, RmsNorm>) {
          cuda::launch_rms_norm(impl_->type, input(op.input), input(op.weight),
                                output(op.output), d.at(0), d.at(1), op.epsilon,
                                stream);
        } else if constexpr (std::is_same_v<T, Rotary>) {
          cuda::launch_rotary(
              impl_->type, input(op.input), output(op.output), d.at(0), d.at(1),
              d.at(2), extent ? extent->first_position : op.first_position,
              op.theta, stream, device_position, positioned(op.output));
        } else if constexpr (std::is_same_v<T, CausalAttention>) {
          cuda::launch_attention(
              impl_->type, input(op.query), cache ? cached_key : input(op.key),
              cache ? cached_value : input(op.value), output(op.output),
              d.at(0), d.at(1), d.at(2), d.at(3), stream,
              extent ? extent->first_position : 0, device_position,
              cache ? cache->capacity : 0, attention_scale);
        } else if constexpr (std::is_same_v<T, Add>) {
          cuda::launch_add(impl_->type, input(op.left), input(op.right),
                           output(op.output), d.at(0), stream);
        } else if constexpr (std::is_same_v<T, SiluMultiply>) {
          cuda::launch_silu_multiply(impl_->type, input(op.gate), input(op.up),
                                     output(op.output), d.at(0), stream);
        } else if constexpr (std::is_same_v<T, GatherRows>) {
          const auto source = input_span(op.input);
          const auto destination = output_span(op.output);
          if (impl_->rows.last_row) {
            require(extent != nullptr,
                    "last-row copy requires a runtime extent");
            Bridge::check(
                device,
                cudaMemcpyAsync(
                    destination.data(),
                    source.subspan((extent->active_rows - 1) * d.at(0)).data(),
                    d.at(0), cudaMemcpyDeviceToDevice, stream));
            return;
          }
          for (std::size_t first = 0; first < op.rows.size();) {
            auto end = first + 1;
            while (end < op.rows.size() &&
                   op.rows.at(end) == op.rows.at(end - 1) + 1) {
              ++end;
            }
            Bridge::check(
                device,
                cudaMemcpyAsync(
                    destination.subspan(first * d.at(0)).data(),
                    source.subspan(op.rows.at(first) * d.at(0)).data(),
                    (end - first) * d.at(0), cudaMemcpyDeviceToDevice, stream));
            first = end;
          }
        }
      },
      impl_->operation);
  Bridge::check_launch(device);
}
}  // namespace dolittle::kernels
