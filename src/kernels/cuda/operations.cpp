/*
 * Axis X5, CUDA kind: prepared dispatch for the validated operation catalog.
 * Contract: preparation copies descriptors, borrows a stable CUDA device, checks
 * capability/shape/numerical support and reports natural operand alignment with
 * zero global workspace. Submit requires one binding per distinct ID, exact
 * prepared byte extents and mutable outputs on the same device/queue. Overlaps
 * involving writes are rejected; read/read aliases are allowed. Resources stay
 * unmoved/alive until completion. Invalid bindings -> invalid_argument; checked
 * overflow -> length_error; CUDA errors -> runtime_error; bad_alloc propagates.
 * Algorithm: transformer overlay "Preparing the first execution recipe" and
 * "One Qwen3 transformer block": validate descriptors, flatten dimensions once,
 * check capabilities, bind/validate existing regions, enqueue the concrete kernel
 * or contiguous gathered-row copies, check launch status, defer completion to Queue.
 * Dynamic row policy keeps maximum binding extents unchanged. At submit,
 * validate 0<B<=Bmax and S+B<=cache capacity, then scale active launch rows
 * (including flattened head/feature counts). RoPE receives S. Attention uses
 * the declared contiguous K/V regions; last-row gather copies local row B-1.
 * All cache extents/device identities and output overlap are checked before
 * launch; the owning executor guarantees earlier queue-ordered append writes.
 * Design/cost: O(operand ranks + gathered rows) preparation and bounded metadata;
 * submit uses fixed stack bindings, no allocation or waits. Gather copies cost
 * one enqueue per maximal consecutive selected range. Teardown conservatively
 * drains the device, including partially submitted work. No TF32/CPU fallback.
 * Optimization: cached dimensions and owned gather metadata avoid hot-path layout
 * discovery, GPU index uploads and hidden workspace. Numerical work stays on GPU.
 * Attention admission accounts for three width-sized FP32 arrays (query and
 * two value sums), 256 reduction values and three scalar slots. RMSNorm uses
 * warp registers without shared scratch. Overflow checks precede byte scaling.
 * Corpus consultation, storage-bound design/implementation, 2026-10-19:
 * cpp-guidelines I.7 (#ri-post): admitted shared bytes cover the launched kernel.
 * cpp-perf-guidelines GDSA.16: count the actual per-output on-chip state.
 * Source: guidelines/gpu-dsa/GDSA.16-tile-through-on-chip-memory.md.
 * Corpus consultation, design/implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit): explicit bindings, numerical mode and device.
 * cpp-guidelines I.7 (#ri-post): exact region/access/completion contracts.
 * cpp-guidelines R.1 (#rr-raii): owned descriptors survive exceptions and teardown.
 * cpp-perf-guidelines GPU.1: keep intermediate/gather values on device. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.9: all tensor/workspace allocation remains in planning.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "kernels/operations.h"
#include "kernels/cuda/launch.h"
#include "gpu/cuda/backend.h"
#include "inference/cache/contiguous.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace dolittle::kernels {
namespace {
using Bridge = gpu::detail::BackendAccess;

void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::length_error("prepared dimension product overflows");
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
    ~Impl() { Bridge::drain_noexcept(*device); }
};

PreparedOperation::PreparedOperation(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PreparedOperation::~PreparedOperation() = default;
PreparedOperation::PreparedOperation(PreparedOperation&&) noexcept = default;
PreparedOperation& PreparedOperation::operator=(PreparedOperation&&) noexcept = default;
WorkspaceRequirement PreparedOperation::workspace() const noexcept { return {0, 1}; }
std::span<const OperandRequirement> PreparedOperation::operands() const noexcept {
    return impl_->requirements;
}
std::string_view PreparedOperation::implementation_name() const noexcept {
    return "cuda-direct-fp32-accumulation";
}

PreparedOperation prepare(gpu::Device &device, const Operation &operation,
                          std::span<const tensors::Layout> layouts, tensors::ComputationMode mode,
                          RowPolicy rows) {
    const auto uses = validate_operation(operation, layouts, mode);
    const auto& properties = Bridge::properties(device);
    require(properties.maxThreadsPerBlock >= 256 && properties.maxThreadsDim[0] >= 256 &&
            properties.maxGridSize[0] >= 65535, "CUDA launch limits are unsupported");
    require(properties.major > 8 || (properties.major == 8 && properties.minor >= 6),
            "CUDA build requires compute capability 8.6 or later");
    auto impl = std::make_unique<PreparedOperation::Impl>(device, operation, mode.activation_storage);
    impl->rows = rows;
    impl->requirements.reserve(uses.size());
    for (std::size_t i = 0; i < uses.size(); ++i) {
        const auto& layout = layouts[uses[i].tensor.value];
        impl->requirements.push_back({uses[i].tensor, uses[i].access,
                                      tensors::element_bytes(layout.element_type)});
        impl->extents[i] = tensors::required_bytes(layout);
    }
    auto& d = impl->dimensions;
    const auto layout = [&](TensorId id) -> const tensors::Layout& { return layouts[id.value]; };
    const auto count = [&](TensorId id) {
        const auto& l = layout(id);
        return tensors::required_bytes(l) / tensors::element_bytes(l.element_type);
    };
    std::size_t shared_bytes = 0;
    std::visit([&](const auto& op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, Embedding>) {
            d = {layout(op.tokens).shape[0], layout(op.weight).shape[1], 0, 0};
            shared_bytes = sizeof(std::uint32_t);
        } else if constexpr (std::is_same_v<T, Projection>) {
            d = {layout(op.input).shape[0], layout(op.weight).shape[1],
                 layout(op.weight).shape[0], 0};
            require(properties.maxThreadsDim[1] >= 16, "CUDA projection block is unsupported");
            shared_bytes = 528 * sizeof(float);
        } else if constexpr (std::is_same_v<T, RmsNorm>) {
            d[1] = layout(op.input).shape.back();
            d[0] = count(op.input) / d[1];
            shared_bytes = 0;
        } else if constexpr (std::is_same_v<T, Rotary>) {
            const auto& shape = layout(op.input).shape;
            d = {shape[0], shape[1], shape[2], 0};
            require(std::isfinite(static_cast<float>(op.theta)) &&
                    static_cast<float>(op.theta) > 0, "CUDA RoPE theta must fit positive FP32");
        } else if constexpr (std::is_same_v<T, CausalAttention>) {
            const auto& shape = layout(op.query).shape;
            d = {shape[0], shape[1], layout(op.key).shape[1], shape[2]};
            const auto shared_values = product(d[3], 3);
            if (shared_values > std::numeric_limits<std::size_t>::max() - 259)
                throw std::length_error("attention shared storage overflows");
            shared_bytes = product(shared_values + 259, sizeof(float));
        } else if constexpr (std::is_same_v<T, Add>) {
            d[0] = count(op.left);
        } else if constexpr (std::is_same_v<T, SiluMultiply>) {
            d[0] = count(op.gate);
        } else if constexpr (std::is_same_v<T, GatherRows>) {
            d[0] = tensors::required_bytes(layout(op.input)) / layout(op.input).shape[0];
        }
    }, operation);
    require(shared_bytes <= properties.sharedMemPerBlock, "operation exceeds CUDA shared memory");
    if (rows.last_row) {
        const auto *gather = std::get_if<GatherRows>(&operation);
        require(rows.max_rows && gather && gather->rows == std::vector<std::size_t>{rows.max_rows - 1},
                "last-row policy requires a final-row gather");
    } else if (rows.max_rows) {
        require(!std::holds_alternative<GatherRows>(operation), "dynamic gather requires last-row policy");
        // Every non-weight input/output has the admitted leading row count.
        for (const auto &use : uses) {
            bool weight = false;
            std::visit(
                [&](const auto &op) {
                    using T = std::decay_t<decltype(op)>;
                    if constexpr (std::is_same_v<T, Embedding> || std::is_same_v<T, Projection> ||
                                  std::is_same_v<T, RmsNorm>)
                        weight = use.tensor.value == op.weight.value;
                },
                operation);
            require(weight || layouts[use.tensor.value].shape[0] == rows.max_rows,
                    "dynamic operation leading dimension mismatch");
        }
    }
    return PreparedOperation(std::move(impl));
}

void PreparedOperation::submit(gpu::Queue &queue, std::span<const BufferBinding> bindings,
                               gpu::DeviceRegion workspace_region, const ExecutionExtent *extent,
                               const inference::cache::ContiguousLayerView *cache) const {
    auto& device = *impl_->device;
    const auto stream = Bridge::stream(queue, device);
    const bool dynamic = impl_->rows.max_rows != 0;
    require(dynamic == (extent != nullptr), "runtime extent/preparation mismatch");
    if (extent) {
        require(extent->active_rows && extent->active_rows <= impl_->rows.max_rows,
                "active rows outside prepared capacity");
        require(extent->first_position <= std::numeric_limits<std::size_t>::max() - extent->active_rows,
                "runtime positions overflow");
    }
    require(!cache || (dynamic && std::holds_alternative<CausalAttention>(impl_->operation)),
            "cache binding requires dynamic attention");
    require(!dynamic || !std::holds_alternative<CausalAttention>(impl_->operation) || cache,
            "dynamic attention requires an initialized cache prefix");
    require(bindings.size() == impl_->requirements.size(), "binding count differs from operation");
    require(workspace_region.bytes == 0, "operation requires zero workspace bytes");
    if (workspace_region.owner) {
        (void)Bridge::pointer(*workspace_region.owner, device);
        require(workspace_region.offset <= workspace_region.owner->size_bytes(),
                "workspace offset exceeds storage");
    } else require(workspace_region.offset == 0, "null workspace has nonzero offset");

    std::array<BoundOperand, 4> bound{};
    for (std::size_t i = 0; i < impl_->requirements.size(); ++i) {
        const auto& required = impl_->requirements[i];
        const BufferBinding* match = nullptr;
        for (const auto& candidate : bindings) {
            if (candidate.tensor.value == required.tensor.value) {
                require(match == nullptr, "duplicate tensor binding");
                match = &candidate;
            }
        }
        require(match != nullptr, "missing tensor binding");
        const auto* owner = std::visit([](auto* p) -> const gpu::DeviceBuffer* { return p; }, match->storage);
        require(owner != nullptr, "null tensor storage");
        if (required.access == OperandAccess::write)
            require(std::holds_alternative<gpu::DeviceBuffer*>(match->storage),
                    "write operand has immutable storage");
        const auto* base = static_cast<const std::byte*>(Bridge::pointer(*owner, device));
        if (match->offset > std::numeric_limits<std::size_t>::max() - match->bytes)
            throw std::length_error("binding extent overflows");
        require(match->bytes == impl_->extents[i] && match->offset <= owner->size_bytes() &&
                match->bytes <= owner->size_bytes() - match->offset, "tensor binding extent mismatch");
        const auto* pointer = base + match->offset;
        require(reinterpret_cast<std::uintptr_t>(pointer) % required.alignment == 0,
                "tensor binding alignment mismatch");
        bound[i] = {owner, pointer, match->offset, match->bytes};
    }
    for (std::size_t i = 0; i < impl_->requirements.size(); ++i) {
        for (std::size_t j = i + 1; j < impl_->requirements.size(); ++j) {
            if (bound[i].owner == bound[j].owner &&
                (impl_->requirements[i].access == OperandAccess::write ||
                 impl_->requirements[j].access == OperandAccess::write)) {
                require(bound[i].offset >= bound[j].offset + bound[j].bytes ||
                        bound[j].offset >= bound[i].offset + bound[i].bytes,
                        "write operand overlaps another operand");
            }
        }
    }
    const auto input = [&](TensorId id) -> const void* {
        for (std::size_t i = 0; i < impl_->requirements.size(); ++i)
            if (impl_->requirements[i].tensor.value == id.value) return bound[i].pointer;
        throw std::logic_error("prepared operand is absent");
    };
    const auto output = [&](TensorId id) -> void* { return const_cast<void*>(input(id)); };
    auto d = impl_->dimensions;
    if (dynamic && !impl_->rows.last_row)
        d[0] = (d[0] / impl_->rows.max_rows) * extent->active_rows;
    const void *cached_key = nullptr;
    const void *cached_value = nullptr;
    if (cache) {
        require(extent->first_position <= cache->capacity &&
                    extent->active_rows <= cache->capacity - extent->first_position &&
                    cache->shape.kv_heads == d[2] && cache->shape.head_width == d[3],
                "cached attention shape mismatch");
        const auto bytes =
            product(product(product(cache->capacity, d[2]), d[3]), tensors::element_bytes(impl_->type));
        const auto pointer = [&](gpu::ConstDeviceRegion region) -> const void * {
            require(region.owner && region.bytes == bytes && region.offset <= region.owner->size_bytes() &&
                        bytes <= region.owner->size_bytes() - region.offset,
                    "cache region extent mismatch");
            const auto *base = static_cast<const std::byte *>(Bridge::pointer(*region.owner, device));
            require(reinterpret_cast<std::uintptr_t>(base + region.offset) %
                            tensors::element_bytes(impl_->type) ==
                        0,
                    "cache alignment mismatch");
            for (std::size_t i = 0; i < impl_->requirements.size(); ++i)
                if (impl_->requirements[i].access == OperandAccess::write && bound[i].owner == region.owner)
                    require(bound[i].offset >= region.offset + bytes ||
                                region.offset >= bound[i].offset + bound[i].bytes,
                            "attention output overlaps cache");
            return base + region.offset;
        };
        cached_key = pointer(cache->keys);
        cached_value = pointer(cache->values);
    }
    std::visit([&](const auto& op) {
        using T = std::decay_t<decltype(op)>;
        if constexpr (std::is_same_v<T, Embedding>) {
            cuda::launch_embedding(impl_->type, static_cast<const std::uint32_t*>(input(op.tokens)),
                input(op.weight), output(op.output), d[0], d[1], stream);
        } else if constexpr (std::is_same_v<T, Projection>) {
            cuda::launch_projection(impl_->type, input(op.input), input(op.weight),
                output(op.output), d[0], d[1], d[2], stream);
        } else if constexpr (std::is_same_v<T, RmsNorm>) {
            cuda::launch_rms_norm(impl_->type, input(op.input), input(op.weight),
                output(op.output), d[0], d[1], op.epsilon, stream);
        } else if constexpr (std::is_same_v<T, Rotary>) {
            cuda::launch_rotary(impl_->type, input(op.input), output(op.output), d[0], d[1], d[2],
                                extent ? extent->first_position : op.first_position, op.theta, stream);
        } else if constexpr (std::is_same_v<T, CausalAttention>) {
            cuda::launch_attention(impl_->type, input(op.query), cache ? cached_key : input(op.key),
                                   cache ? cached_value : input(op.value), output(op.output), d[0], d[1],
                                   d[2], d[3], stream, extent ? extent->first_position : 0);
        } else if constexpr (std::is_same_v<T, Add>) {
            cuda::launch_add(impl_->type, input(op.left), input(op.right), output(op.output), d[0], stream);
        } else if constexpr (std::is_same_v<T, SiluMultiply>) {
            cuda::launch_silu_multiply(impl_->type, input(op.gate), input(op.up), output(op.output), d[0], stream);
        } else if constexpr (std::is_same_v<T, GatherRows>) {
            const auto* source = static_cast<const std::byte*>(input(op.input));
            auto* destination = static_cast<std::byte*>(output(op.output));
            if (impl_->rows.last_row) {
                Bridge::check(device, cudaMemcpyAsync(destination, source + (extent->active_rows - 1) * d[0],
                                                      d[0], cudaMemcpyDeviceToDevice, stream));
                return;
            }
            for (std::size_t first = 0; first < op.rows.size();) {
                auto end = first + 1;
                while (end < op.rows.size() && op.rows[end] == op.rows[end - 1] + 1) ++end;
                Bridge::check(device, cudaMemcpyAsync(destination + first * d[0],
                    source + op.rows[first] * d[0], (end - first) * d[0],
                    cudaMemcpyDeviceToDevice, stream));
                first = end;
            }
        }
    }, impl_->operation);
    Bridge::check_launch(device);
}
}  // namespace dolittle::kernels
