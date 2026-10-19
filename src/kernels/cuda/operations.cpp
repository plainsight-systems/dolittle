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
 * Design/cost: O(operand ranks + gathered rows) preparation and bounded metadata;
 * submit uses fixed stack bindings, no allocation or waits. Gather copies cost
 * one enqueue per maximal consecutive selected range. Teardown conservatively
 * drains the device, including partially submitted work. No TF32/CPU fallback.
 * Optimization: cached dimensions and owned gather metadata avoid hot-path layout
 * discovery, GPU index uploads and hidden workspace. Numerical work stays on GPU.
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

PreparedOperation prepare(gpu::Device& device, const Operation& operation,
                          std::span<const tensors::Layout> layouts,
                          tensors::ComputationMode mode) {
    const auto uses = validate_operation(operation, layouts, mode);
    const auto& properties = Bridge::properties(device);
    require(properties.maxThreadsPerBlock >= 256 && properties.maxThreadsDim[0] >= 256 &&
            properties.maxGridSize[0] >= 65535, "CUDA launch limits are unsupported");
    require(properties.major > 8 || (properties.major == 8 && properties.minor >= 6),
            "CUDA build requires compute capability 8.6 or later");
    auto impl = std::make_unique<PreparedOperation::Impl>(device, operation, mode.activation_storage);
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
            shared_bytes = 256 * sizeof(float);
        } else if constexpr (std::is_same_v<T, Rotary>) {
            const auto& shape = layout(op.input).shape;
            d = {shape[0], shape[1], shape[2], 0};
            require(std::isfinite(static_cast<float>(op.theta)) &&
                    static_cast<float>(op.theta) > 0, "CUDA RoPE theta must fit positive FP32");
        } else if constexpr (std::is_same_v<T, CausalAttention>) {
            const auto& shape = layout(op.query).shape;
            d = {shape[0], shape[1], layout(op.key).shape[1], shape[2]};
            const auto shared_values = product(d[3], 2);
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
    return PreparedOperation(std::move(impl));
}

void PreparedOperation::submit(gpu::Queue& queue, std::span<const BufferBinding> bindings,
                               gpu::DeviceRegion workspace_region) const {
    auto& device = *impl_->device;
    const auto stream = Bridge::stream(queue, device);
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
    const auto& d = impl_->dimensions;
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
            cuda::launch_rotary(impl_->type, input(op.input), output(op.output),
                d[0], d[1], d[2], op.first_position, op.theta, stream);
        } else if constexpr (std::is_same_v<T, CausalAttention>) {
            cuda::launch_attention(impl_->type, input(op.query), input(op.key), input(op.value),
                output(op.output), d[0], d[1], d[2], d[3], stream);
        } else if constexpr (std::is_same_v<T, Add>) {
            cuda::launch_add(impl_->type, input(op.left), input(op.right), output(op.output), d[0], stream);
        } else if constexpr (std::is_same_v<T, SiluMultiply>) {
            cuda::launch_silu_multiply(impl_->type, input(op.gate), input(op.up), output(op.output), d[0], stream);
        } else if constexpr (std::is_same_v<T, GatherRows>) {
            const auto* source = static_cast<const std::byte*>(input(op.input));
            auto* destination = static_cast<std::byte*>(output(op.output));
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
