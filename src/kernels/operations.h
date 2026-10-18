/*
 * Header design; operation preparation and kernels are not implemented.
 * Axis X5: bounded mathematical operations and backend capability selection.
 * Contract: TensorId indexes the supplied layout table; invalid IDs, shapes,
 * strides, constants and unsupported modes fail with invalid_argument. Size
 * overflow -> length_error; device failure -> runtime_error; bad_alloc
 * propagates. Preparation owns copied descriptors, borrows a stable Device,
 * chooses a supported implementation and reports workspace bytes/alignment
 * before tensor allocation. No silent precision change or CPU fallback.
 * operands() declares required alignment/access for each referenced tensor;
 * planner placement satisfies these requirements before binding buffers.
 *
 * Initial operations require dense row-major tensors and distinct output
 * storage, floating inputs/outputs in activation_storage and FP32 reductions.
 * Embedding reads uint32 [N] IDs and [V,D] weights, produces [N,D]. The executor
 * validates every ID < V before upload. Projection uses W[out,in], Y=X*W^T,
 * without bias; X/Y have equal leading row counts, trailing dimensions flatten
 * to in/out. RMSNorm normalizes the final dimension with matching vector weight
 * and positive finite epsilon. Rotary acts on [N,H,d], even d, using split-half
 * pairs (i,i+d/2), angle=(first_position+p)*theta^(-2*i/d), no scaling; theta
 * is positive finite. Attention reads [N,Hq,d], [N,Hkv,d], [N,Hkv,d], emits
 * [N,Hq,d], uses Hq divisible by Hkv, kv_head=floor(h/(Hq/Hkv)), j<=p,
 * and softmax(dot/sqrt(d)). No cache, repeated KV heads or N*N buffer required.
 * Add is elementwise with identical shapes. SiluMultiply computes
 * (gate/(1+exp(-gate)))*up with identical shapes. GatherRows copies selected
 * leading rows bit-for-bit in declared order; indices are fixed host metadata
 * copied during preparation, in range and strictly increasing. No-output-row
 * operations are omitted by planning, never launched with invalid zero grids.
 * BF16 rounding is explicit: RMSNorm rounds normalized values before weight
 * multiplication; Rotary rounds trigonometric values and each product before
 * addition; SiluMultiply rounds SiLU before multiplying up. Projection and
 * attention accumulate in FP32 and round final outputs. No implicit TF32.
 *
 * submit binds exactly the operation's distinct referenced tensor IDs, once
 * each. A write operand needs mutable storage. Extents/alignment and device
 * identity must match preparation; overlapping regions involving an output or
 * workspace are rejected, but read/read aliasing is allowed. All borrowed
 * resources and this prepared operation stay unmoved/alive until queue
 * completion. submit enqueues without host wait;
 * asynchronous errors surface at Queue::wait. Workspace is initialized before
 * each read. No tensor allocation or layout discovery belongs in submit.
 * Prepared operations are move-only; moved-from supports destruction or
 * reassignment only. Zero workspace permits a null owner and zero byte range.
 * validate_operation is pure: check equations/layouts and return distinct
 * tensor access roles, with no device probing or allocation of tensor storage.
 * Repeated read operands are allowed; output may not alias an input. prepare
 * uses this validation before adding backend alignment and launch constraints.
 * Unit tests assert accepted/rejected mathematical contracts, not launch grids.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "One Qwen3 transformer
 * block" and "Preparing the first execution recipe": implement each declared
 * equation, preserving model order and logical values. Model emits the recipe;
 * executor places observation copies, kernels own tiling and launch geometry.
 * Cost: projection 2*rows*in*out FLOPs; attention O(N*N*Hq*d); other operations
 * linear in their data. Report scratch rather than hiding it inside launch.
 * Optimization: resolve dispatch once and reuse workspace, keep intermediate
 * values on device. Fusion cannot erase requested observation values.
 *
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.4 (#ri-typed): operation variants and tensor IDs replace
 * untyped callbacks. I.7 (#ri-post): exact equations, readiness and limits.
 * cpp-perf-guidelines GPU.1: intermediate tensors stay on device.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.9: caller admits/reuses declared workspace.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#pragma once

#include "tensors/tensor.h"
#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace dolittle::gpu {
class Device;
class Queue;
class DeviceBuffer;
struct DeviceRegion;
}

namespace dolittle::kernels {

struct TensorId { std::size_t value; };
struct Embedding { TensorId tokens, weight, output; };
struct Projection { TensorId input, weight, output; };
struct RmsNorm { TensorId input, weight, output; float epsilon; };
struct Rotary {
    TensorId input, output;
    std::size_t first_position;
    double theta;
};
struct CausalAttention { TensorId query, key, value, output; };
struct Add { TensorId left, right, output; };
struct SiluMultiply { TensorId gate, up, output; };
struct GatherRows { TensorId input, output; std::vector<std::size_t> rows; };

using Operation = std::variant<Embedding, Projection, RmsNorm, Rotary,
    CausalAttention, Add, SiluMultiply, GatherRows>;

struct WorkspaceRequirement { std::size_t bytes; std::size_t alignment; };
enum class OperandAccess { read, write };
struct OperandUse { TensorId tensor; OperandAccess access; };
[[nodiscard]] std::vector<OperandUse> validate_operation(
    const Operation& operation, std::span<const tensors::Layout> tensors,
    tensors::ComputationMode mode);
struct OperandRequirement {
    TensorId tensor;
    OperandAccess access;
    std::size_t alignment;
};
struct BufferBinding {
    TensorId tensor;
    std::variant<const gpu::DeviceBuffer*, gpu::DeviceBuffer*> storage;
    std::size_t offset;
    std::size_t bytes;
};

class PreparedOperation {
public:
    ~PreparedOperation();
    PreparedOperation(PreparedOperation&&) noexcept;
    PreparedOperation& operator=(PreparedOperation&&) noexcept;
    PreparedOperation(const PreparedOperation&) = delete;
    PreparedOperation& operator=(const PreparedOperation&) = delete;
    [[nodiscard]] WorkspaceRequirement workspace() const noexcept;
    [[nodiscard]] std::span<const OperandRequirement> operands() const noexcept;
    [[nodiscard]] std::string_view implementation_name() const noexcept;
    void submit(gpu::Queue& queue, std::span<const BufferBinding> bindings,
                gpu::DeviceRegion workspace) const;
private:
    struct Impl;
    explicit PreparedOperation(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
    friend PreparedOperation prepare(
        gpu::Device&, const Operation&, std::span<const tensors::Layout>,
        tensors::ComputationMode);
};

[[nodiscard]] PreparedOperation prepare(
    gpu::Device& device, const Operation& operation,
    std::span<const tensors::Layout> tensors, tensors::ComputationMode mode);

}  // namespace dolittle::kernels
