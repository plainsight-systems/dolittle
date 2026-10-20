/*
 * Pure operation validation and CUDA preparation/submission implement this
 * catalog; mathematical kernels are supplied by the CUDA implementation files.
 * Axis X5: bounded mathematical operations and backend capability selection.
 * Contract: TensorId indexes the supplied layout table; invalid IDs, shapes,
 * strides, constants and unsupported modes fail with invalid_argument. Size
 * overflow -> length_error; device failure -> runtime_error; bad_alloc
 * propagates. Preparation owns copied descriptors, borrows a stable Device,
 * chooses a supported implementation and reports workspace bytes/alignment
 * before tensor allocation. No silent precision change or CPU fallback.
 * CUDA attention owns one initialized FP32 scale in operation workspace;
 * other operations need no global workspace. Operands require natural
 * element alignment. RoPE theta must also fit a positive finite FP32 value.
 * operands() declares required alignment/access for each referenced tensor;
 * planner placement satisfies these requirements before binding buffers.
 *
 * Initial operations require nonempty dense row-major tensors and distinct
 * output storage, floating inputs/outputs in activation_storage and FP32
 * reductions. Activation storage is BF16 or FP32. Singleton-axis strides may
 * vary because they do not change any logical address. Projection X/Y have rank
 * at least two. Embedding reads uint32 [N] IDs and [V,D] weights, produces
 * [N,D]. The executor validates every ID < V before upload. Projection uses
 * W[out,in], Y=X*W^T, without bias; X/Y have equal leading row counts, trailing
 * dimensions flatten to in/out. RMSNorm normalizes the final dimension with
 * matching vector weight and positive finite epsilon. Rotary acts on [N,H,d],
 * even d, using split-half pairs (i,i+d/2),
 * angle=(first_position+p)*theta^(-2*i/d), no scaling; theta is positive
 * finite. Attention reads [N,Hq,d], [N,Hkv,d], [N,Hkv,d], emits [N,Hq,d], uses
 * Hq divisible by Hkv, kv_head=floor(h/(Hq/Hkv)), j<=p, and
 * softmax(dot/sqrt(d)). No cache, repeated KV heads or N*N buffer required. Add
 * is elementwise with identical shapes. SiluMultiply computes
 * (gate/(1+exp(-gate)))*up with identical shapes. GatherRows copies selected
 * leading rows bit-for-bit in declared order; indices are fixed host metadata
 * copied during preparation, in range and strictly increasing. No-output-row
 * operations are omitted by planning, never launched with invalid zero grids.
 * BF16 rounding is explicit: RMSNorm rounds normalized values before weight
 * multiplication; Rotary rounds trigonometric values and each product before
 * addition; SiluMultiply rounds SiLU before multiplying up. Projection and
 * attention accumulate in FP32 and round final outputs. No implicit TF32.
 * Fixed FP32 arithmetic (transformer-execution.md): projection uses 16 or 32
 * strided FMA partials selected by feature widths; RMSNorm uses four-component
 * warp sums; attention dots use eight contiguous feature ranges. Attention
 * rounds normalized probabilities to FP32 before FMA accumulation of values.
 * These orders depend on feature widths and visible keys, never batch size or
 * cache chunking. Corpus consultation, numerical contract design, 2026-10-19:
 * cpp-guidelines I.7 (#ri-post): expose arithmetic guarantees to consumers.
 * cpp-perf-guidelines GDSA.2: state fixed reductions and invariance explicitly.
 * Source: guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 *
 * submit binds exactly the operation's distinct referenced tensor IDs, once
 * each. A write operand needs mutable storage. Extents/alignment and device
 * identity must match preparation; bound bytes equal the prepared extent.
 * Overlapping regions involving an output or
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
 * Corpus consultation, validation/CUDA implementation, 2026-10-18:
 * cpp-guidelines I.4 (#ri-typed): operation variants and tensor IDs replace
 * untyped callbacks. I.7 (#ri-post): exact equations, readiness and limits.
 * ES.103 (#res-overflow): reject overflow in flattening and position bounds.
 * cpp-perf-guidelines GPU.2: require declared dense addressing before dispatch.
 * Source: guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 * cpp-perf-guidelines GPU.1: intermediate tensors stay on device.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.9: caller admits/reuses declared workspace.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 *
 * Incremental operation contract: cached attention reads
 * Q[B,Hq,d], persistent K/V with declared capacity C, committed length S and
 * new rows [S,S+B). Its result is Y[B,Hq,d]. Query row i uses key j <= S+i
 * and KV head floor(h/(Hq/Hkv)); the score/softmax/rounding equations remain
 * those of CausalAttention. Cache append precedes attention for each block.
 *
 * Operation preparation validates Bmax, mode and tensor extents. The cached
 * plan admits C and owns contiguous storage. Submission validates S/B and the
 * cache view's capacity, shape, byte extents and device ownership; the executor
 * guarantees that rows [0,S+B) have been written in queue order. Uninitialized
 * capacity must never be treated as valid sequence data.
 * Rotary likewise receives S as a checked runtime position base. Projection,
 * normalization, embedding and elementwise launches receive active row counts
 * within the prepared shape; inactive rows are neither read nor written.
 * RowPolicy declares the prepared row bound and last-active-row gather.
 * ExecutionExtent supplies S/B for each call; a ContiguousLayerView supplies
 * read-only K/V bindings for dynamic attention. Static full-forward descriptors
 * remain immutable and retain the same extent/alias validation.
 * Cache destinations are disjoint append subregions; old-prefix regions remain
 * read-only. Append sources must survive the ordered copy; attention reads only
 * after that copy. No CPU callback per element or implicit host cache copy.
 * Cost: attention O(B*(S+B)*Hq*d), append O(B*Hkv*d) per block. Workspace is
 * bounded by prepared shapes; no sequence-growing score matrix is required.
 * Overlay: transformer-execution.md, "Attention and positional semantics" and
 * "Prefill and decoding".
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): cache effects, runtime
 * extents and causal visibility are explicit contracts, distinct from buffer
 * capacity. cpp-perf-guidelines GPU.9: reuse admitted scratch without
 * overwriting live K/V. Source:
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines GPU.1: all append/attention intermediates stay on device.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Device-loop extension (design/implementation, 2026-10-20): a prepared
 * prefill/decode body borrows an immutable-for-this-iteration device position.
 * The graph advances it only after every reader completes. RoPE/attention use
 * that absolute position; positioned output bindings address cache row S.
 * Arithmetic and reduction order are unchanged. Graph storage is admitted for
 * the maximum position; the device loop enforces S < capacity. No per-token
 * host argument update, copy of position, or host launch is required.
 * Core I.7/CP.3: explicit bounds and ordered borrowing. Performance
 * GPU.1/GPU.6: resident control and stable graph arguments
 * (guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md). Both corpora queried and rules
 * read during header design and implementation.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "tensors/tensor.h"

namespace dolittle::inference::cache {
struct ContiguousLayerView;
}

namespace dolittle::gpu {
class Device;
class Queue;
class DeviceBuffer;
struct DeviceRegion;
struct ConstDeviceRegion;
}  // namespace dolittle::gpu

namespace dolittle::kernels {

struct TensorId {
  std::size_t value;
};
struct Embedding {
  TensorId tokens, weight, output;
};
struct Projection {
  TensorId input, weight, output;
};
struct RmsNorm {
  TensorId input, weight, output;
  float epsilon;
};
struct Rotary {
  TensorId input, output;
  std::size_t first_position;
  double theta;
};
struct CausalAttention {
  TensorId query, key, value, output;
};
struct Add {
  TensorId left, right, output;
};
struct SiluMultiply {
  TensorId gate, up, output;
};
struct GatherRows {
  TensorId input, output;
  std::vector<std::size_t> rows;
};

using Operation = std::variant<Embedding, Projection, RmsNorm, Rotary,
                               CausalAttention, Add, SiluMultiply, GatherRows>;

struct WorkspaceRequirement {
  std::size_t bytes;
  std::size_t alignment;
};
enum class OperandAccess : std::uint8_t { read, write };
struct OperandUse {
  TensorId tensor;
  OperandAccess access;
};
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
  bool positioned_output = false;
};

// Active rows are bounded by max_rows; last_row selects the final active row.
struct RowPolicy {
  std::size_t max_rows = 0;
  bool last_row = false;
};
struct ExecutionExtent {
  std::size_t first_position{};
  std::size_t active_rows{};
  const gpu::ConstDeviceRegion* device_position = nullptr;
  std::size_t device_capacity = 0;
};
class PreparedOperation {
 public:
  ~PreparedOperation();
  PreparedOperation(PreparedOperation&&) noexcept;
  PreparedOperation& operator=(PreparedOperation&&) noexcept;
  PreparedOperation(const PreparedOperation&) = delete;
  PreparedOperation& operator=(const PreparedOperation&) = delete;
  // Persistent per-operation device constants; initialize once on the owning
  // queue before any submit, including graph capture. The executor suballocates
  // these disjoint regions in one arena, never recomputing values on the CPU.
  [[nodiscard]] WorkspaceRequirement workspace() const noexcept;
  void initialize(gpu::Queue&, gpu::DeviceRegion) const;
  [[nodiscard]] std::span<const OperandRequirement> operands() const noexcept;
  [[nodiscard]] std::string_view implementation_name() const noexcept;
  void submit(
      gpu::Queue& queue, std::span<const BufferBinding> bindings,
      gpu::DeviceRegion workspace, const ExecutionExtent* extent = nullptr,
      const inference::cache::ContiguousLayerView* cache = nullptr) const;

 private:
  struct Impl;
  explicit PreparedOperation(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
  friend PreparedOperation prepare(gpu::Device&, const Operation&,
                                   std::span<const tensors::Layout>,
                                   tensors::ComputationMode, RowPolicy);
};

[[nodiscard]] PreparedOperation prepare(
    gpu::Device& device, const Operation& operation,
    std::span<const tensors::Layout> tensors, tensors::ComputationMode mode,
    RowPolicy rows = {});

}  // namespace dolittle::kernels
