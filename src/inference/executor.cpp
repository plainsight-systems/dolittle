/*
 * Prepared no-cache execution with owned arenas and result metadata.
 * Axis X1: orchestrate one prepared full causal forward pass.
 * Contract: implement executor.h's borrowed checkpoint/device lifetime, serial
 * reuse, host-ready owned result and failure cleanup. No persistent KV cache,
 * sampler, tokenizer or NLA. Validate all token IDs before enqueuing their upload.
 * Algorithm: transformer overlay, "Preparing the first execution recipe",
 * "Configurable observation points" and "Storage lifetimes":
 * 1. Resolve selections; obtain and validate the model recipe; prepare kernels
 *    and workspace/operand requirements without allocating tensor storage.
 * 2. Insert requested row copies into the ordered schedule. Extend source last
 *    uses through these copies; place scratch and admit all weight, retained
 *    output, workspace, alignment and staging bytes before allocation/upload.
 * 3. Own one queue and admitted arenas. Upload weights through bounded staging,
 *    waiting before staging reuse during setup. No full-weight pinned duplicate.
 * 4. Per call, copy token provenance, upload IDs, execute prepared operations and
 *    capture copies, enqueue final readbacks and wait for completion. Copy
 *    completed staging into result-owned bytes with stable metadata/views.
 * 5. On failure drain submitted work before releasing its resources; publish no
 *    successful partial result. Poisoned devices invalidate plans.
 * Design and cost: model/placement setup is amortized. Per-call host allocations,
 * input upload and final readback remain explicit. Optimization: no per-layer host
 * wait or tensor allocation, no all-position vocabulary projection when unneeded.
 * Build views only after owner storage is stable. Result destruction needs no GPU.
 * Pure validators/placement are unit-tested independently. Actual forward/device
 * execution and teardown validation belong to the limited integration scenarios.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): resource ownership survives exceptions and teardown.
 * cpp-guidelines I.7 (#ri-post): readiness and failure cleanup precede publication.
 * cpp-perf-guidelines GPU.7: completion tracks resource dependencies, not global waits.
 * Source: guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 * cpp-perf-guidelines GPU.9: prepare storage once and preserve last-use lifetimes.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "inference/executor.h"

#include "gpu/device.h"
#include "inference/storage_plan.h"
#include "models/checkpoint.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace dolittle::inference {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::length_error("forward byte extent overflows");
    return a + b;
}
std::size_t multiply(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::length_error("forward byte product overflows");
    return a * b;
}
std::size_t align_up(std::size_t n, std::size_t alignment) {
    if (!alignment || (alignment & (alignment - 1)))
        throw std::invalid_argument("invalid operation alignment");
    return add(n, (alignment - n % alignment) % alignment);
}
bool same_point(models::ObservationPoint a, models::ObservationPoint b) {
    return a.boundary == b.boundary && a.block.value == b.block.value;
}
enum class Arena { scratch, weight, output };
struct Slot { Arena arena = Arena::scratch; std::size_t offset = 0, bytes = 0, alignment = 1; };
struct Step {
    kernels::PreparedOperation operation;
    std::vector<kernels::BufferBinding> bindings;
};
struct ResultCapture {
    models::ObservationSite site;
    std::vector<models::TokenPosition> positions;
    std::size_t offset, bytes;
};
}  // namespace

struct ForwardResult::Impl {
    ExecutionId id;
    std::string identity;
    std::vector<std::uint32_t> tokens;
    tensors::ComputationMode mode;
    std::size_t vocabulary;
    std::vector<models::TokenPosition> logit_positions;
    std::vector<std::byte> logits;
    std::vector<std::byte> capture_bytes;
    std::vector<ResultCapture> capture_metadata;
    std::vector<CapturedRows> capture_views;
};
ForwardResult::ForwardResult(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ForwardResult::~ForwardResult() = default;
ForwardResult::ForwardResult(ForwardResult&&) noexcept = default;
ForwardResult& ForwardResult::operator=(ForwardResult&&) noexcept = default;
ExecutionId ForwardResult::execution_id() const noexcept { return impl_->id; }
std::string_view ForwardResult::checkpoint_identity() const noexcept { return impl_->identity; }
std::span<const std::uint32_t> ForwardResult::token_ids() const noexcept { return impl_->tokens; }
tensors::ComputationMode ForwardResult::numerical_mode() const noexcept { return impl_->mode; }
LogitRows ForwardResult::logits() const noexcept {
    return {impl_->logit_positions, impl_->vocabulary, impl_->mode.activation_storage, impl_->logits};
}
std::span<const CapturedRows> ForwardResult::captures() const noexcept { return impl_->capture_views; }

struct FullForwardPlan::Impl {
    const models::BoundCheckpoint* checkpoint;
    models::SequenceLength length;
    tensors::ComputationMode mode;
    models::ForwardRecipe recipe;
    ObservationPlan observations;
    std::vector<models::TokenPosition> logit_positions;
    std::vector<Slot> slots;
    std::vector<Step> steps;
    std::vector<std::optional<kernels::TensorId>> capture_slots;
    std::size_t peak = 0, host_bytes = 0, output_bytes = 0, logit_bytes = 0;
    bool failed = false;
    std::optional<gpu::Queue> queue;
    std::optional<gpu::DeviceBuffer> weights, scratch, output, workspace;
    std::optional<gpu::HostBuffer> staging, readback;

    Impl(const models::BoundCheckpoint& checkpoint_in, models::SequenceLength length_in,
         tensors::ComputationMode mode_in, models::ForwardRecipe recipe_in,
         ObservationPlan observations_in, std::vector<models::TokenPosition> positions)
        : checkpoint(&checkpoint_in), length(length_in), mode(mode_in),
          recipe(std::move(recipe_in)), observations(std::move(observations_in)),
          logit_positions(std::move(positions)) {}
    ~Impl() {
        if (queue) { try { queue->drain(); } catch (...) {} }
    }
    gpu::DeviceBuffer* buffer(Arena arena) {
        switch (arena) {
        case Arena::weight: return &*weights;
        case Arena::scratch: return &*scratch;
        case Arena::output: return &*output;
        }
        throw std::logic_error("unknown forward arena");
    }
};
FullForwardPlan::FullForwardPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FullForwardPlan::~FullForwardPlan() = default;
FullForwardPlan::FullForwardPlan(FullForwardPlan&&) noexcept = default;
FullForwardPlan& FullForwardPlan::operator=(FullForwardPlan&&) noexcept = default;
models::SequenceLength FullForwardPlan::sequence_length() const noexcept { return impl_->length; }
std::size_t FullForwardPlan::estimated_peak_device_bytes() const noexcept { return impl_->peak; }
std::size_t FullForwardPlan::host_output_bytes() const noexcept { return impl_->host_bytes; }

FullForwardPlan prepare_full_forward(gpu::Device& device,
    const models::BoundCheckpoint& checkpoint, models::SequenceLength length,
    const ForwardOptions& options) {
    const auto& model = checkpoint.model();
    if (!length.value || length.value > model.context_limit().value)
        throw std::invalid_argument("forward length outside context");
    std::vector<models::TokenPosition> positions;
    if (options.logit_positions.valueless_by_exception())
        throw std::invalid_argument("missing logit selection");
    const auto all = std::holds_alternative<AllPositions>(options.logit_positions);
    const auto count = all ? length.value :
        std::get<std::vector<models::TokenPosition>>(options.logit_positions).size();
    const auto logit_bytes = multiply(multiply(count, model.vocabulary_size()),
                                      tensors::element_bytes(options.numerical_mode.activation_storage));
    if (logit_bytes > options.max_device_bytes)
        throw std::length_error("logit selection exceeds device budget");
    positions.reserve(count);
    if (all) {
        for (std::size_t i = 0; i < length.value; ++i) positions.push_back({i});
    } else {
        positions = std::get<std::vector<models::TokenPosition>>(options.logit_positions);
    }
    auto observation_plan = prepare_observations(model.observation_sites(options.numerical_mode),
        length, options.observations, options.observation_limits);
    auto recipe = model.full_forward(length, options.numerical_mode, positions);
    models::validate_recipe(model, recipe, length, options.numerical_mode, positions);
    auto plan = std::make_unique<FullForwardPlan::Impl>(checkpoint, length,
        options.numerical_mode, std::move(recipe), std::move(observation_plan), std::move(positions));
    plan->logit_bytes = logit_bytes;
    const auto captures = plan->observations.entries();
    plan->capture_slots.resize(captures.size());
    std::vector<std::vector<kernels::Operation>> copies(plan->recipe.tensors.size());
    for (std::size_t c = 0; c < captures.size(); ++c) {
        const auto& capture = captures[c];
        if (capture.positions.empty()) continue;
        const auto found = std::find_if(plan->recipe.observations.begin(), plan->recipe.observations.end(),
            [&](const auto& value) { return same_point(value.site.point, capture.site.point); });
        if (found == plan->recipe.observations.end()) throw std::logic_error("unresolved observation");
        const kernels::TensorId target{plan->recipe.tensors.size()};
        plan->capture_slots[c] = target;
        plan->recipe.tensors.push_back(tensors::contiguous_layout(capture.site.storage_type,
            std::array{capture.positions.size(), capture.site.vector_width}));
        std::vector<std::size_t> rows;
        rows.reserve(capture.positions.size());
        for (auto position : capture.positions) rows.push_back(position.value);
        copies[found->tensor.value].emplace_back(kernels::GatherRows{found->tensor, target, std::move(rows)});
    }
    // Resolve every operation, including captures, before allocating tensor storage.
    for (const auto& operation : plan->recipe.operations) {
        auto prepared = kernels::prepare(device, operation, plan->recipe.tensors, plan->mode);
        std::vector<kernels::TensorId> produced;
        for (const auto& use : prepared.operands())
            if (use.access == kernels::OperandAccess::write) produced.push_back(use.tensor);
        plan->steps.push_back({std::move(prepared), {}});
        for (auto output : produced)
            for (const auto& copy : copies[output.value])
                plan->steps.push_back({kernels::prepare(device, copy, plan->recipe.tensors, plan->mode), {}});
    }
    plan->slots.resize(plan->recipe.tensors.size());
    std::vector<std::size_t> first(plan->slots.size(), std::numeric_limits<std::size_t>::max());
    std::vector<std::size_t> last(plan->slots.size(), 0);
    for (std::size_t i = 0; i < plan->slots.size(); ++i) {
        plan->slots[i].bytes = tensors::required_bytes(plan->recipe.tensors[i]);
        plan->slots[i].alignment = tensors::element_bytes(plan->recipe.tensors[i].element_type);
    }
    std::size_t workspace_bytes = 0, workspace_alignment = 1;
    for (std::size_t s = 0; s < plan->steps.size(); ++s) {
        const auto& op = plan->steps[s].operation;
        for (const auto& use : op.operands()) {
            auto& slot = plan->slots[use.tensor.value];
            slot.alignment = std::max(slot.alignment, use.alignment);
            first[use.tensor.value] = std::min(first[use.tensor.value], s);
            last[use.tensor.value] = s;
        }
        workspace_bytes = std::max(workspace_bytes, op.workspace().bytes);
        workspace_alignment = std::max(workspace_alignment, op.workspace().alignment);
    }
    std::size_t weight_bytes = 0, weight_alignment = 1;
    for (const auto& input : plan->recipe.parameters) {
        auto& slot = plan->slots[input.tensor.value];
        slot.arena = Arena::weight;
        const auto view = checkpoint.parameter(input.parameter);
        tensors::validate_view(view);
        const auto& expected = plan->recipe.tensors[input.tensor.value];
        if (view.layout->shape != expected.shape || view.layout->element_type != expected.element_type ||
            view.bytes.size() < slot.bytes)
            throw std::invalid_argument("checkpoint parameter differs from recipe");
        for (std::size_t axis = 0; axis < expected.shape.size(); ++axis)
            if (expected.shape[axis] > 1 && view.layout->byte_strides[axis] != expected.byte_strides[axis])
                throw std::invalid_argument("checkpoint parameter strides differ from recipe");
        // Empty logit selections need no final projection residency.
        if (first[input.tensor.value] == std::numeric_limits<std::size_t>::max()) continue;
        weight_bytes = align_up(weight_bytes, slot.alignment);
        slot.offset = weight_bytes;
        weight_bytes = add(weight_bytes, slot.bytes);
        weight_alignment = std::max(weight_alignment, slot.alignment);
    }
    std::size_t output_alignment = 1;
    const auto retain = [&](kernels::TensorId id) {
        auto& slot = plan->slots[id.value];
        slot.arena = Arena::output;
        slot.offset = align_up(plan->output_bytes, slot.alignment);
        plan->output_bytes = add(slot.offset, slot.bytes);
        output_alignment = std::max(output_alignment, slot.alignment);
    };
    if (plan->recipe.logits) retain(*plan->recipe.logits);
    for (const auto id : plan->capture_slots) if (id) retain(*id);
    first[plan->recipe.token_input.value] = 0;
    std::vector<ResourceLifetime> resources;
    for (std::size_t i = 0; i < plan->slots.size(); ++i) {
        const auto& slot = plan->slots[i];
        if (slot.arena == Arena::scratch) {
            if (first[i] == std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument("scratch slot has no use or producer");
            resources.push_back({{i}, slot.bytes, slot.alignment, first[i], last[i]});
        }
    }
    const auto fixed = add(add(weight_bytes, plan->output_bytes), workspace_bytes);
    if (fixed > options.max_device_bytes) throw std::length_error("forward fixed arenas exceed budget");
    const auto storage = plan_storage(resources, options.max_device_bytes - fixed);
    plan->peak = add(fixed, storage.arena_bytes);
    for (const auto& placement : storage.placements) plan->slots[placement.id.value].offset = placement.offset;
    const auto token_bytes = multiply(length.value, sizeof(std::uint32_t));
    plan->host_bytes = add(add(logit_bytes, plan->observations.payload_bytes()), token_bytes);
    plan->host_bytes = add(plan->host_bytes, multiply(plan->logit_positions.size(), sizeof(models::TokenPosition)));
    for (const auto& capture : captures) {
        plan->host_bytes = add(plan->host_bytes, multiply(capture.positions.size(), sizeof(models::TokenPosition)));
        plan->host_bytes = add(plan->host_bytes, sizeof(ResultCapture) + sizeof(CapturedRows));
    }
    plan->host_bytes = add(plan->host_bytes, checkpoint.identity().size());
    plan->queue.emplace(device.create_queue());
    if (weight_bytes) plan->weights.emplace(device.allocate({weight_bytes, weight_alignment}));
    if (storage.arena_bytes) plan->scratch.emplace(device.allocate({storage.arena_bytes, storage.arena_alignment}));
    if (plan->output_bytes) {
        plan->output.emplace(device.allocate({plan->output_bytes, output_alignment}));
        plan->readback.emplace(device.allocate_staging({plan->output_bytes, output_alignment}));
    }
    if (workspace_bytes) plan->workspace.emplace(device.allocate({workspace_bytes, workspace_alignment}));
    // Bounded weight staging is reused serially during setup; token staging may
    // require more bytes for an explicitly admitted long sequence.
    const auto staging_bytes = std::max(token_bytes, std::min(weight_bytes, std::size_t{16} * 1024 * 1024));
    plan->staging.emplace(device.allocate_staging({staging_bytes, alignof(std::uint32_t)}));
    for (const auto& input : plan->recipe.parameters) {
        if (first[input.tensor.value] == std::numeric_limits<std::size_t>::max()) continue;
        const auto view = checkpoint.parameter(input.parameter);
        const auto& slot = plan->slots[input.tensor.value];
        for (std::size_t offset = 0; offset < slot.bytes;) {
            const auto bytes = std::min(staging_bytes, slot.bytes - offset);
            std::memcpy(plan->staging->bytes().data(), view.bytes.data() + offset, bytes);
            plan->queue->upload({&*plan->weights, slot.offset + offset, bytes}, {&*plan->staging, 0, bytes});
            plan->queue->wait(plan->queue->mark());
            offset += bytes;
        }
    }
    for (auto& step : plan->steps) {
        for (const auto& use : step.operation.operands()) {
            const auto& slot = plan->slots[use.tensor.value];
            auto* owner = plan->buffer(slot.arena);
            if (use.access == kernels::OperandAccess::read)
                step.bindings.push_back({use.tensor, static_cast<const gpu::DeviceBuffer*>(owner), slot.offset, slot.bytes});
            else step.bindings.push_back({use.tensor, owner, slot.offset, slot.bytes});
        }
    }
    return FullForwardPlan(std::move(plan));
}

ForwardResult FullForwardPlan::execute(std::span<const std::uint32_t> token_ids, ExecutionId id) {
    auto& plan = *impl_;
    if (plan.failed) throw std::runtime_error("forward plan unusable after execution failure");
    if (token_ids.size() != plan.length.value) throw std::invalid_argument("wrong token count");
    for (const auto token : token_ids)
        if (token >= plan.checkpoint->model().vocabulary_size()) throw std::invalid_argument("token outside vocabulary");
    auto result = std::make_unique<ForwardResult::Impl>();
    result->id = id;
    result->identity = plan.checkpoint->identity();
    result->tokens.assign(token_ids.begin(), token_ids.end());
    result->mode = plan.mode;
    result->vocabulary = plan.checkpoint->model().vocabulary_size();
    result->logit_positions = plan.logit_positions;
    result->logits.resize(plan.logit_bytes);
    result->capture_bytes.resize(plan.observations.payload_bytes());
    const auto captures = plan.observations.entries();
    result->capture_metadata.reserve(captures.size());
    result->capture_views.reserve(captures.size());
    for (const auto& capture : captures)
        result->capture_metadata.push_back({capture.site, {capture.positions.begin(), capture.positions.end()},
                                            capture.byte_offset, capture.byte_count});
    for (const auto& capture : result->capture_metadata)
        result->capture_views.push_back({capture.site, capture.positions,
            std::span<const std::byte>(result->capture_bytes).subspan(capture.offset, capture.bytes)});
    try {
        const auto& tokens = plan.slots[plan.recipe.token_input.value];
        std::memcpy(plan.staging->bytes().data(), token_ids.data(), tokens.bytes);
        plan.queue->upload({plan.buffer(tokens.arena), tokens.offset, tokens.bytes},
                           {&*plan.staging, 0, tokens.bytes});
        for (const auto& step : plan.steps) {
            const auto bytes = step.operation.workspace().bytes;
            step.operation.submit(*plan.queue, step.bindings,
                {bytes ? &*plan.workspace : nullptr, 0, bytes});
        }
        if (plan.output_bytes)
            plan.queue->download({&*plan.readback, 0, plan.output_bytes}, {&*plan.output, 0, plan.output_bytes});
        plan.queue->wait(plan.queue->mark());
        if (plan.recipe.logits) {
            const auto& slot = plan.slots[plan.recipe.logits->value];
            std::memcpy(result->logits.data(), plan.readback->bytes().data() + slot.offset, slot.bytes);
        }
        for (std::size_t c = 0; c < captures.size(); ++c) {
            if (!plan.capture_slots[c]) continue;
            const auto& slot = plan.slots[plan.capture_slots[c]->value];
            std::memcpy(result->capture_bytes.data() + captures[c].byte_offset,
                        plan.readback->bytes().data() + slot.offset, slot.bytes);
        }
    } catch (...) {
        plan.failed = true;
        try { plan.queue->drain(); } catch (...) {}
        throw;
    }
    return ForwardResult(std::move(result));
}
}  // namespace dolittle::inference
