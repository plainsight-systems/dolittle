/*
 * Prepared full and cached execution with owned arenas and result metadata.
 * Axis X1: orchestrate full passes and serial incremental token consumption.
 * Contract: implement executor.h's borrowed checkpoint/device lifetime, serial
 * reuse, host-ready owned result and failure cleanup. Sampling, tokenization
 * and NLA execution are separate responsibilities. Validate all token IDs before enqueuing their upload.
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
 * Cached algorithm: inputs are B new IDs, committed prefix length S, prepared
 * model/cache bindings and selected semantic sites. Output is an owned chunk.
 * 1. Admit C positions and Bmax scratch once. Obtain K/V producer IDs and the
 *    logit-tail boundary from IncrementalRecipe; do not infer model names.
 * 2. Upload active IDs only; execute the bounded recipe with runtime rows B
 *    and RoPE base S. Before each attention step, copy its newly rotated K and
 *    projected V to the owned per-block cache at [S,S+B).
 * 3. Attention reads keys j<=S+i for local query i. Copy each selected residual
 *    after its producer and before its scratch storage can be reused.
 * 4. When logits are requested, gather local row B-1 and run final norm and
 *    vocabulary projection. Otherwise skip that tail; all blocks still run.
 * 5. Enqueue the logit/capture downloads and wait once. Materialize owned host
 *    bytes, then commit history and S+B. A submitted failure invalidates the
 *    entire cache; earlier returned chunks remain owned and readable.
 * Optimization: one weight/cache/scratch allocation per turn, one prepared
 * schedule for both chunked prefill and B=1 decode. Per-call host results and
 * one final wait remain explicit costs. Cache lifecycle limitations are in
 * cache/contiguous.h; no pooling or cross-turn prefix reuse is implemented.
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

 * Static-analysis design/implementation, 2026-10-19:
 * Checked resource access rejects missing prepared optionals before use; span
 * indexing uses bounds.h. Destruction drains without throwing; execution failures
 * poison the plan before unwinding. These checks add no GPU work or copies.
 * cpp-guidelines SL.con.3 (#rsl-bounds), I.30 (#ri-encapsulate): make checked
 * host access explicit and confine necessary native API operations.
 * cpp-perf-guidelines COPY.7: borrow spans/references without hidden payload copies.
 * Source: guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 */
#include "bounds.h"
#include <cstdint>
#include "inference/executor.h"

#include "gpu/device.h"
#include "inference/storage_plan.h"
#include "inference/cache/contiguous.h"
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
template <class Optional> decltype(auto) present(Optional &value) {
    if (!value.has_value()) {
        throw std::logic_error("prepared execution resource is absent");
    }
    return value.value();
}
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::length_error("forward byte extent overflows");
    }
    return a + b;
}
std::size_t multiply(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::length_error("forward byte product overflows");
    }
    return a * b;
}
std::size_t align_up(std::size_t n, std::size_t alignment) {
    if (!alignment || (alignment & (alignment - 1))) {
        throw std::invalid_argument("invalid operation alignment");
    }
    return add(n, (alignment - n % alignment) % alignment);
}
bool same_point(models::ObservationPoint a, models::ObservationPoint b) {
    return a.boundary == b.boundary && a.block.value == b.block.value;
}
enum class Arena : std::uint8_t { scratch, weight, output };
struct Slot {
    Arena arena = Arena::scratch;
    std::size_t offset = 0, bytes = 0, alignment = 1;
};
struct Step {
    kernels::PreparedOperation operation;
    std::vector<kernels::BufferBinding> bindings;
};
struct ResultCapture {
    models::ObservationSite site;
    std::vector<models::TokenPosition> positions;
    std::size_t offset, bytes;
};
} // namespace

struct ForwardResult::Impl {
    ExecutionId id{};
    std::string identity;
    std::vector<std::uint32_t> tokens;
    tensors::ComputationMode mode{};
    std::size_t vocabulary{};
    std::vector<models::TokenPosition> logit_positions;
    std::vector<std::byte> logits;
    std::vector<std::byte> capture_bytes;
    std::vector<ResultCapture> capture_metadata;
    std::vector<CapturedRows> capture_views;
};
ForwardResult::ForwardResult(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ForwardResult::~ForwardResult() = default;
ForwardResult::ForwardResult(ForwardResult &&) noexcept = default;
ForwardResult &ForwardResult::operator=(ForwardResult &&) noexcept = default;
ExecutionId ForwardResult::execution_id() const noexcept { return impl_->id; }
std::string_view ForwardResult::checkpoint_identity() const noexcept { return impl_->identity; }
std::span<const std::uint32_t> ForwardResult::token_ids() const noexcept { return impl_->tokens; }
tensors::ComputationMode ForwardResult::numerical_mode() const noexcept { return impl_->mode; }
LogitRows ForwardResult::logits() const noexcept {
    return {impl_->logit_positions, impl_->vocabulary, impl_->mode.activation_storage, impl_->logits};
}
std::span<const CapturedRows> ForwardResult::captures() const noexcept { return impl_->capture_views; }

struct FullForwardPlan::Impl {
    const models::BoundCheckpoint *checkpoint;
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

    Impl(const models::BoundCheckpoint &checkpoint_in, models::SequenceLength length_in,
         tensors::ComputationMode mode_in, models::ForwardRecipe recipe_in, ObservationPlan observations_in,
         std::vector<models::TokenPosition> positions)
        : checkpoint(&checkpoint_in), length(length_in), mode(mode_in), recipe(std::move(recipe_in)),
          observations(std::move(observations_in)), logit_positions(std::move(positions)) {}
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;
    Impl(Impl &&) = delete;
    Impl &operator=(Impl &&) = delete;
    ~Impl() {
        if (queue) {
            try {
                present(queue).drain();
            } catch (...) { // NOLINT(bugprone-empty-catch)
                // Preserve the active exception; buffer owners perform a final no-throw drain.
            }
        }
    }
    gpu::DeviceBuffer *buffer(Arena arena) {
        switch (arena) {
        case Arena::weight:
            return &present(weights);
        case Arena::scratch:
            return &present(scratch);
        case Arena::output:
            return &present(output);
        }
        throw std::logic_error("unknown forward arena");
    }
};
FullForwardPlan::FullForwardPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FullForwardPlan::~FullForwardPlan() = default;
FullForwardPlan::FullForwardPlan(FullForwardPlan &&) noexcept = default;
FullForwardPlan &FullForwardPlan::operator=(FullForwardPlan &&) noexcept = default;
models::SequenceLength FullForwardPlan::sequence_length() const noexcept { return impl_->length; }
std::size_t FullForwardPlan::estimated_peak_device_bytes() const noexcept { return impl_->peak; }
std::size_t FullForwardPlan::host_output_bytes() const noexcept { return impl_->host_bytes; }

FullForwardPlan prepare_full_forward(gpu::Device &device, const models::BoundCheckpoint &checkpoint,
                                     models::SequenceLength length, const ForwardOptions &options) {
    const auto &model = checkpoint.model();
    if (!length.value || length.value > model.context_limit().value) {
        throw std::invalid_argument("forward length outside context");
    }
    std::vector<models::TokenPosition> positions;
    if (options.logit_positions.valueless_by_exception()) {
        throw std::invalid_argument("missing logit selection");
    }
    const auto all = std::holds_alternative<AllPositions>(options.logit_positions);
    const auto count =
        all ? length.value : std::get<std::vector<models::TokenPosition>>(options.logit_positions).size();
    const auto logit_bytes = multiply(multiply(count, model.vocabulary_size()),
                                      tensors::element_bytes(options.numerical_mode.activation_storage));
    if (logit_bytes > options.max_device_bytes) {
        throw std::length_error("logit selection exceeds device budget");
    }
    positions.reserve(count);
    if (all) {
        for (std::size_t i = 0; i < length.value; ++i) {
            positions.push_back({i});
        }
    } else {
        positions = std::get<std::vector<models::TokenPosition>>(options.logit_positions);
    }
    auto observation_plan = prepare_observations(model.observation_sites(options.numerical_mode), length,
                                                 options.observations, options.observation_limits);
    auto recipe = model.full_forward(length, options.numerical_mode, positions);
    models::validate_recipe(model, recipe, length, options.numerical_mode, positions);
    auto plan =
        std::make_unique<FullForwardPlan::Impl>(checkpoint, length, options.numerical_mode, std::move(recipe),
                                                std::move(observation_plan), std::move(positions));
    plan->logit_bytes = logit_bytes;
    const auto captures = plan->observations.entries();
    plan->capture_slots.resize(captures.size());
    std::vector<std::vector<kernels::Operation>> copies(plan->recipe.tensors.size());
    for (std::size_t c = 0; c < captures.size(); ++c) {
        const auto &capture = dolittle::at(captures, c);
        if (capture.positions.empty()) {
            continue;
        }
        const auto found =
            std::find_if(plan->recipe.observations.begin(), plan->recipe.observations.end(),
                         [&](const auto &value) { return same_point(value.site.point, capture.site.point); });
        if (found == plan->recipe.observations.end()) {
            throw std::logic_error("unresolved observation");
        }
        const kernels::TensorId target{plan->recipe.tensors.size()};
        plan->capture_slots.at(c) = target;
        plan->recipe.tensors.push_back(tensors::contiguous_layout(
            capture.site.storage_type, std::array{capture.positions.size(), capture.site.vector_width}));
        std::vector<std::size_t> rows;
        rows.reserve(capture.positions.size());
        for (auto position : capture.positions) {
            rows.push_back(position.value);
        }
        copies.at(found->tensor.value)
            .emplace_back(kernels::GatherRows{found->tensor, target, std::move(rows)});
    }
    // Resolve every operation, including captures, before allocating tensor storage.
    for (const auto &operation : plan->recipe.operations) {
        auto prepared = kernels::prepare(device, operation, plan->recipe.tensors, plan->mode);
        std::vector<kernels::TensorId> produced;
        for (const auto &use : prepared.operands()) {
            if (use.access == kernels::OperandAccess::write) {
                produced.push_back(use.tensor);
            }
        }
        plan->steps.push_back({std::move(prepared), {}});
        for (auto output : produced) {
            for (const auto &copy : copies.at(output.value)) {
                plan->steps.push_back({kernels::prepare(device, copy, plan->recipe.tensors, plan->mode), {}});
            }
        }
    }
    plan->slots.resize(plan->recipe.tensors.size());
    std::vector<std::size_t> first(plan->slots.size(), std::numeric_limits<std::size_t>::max());
    std::vector<std::size_t> last(plan->slots.size(), 0);
    for (std::size_t i = 0; i < plan->slots.size(); ++i) {
        plan->slots.at(i).bytes = tensors::required_bytes(plan->recipe.tensors.at(i));
        plan->slots.at(i).alignment = tensors::element_bytes(plan->recipe.tensors.at(i).element_type);
    }
    std::size_t workspace_bytes = 0, workspace_alignment = 1;
    for (std::size_t s = 0; s < plan->steps.size(); ++s) {
        const auto &op = plan->steps.at(s).operation;
        for (const auto &use : op.operands()) {
            auto &slot = plan->slots.at(use.tensor.value);
            slot.alignment = std::max(slot.alignment, use.alignment);
            first.at(use.tensor.value) = std::min(first.at(use.tensor.value), s);
            last.at(use.tensor.value) = s;
        }
        workspace_bytes = std::max(workspace_bytes, op.workspace().bytes);
        workspace_alignment = std::max(workspace_alignment, op.workspace().alignment);
    }
    std::size_t weight_bytes = 0, weight_alignment = 1;
    for (const auto &input : plan->recipe.parameters) {
        auto &slot = plan->slots.at(input.tensor.value);
        slot.arena = Arena::weight;
        const auto view = checkpoint.parameter(input.parameter);
        tensors::validate_view(view);
        const auto &expected = plan->recipe.tensors.at(input.tensor.value);
        if (view.layout->shape != expected.shape || view.layout->element_type != expected.element_type ||
            view.bytes.size() < slot.bytes) {
            throw std::invalid_argument("checkpoint parameter differs from recipe");
        }
        for (std::size_t axis = 0; axis < expected.shape.size(); ++axis) {
            if (expected.shape.at(axis) > 1 &&
                view.layout->byte_strides.at(axis) != expected.byte_strides.at(axis)) {
                throw std::invalid_argument("checkpoint parameter strides differ from recipe");
            }
        }
        // Empty logit selections need no final projection residency.
        if (first.at(input.tensor.value) == std::numeric_limits<std::size_t>::max()) {
            continue;
        }
        weight_bytes = align_up(weight_bytes, slot.alignment);
        slot.offset = weight_bytes;
        weight_bytes = add(weight_bytes, slot.bytes);
        weight_alignment = std::max(weight_alignment, slot.alignment);
    }
    std::size_t output_alignment = 1;
    const auto retain = [&](kernels::TensorId id) {
        auto &slot = plan->slots.at(id.value);
        slot.arena = Arena::output;
        slot.offset = align_up(plan->output_bytes, slot.alignment);
        plan->output_bytes = add(slot.offset, slot.bytes);
        output_alignment = std::max(output_alignment, slot.alignment);
    };
    if (plan->recipe.logits) {
        retain(*plan->recipe.logits);
    }
    for (const auto id : plan->capture_slots) {
        if (id) {
            retain(*id);
        }
    }
    first.at(plan->recipe.token_input.value) = 0;
    std::vector<ResourceLifetime> resources;
    for (std::size_t i = 0; i < plan->slots.size(); ++i) {
        const auto &slot = plan->slots.at(i);
        if (slot.arena == Arena::scratch) {
            if (first.at(i) == std::numeric_limits<std::size_t>::max()) {
                throw std::invalid_argument("scratch slot has no use or producer");
            }
            resources.push_back({{i}, slot.bytes, slot.alignment, first.at(i), last.at(i)});
        }
    }
    const auto fixed = add(add(weight_bytes, plan->output_bytes), workspace_bytes);
    if (fixed > options.max_device_bytes) {
        throw std::length_error("forward fixed arenas exceed budget");
    }
    const auto storage = plan_storage(resources, options.max_device_bytes - fixed);
    plan->peak = add(fixed, storage.arena_bytes);
    for (const auto &placement : storage.placements) {
        plan->slots.at(placement.id.value).offset = placement.offset;
    }
    const auto token_bytes = multiply(length.value, sizeof(std::uint32_t));
    plan->host_bytes = add(add(logit_bytes, plan->observations.payload_bytes()), token_bytes);
    plan->host_bytes =
        add(plan->host_bytes, multiply(plan->logit_positions.size(), sizeof(models::TokenPosition)));
    for (const auto &capture : captures) {
        plan->host_bytes =
            add(plan->host_bytes, multiply(capture.positions.size(), sizeof(models::TokenPosition)));
        plan->host_bytes = add(plan->host_bytes, sizeof(ResultCapture) + sizeof(CapturedRows));
    }
    plan->host_bytes = add(plan->host_bytes, checkpoint.identity().size());
    plan->queue.emplace(device.create_queue());
    if (weight_bytes) {
        plan->weights.emplace(device.allocate({weight_bytes, weight_alignment}));
    }
    if (storage.arena_bytes) {
        plan->scratch.emplace(device.allocate({storage.arena_bytes, storage.arena_alignment}));
    }
    if (plan->output_bytes) {
        plan->output.emplace(device.allocate({plan->output_bytes, output_alignment}));
        plan->readback.emplace(device.allocate_staging({plan->output_bytes, output_alignment}));
    }
    if (workspace_bytes) {
        plan->workspace.emplace(device.allocate({workspace_bytes, workspace_alignment}));
    }
    // Bounded weight staging is reused serially during setup; token staging may
    // require more bytes for an explicitly admitted long sequence.
    const auto staging_bytes = std::max(token_bytes, std::min(weight_bytes, std::size_t{16} * 1024 * 1024));
    plan->staging.emplace(device.allocate_staging({staging_bytes, alignof(std::uint32_t)}));
    for (const auto &input : plan->recipe.parameters) {
        if (first.at(input.tensor.value) == std::numeric_limits<std::size_t>::max()) {
            continue;
        }
        const auto view = checkpoint.parameter(input.parameter);
        const auto &slot = plan->slots.at(input.tensor.value);
        for (std::size_t offset = 0; offset < slot.bytes;) {
            const auto bytes = std::min(staging_bytes, slot.bytes - offset);
            std::memcpy(present(plan->staging).bytes().data(), view.bytes.subspan(offset).data(), bytes);
            present(plan->queue)
                .upload({&present(plan->weights), slot.offset + offset, bytes},
                        {&present(plan->staging), 0, bytes});
            present(plan->queue).wait(present(plan->queue).mark());
            offset += bytes;
        }
    }
    for (auto &step : plan->steps) {
        for (const auto &use : step.operation.operands()) {
            const auto &slot = plan->slots.at(use.tensor.value);
            auto *owner = plan->buffer(slot.arena);
            if (use.access == kernels::OperandAccess::read) {
                step.bindings.push_back(
                    {use.tensor, static_cast<const gpu::DeviceBuffer *>(owner), slot.offset, slot.bytes});
            } else {
                step.bindings.push_back({use.tensor, owner, slot.offset, slot.bytes});
            }
        }
    }
    return FullForwardPlan(std::move(plan));
}

ForwardResult FullForwardPlan::execute(std::span<const std::uint32_t> token_ids, ExecutionId id) {
    auto &plan = *impl_;
    if (plan.failed) {
        throw std::runtime_error("forward plan unusable after execution failure");
    }
    if (token_ids.size() != plan.length.value) {
        throw std::invalid_argument("wrong token count");
    }
    for (const auto token : token_ids) {
        if (token >= plan.checkpoint->model().vocabulary_size()) {
            throw std::invalid_argument("token outside vocabulary");
        }
    }
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
    for (const auto &capture : captures) {
        result->capture_metadata.push_back({capture.site,
                                            {capture.positions.begin(), capture.positions.end()},
                                            capture.byte_offset,
                                            capture.byte_count});
    }
    for (const auto &capture : result->capture_metadata) {
        result->capture_views.push_back(
            {capture.site, capture.positions,
             std::span<const std::byte>(result->capture_bytes).subspan(capture.offset, capture.bytes)});
    }
    try {
        const auto &tokens = plan.slots.at(plan.recipe.token_input.value);
        std::memcpy(present(plan.staging).bytes().data(), token_ids.data(), tokens.bytes);
        present(plan.queue)
            .upload({plan.buffer(tokens.arena), tokens.offset, tokens.bytes},
                    {&present(plan.staging), 0, tokens.bytes});
        for (const auto &step : plan.steps) {
            const auto bytes = step.operation.workspace().bytes;
            step.operation.submit(present(plan.queue), step.bindings,
                                  {bytes ? &present(plan.workspace) : nullptr, 0, bytes});
        }
        if (plan.output_bytes) {
            present(plan.queue)
                .download({&present(plan.readback), 0, plan.output_bytes},
                          {&present(plan.output), 0, plan.output_bytes});
        }
        present(plan.queue).wait(present(plan.queue).mark());
        if (plan.recipe.logits) {
            const auto &slot = plan.slots.at(plan.recipe.logits->value);
            std::memcpy(result->logits.data(), present(plan.readback).bytes().subspan(slot.offset).data(),
                        slot.bytes);
        }
        for (std::size_t c = 0; c < captures.size(); ++c) {
            if (!plan.capture_slots.at(c)) {
                continue;
            }
            const auto &slot = plan.slots.at(present(plan.capture_slots.at(c)).value);
            std::memcpy(
                std::span(result->capture_bytes).subspan(dolittle::at(captures, c).byte_offset).data(),
                present(plan.readback).bytes().subspan(slot.offset).data(), slot.bytes);
        }
    } catch (...) {
        plan.failed = true;
        try {
            present(plan.queue).drain();
        } catch (...) { // NOLINT(bugprone-empty-catch)
            // Preserve the active exception; buffer owners perform a final no-throw drain.
        }
        throw;
    }
    return ForwardResult(std::move(result));
}

struct CachedForwardPlan::Impl {
    struct Capture {
        models::ObservationSite site;
        kernels::TensorId source;
        std::size_t offset, row_bytes;
    };
    const models::BoundCheckpoint *checkpoint;
    CachedForwardOptions options;
    models::IncrementalRecipe recipe;
    std::vector<Slot> slots;
    std::vector<Step> steps;
    std::vector<Capture> captures;
    std::vector<models::ObservationSite> sites;
    std::vector<std::vector<std::size_t>> copies_after;
    std::vector<std::optional<std::size_t>> cache_at;
    std::vector<std::uint32_t> history;
    std::size_t committed = 0, peak = 0, output_bytes = 0, logit_bytes = 0;
    bool failed = false, finished = false;
    std::optional<gpu::Queue> queue;
    std::optional<gpu::DeviceBuffer> weights, scratch, output;
    std::optional<gpu::HostBuffer> upload, readback;
    std::unique_ptr<cache::Contiguous> cache;
    Impl(const models::BoundCheckpoint &c, CachedForwardOptions o, models::IncrementalRecipe r)
        : checkpoint(&c), options(std::move(o)), recipe(std::move(r)) {}
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;
    Impl(Impl &&) = delete;
    Impl &operator=(Impl &&) = delete;
    ~Impl() {
        if (queue) {
            try {
                present(queue).drain();

            } catch (...) { // NOLINT(bugprone-empty-catch)
                // Preserve the active exception; buffer owners perform a final no-throw drain.
            }
        }
    }
    gpu::DeviceBuffer *buffer(Arena arena) {
        switch (arena) {
        case Arena::weight:
            return &present(weights);
        case Arena::scratch:
            return &present(scratch);
        case Arena::output:
            return &present(output);
        }
        throw std::logic_error("unknown cached arena");
    }
    gpu::ConstDeviceRegion source(kernels::TensorId id, std::size_t bytes) {
        const auto &slot = slots.at(id.value);
        return {buffer(slot.arena), slot.offset, bytes};
    }
};
CachedForwardPlan::CachedForwardPlan(std::unique_ptr<Impl> p) : impl_(std::move(p)) {}
CachedForwardPlan::~CachedForwardPlan() = default;
CachedForwardPlan::CachedForwardPlan(CachedForwardPlan &&) noexcept = default;
CachedForwardPlan &CachedForwardPlan::operator=(CachedForwardPlan &&) noexcept = default;
std::string_view CachedForwardPlan::checkpoint_identity() const noexcept {
    return impl_->checkpoint->identity();
}
cache::SequenceId CachedForwardPlan::sequence_id() const noexcept { return impl_->options.sequence; }
tensors::ComputationMode CachedForwardPlan::numerical_mode() const noexcept {
    return impl_->options.numerical_mode;
}
cache::Prefix CachedForwardPlan::prefix() const noexcept {
    return {impl_->committed, !impl_->failed && !impl_->finished};
}
std::size_t CachedForwardPlan::capacity() const noexcept { return impl_->options.capacity; }
std::size_t CachedForwardPlan::max_prefill_chunk() const noexcept { return impl_->options.max_prefill_chunk; }
std::size_t CachedForwardPlan::estimated_peak_device_bytes() const noexcept { return impl_->peak; }
std::size_t CachedForwardPlan::vocabulary_size() const noexcept {
    return impl_->checkpoint->model().vocabulary_size();
}
std::span<const models::ObservationSite> CachedForwardPlan::capture_sites() const noexcept {
    return impl_->sites;
}
void CachedForwardPlan::finish() {
    if (impl_->failed) {
        throw std::runtime_error("failed cached execution plan");
    }
    impl_->finished = true;
}
CachedForwardPlan prepare_cached_forward(gpu::Device &device, const models::BoundCheckpoint &checkpoint,
                                         const CachedForwardOptions &options) {
    const auto &model = checkpoint.model();
    const auto count = options.max_prefill_chunk;
    if (!count || !options.capacity || count > options.capacity ||
        options.capacity > model.context_limit().value) {
        throw std::invalid_argument("invalid cached capacity/chunk");
    }
    auto recipe = model.incremental_forward({count}, options.numerical_mode);
    models::validate_incremental_recipe(model, recipe, {count}, options.numerical_mode);
    std::vector<ObservationRequest> requests;
    requests.reserve(options.capture_sites.size());
    for (auto point : options.capture_sites) {
        requests.push_back({point, AllPositions{}});
    }
    const auto observations = prepare_observations(model.observation_sites(options.numerical_mode),
                                                   {options.capacity}, requests, options.observation_limits);
    auto p = std::make_unique<CachedForwardPlan::Impl>(checkpoint, options, std::move(recipe));
    auto &forward = p->recipe.forward;
    cache::Specification spec{
        std::string(checkpoint.identity()), options.sequence, options.numerical_mode, options.capacity, {}};
    p->cache_at.resize(forward.operations.size());
    for (const auto &append : p->recipe.cache_appends) {
        const auto &shape = forward.tensors.at(append.rotated_key.value).shape;
        spec.layers.push_back({shape.at(1), shape.at(2)});
        p->cache_at.at(append.attention_step) = append.block.value;
    }
    const auto cache_bytes = cache::estimate_contiguous(spec).allocation_bytes;
    p->slots.resize(forward.tensors.size());
    p->copies_after.resize(forward.operations.size());
    std::vector<std::size_t> first(p->slots.size(), std::numeric_limits<std::size_t>::max()),
        last(p->slots.size(), 0);
    for (std::size_t i = 0; i < p->slots.size(); ++i) {
        p->slots.at(i).bytes = tensors::required_bytes(forward.tensors.at(i));
        p->slots.at(i).alignment = tensors::element_bytes(forward.tensors.at(i).element_type);
    }
    for (std::size_t i = 0; i < forward.operations.size(); ++i) {
        kernels::RowPolicy rows;
        if (i <= p->recipe.logits_begin) {
            rows = {count, i == p->recipe.logits_begin};
        }
        auto operation =
            kernels::prepare(device, forward.operations.at(i), forward.tensors, options.numerical_mode, rows);
        if (operation.workspace().bytes) {
            throw std::invalid_argument("cached plan requires zero-workspace operation kind");
        }
        for (const auto &use : operation.operands()) {
            auto &slot = p->slots.at(use.tensor.value);
            slot.alignment = std::max(slot.alignment, use.alignment);
            first.at(use.tensor.value) = std::min(first.at(use.tensor.value), i);
            last.at(use.tensor.value) = i;
        }
        p->steps.push_back({std::move(operation), {}});
    }
    // Retain one vocabulary row; captures are chunk-sized, not capacity-sized.
    const auto logits = present(forward.logits);
    auto &logit_slot = p->slots.at(logits.value);
    logit_slot.arena = Arena::output;
    logit_slot.offset = 0;
    p->logit_bytes = logit_slot.bytes;
    p->output_bytes = p->logit_bytes;
    for (const auto &observation : observations.entries()) {
        const auto it =
            std::find_if(forward.observations.begin(), forward.observations.end(), [&](const auto &value) {
                return same_point(value.site.point, observation.site.point);
            });
        if (it == forward.observations.end()) {
            throw std::invalid_argument("unresolved cached observation");
        }
        const auto row =
            multiply(observation.site.vector_width, tensors::element_bytes(observation.site.storage_type));
        p->sites.push_back(observation.site);
        p->captures.push_back({observation.site, it->tensor, p->output_bytes, row});
        p->output_bytes = add(p->output_bytes, multiply(count, row));
        bool found = false;
        for (std::size_t i = 0; i < p->steps.size(); ++i) {
            for (const auto &use : p->steps.at(i).operation.operands()) {
                if (use.tensor.value == it->tensor.value && use.access == kernels::OperandAccess::write) {
                    p->copies_after.at(i).push_back(p->captures.size() - 1);
                    last.at(it->tensor.value) = std::max(last.at(it->tensor.value), i);
                    found = true;
                }
            }
        }
        if (!found) {
            throw std::invalid_argument("capture lacks producer");
        }
    }
    std::size_t weight_bytes = 0;
    for (const auto &input : forward.parameters) {
        auto &slot = p->slots.at(input.tensor.value);
        slot.arena = Arena::weight;
        const auto view = checkpoint.parameter(input.parameter);
        tensors::validate_view(view);
        const auto &expected = forward.tensors.at(input.tensor.value);
        if (view.layout->shape != expected.shape || view.layout->element_type != expected.element_type ||
            view.bytes.size() != slot.bytes) {
            throw std::invalid_argument("cached parameter shape/extent mismatch");
        }
        for (std::size_t axis = 0; axis < expected.shape.size(); ++axis) {
            if (expected.shape.at(axis) > 1 &&
                expected.byte_strides.at(axis) != view.layout->byte_strides.at(axis)) {
                throw std::invalid_argument("cached parameter strides mismatch");
            }
        }
        weight_bytes = align_up(weight_bytes, slot.alignment);
        slot.offset = weight_bytes;
        weight_bytes = add(weight_bytes, slot.bytes);
    }
    first.at(forward.token_input.value) = 0;
    std::vector<ResourceLifetime> lifetimes;
    for (std::size_t i = 0; i < p->slots.size(); ++i) {
        const auto &slot = p->slots.at(i);
        if (slot.arena == Arena::scratch) {
            if (first.at(i) == std::numeric_limits<std::size_t>::max()) {
                throw std::invalid_argument("unused cached scratch");
            }
            lifetimes.push_back({{i}, slot.bytes, slot.alignment, first.at(i), last.at(i)});
        }
    }
    const auto fixed = add(add(weight_bytes, cache_bytes), p->output_bytes);
    if (fixed > options.max_device_bytes) {
        throw std::length_error("cached fixed arenas exceed device budget");
    }
    const auto storage = plan_storage(lifetimes, options.max_device_bytes - fixed);
    p->peak = add(fixed, storage.arena_bytes);
    for (const auto &placement : storage.placements) {
        p->slots.at(placement.id.value).offset = placement.offset;
    }
    const auto upload_bytes = std::max(multiply(count, sizeof(std::uint32_t)),
                                       std::min(weight_bytes, std::size_t{16} * 1024 * 1024));
    auto host_bytes = add(upload_bytes, multiply(2, p->output_bytes));
    host_bytes = add(host_bytes, observations.payload_bytes());
    host_bytes = add(host_bytes, multiply(options.capacity, 2 * sizeof(std::uint32_t)));
    host_bytes = add(host_bytes, multiply(count, sizeof(std::uint32_t)));
    host_bytes =
        add(host_bytes, multiply(multiply(count, p->captures.size()), sizeof(models::TokenPosition)));
    host_bytes = add(host_bytes,
                     multiply(multiply(options.capacity, p->captures.size()), sizeof(models::TokenPosition)));
    host_bytes = add(host_bytes, multiply(model.vocabulary_size(),
                                          sizeof(float) + sizeof(double) + 2 * sizeof(std::uint32_t)));
    if (host_bytes > options.max_host_bytes) {
        throw std::length_error("cached host outputs exceed budget");
    }
    p->history.reserve(options.capacity);
    p->queue.emplace(device.create_queue());
    p->weights.emplace(device.allocate({weight_bytes, 4}));
    p->scratch.emplace(device.allocate({storage.arena_bytes, storage.arena_alignment}));
    p->output.emplace(device.allocate({p->output_bytes, 4}));
    p->upload.emplace(device.allocate_staging({upload_bytes, 4}));
    p->readback.emplace(device.allocate_staging({p->output_bytes, 4}));
    p->cache = cache::allocate_contiguous(device, present(p->queue), spec, cache_bytes);
    for (const auto &input : forward.parameters) {
        const auto view = checkpoint.parameter(input.parameter);
        const auto &slot = p->slots.at(input.tensor.value);
        for (std::size_t offset = 0; offset < slot.bytes;) {
            const auto bytes = std::min(upload_bytes, slot.bytes - offset);
            std::memcpy(present(p->upload).bytes().data(), view.bytes.subspan(offset).data(), bytes);
            present(p->queue).upload({&present(p->weights), slot.offset + offset, bytes},
                                     {&present(p->upload), 0, bytes});
            present(p->queue).wait(present(p->queue).mark());
            offset += bytes;
        }
    }
    for (auto &step : p->steps) {
        for (const auto &use : step.operation.operands()) {
            const auto &slot = p->slots.at(use.tensor.value);
            auto *buffer = p->buffer(slot.arena);
            if (use.access == kernels::OperandAccess::read) {
                step.bindings.push_back(
                    {use.tensor, static_cast<const gpu::DeviceBuffer *>(buffer), slot.offset, slot.bytes});
            } else {
                step.bindings.push_back({use.tensor, buffer, slot.offset, slot.bytes});
            }
        }
    }
    return CachedForwardPlan(std::move(p));
}
ConsumedChunk CachedForwardPlan::consume(std::span<const std::uint32_t> ids, bool need_logits) {
    auto &p = *impl_;
    if (p.failed || p.finished) {
        throw std::runtime_error("cached plan is no longer usable");
    }
    if (ids.empty() || ids.size() > p.options.max_prefill_chunk ||
        ids.size() > p.options.capacity - p.committed) {
        throw std::invalid_argument("cached token chunk outside admitted bounds");
    }
    for (auto id : ids) {
        if (id >= vocabulary_size()) {
            throw std::invalid_argument("token outside vocabulary");
        }
    }
    ConsumedChunk result;
    result.first_position = {p.committed};
    result.tokens.assign(ids.begin(), ids.end());
    if (need_logits) {
        result.next_logits = OwnedLogitRow{{p.committed + ids.size() - 1},
                                           vocabulary_size(),
                                           p.options.numerical_mode.activation_storage,
                                           std::vector<std::byte>(p.logit_bytes)};
    }
    result.captures.reserve(p.captures.size());
    for (const auto &capture : p.captures) {
        OwnedCapture out{capture.site, {}, std::vector<std::byte>(multiply(ids.size(), capture.row_bytes))};
        out.positions.reserve(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            out.positions.push_back({p.committed + i});
        }
        result.captures.push_back(std::move(out));
    }
    const kernels::ExecutionExtent extent{p.committed, ids.size()};
    try {
        const auto &tokens = p.slots.at(p.recipe.forward.token_input.value);
        const auto bytes = ids.size_bytes();
        std::memcpy(present(p.upload).bytes().data(), ids.data(), bytes);
        present(p.queue).upload({&present(p.scratch), tokens.offset, bytes}, {&present(p.upload), 0, bytes});
        for (std::size_t i = 0; i < p.steps.size(); ++i) {
            if (!need_logits && i == p.recipe.logits_begin) {
                break;
            }
            std::optional<cache::ContiguousLayerView> view;
            if (p.cache_at.at(i)) {
                const auto block = present(p.cache_at.at(i));
                const auto &append = p.recipe.cache_appends.at(block);
                view = p.cache->layer(block);
                const auto bytes_per_row =
                    multiply(multiply(view->shape.kv_heads, view->shape.head_width),
                             tensors::element_bytes(p.options.numerical_mode.activation_storage));
                p.cache->append(present(p.queue), block, p.committed, ids.size(),
                                p.source(append.rotated_key, multiply(ids.size(), bytes_per_row)),
                                p.source(append.value, multiply(ids.size(), bytes_per_row)));
            }
            auto &step = p.steps.at(i);
            step.operation.submit(present(p.queue), step.bindings, {},
                                  i <= p.recipe.logits_begin ? &extent : nullptr, view ? &*view : nullptr);
            for (auto c : p.copies_after.at(i)) {
                const auto &capture = p.captures.at(c);
                const auto bytes = multiply(ids.size(), capture.row_bytes);
                present(p.queue).copy({&present(p.output), capture.offset, bytes},
                                      p.source(capture.source, bytes));
            }
        }
        if (need_logits) {
            present(p.queue).download({&present(p.readback), 0, p.logit_bytes},
                                      {&present(p.output), 0, p.logit_bytes});
        }
        for (const auto &capture : p.captures) {
            const auto bytes = multiply(ids.size(), capture.row_bytes);
            present(p.queue).download({&present(p.readback), capture.offset, bytes},
                                      {&present(p.output), capture.offset, bytes});
        }
        present(p.queue).wait(present(p.queue).mark());
        if (need_logits) {
            std::memcpy(result.next_logits->bytes.data(), present(p.readback).bytes().data(), p.logit_bytes);
        }
        for (std::size_t c = 0; c < p.captures.size(); ++c) {
            std::memcpy(result.captures.at(c).bytes.data(),
                        present(p.readback).bytes().subspan(p.captures.at(c).offset).data(),
                        result.captures.at(c).bytes.size());
        }
        p.history.insert(p.history.end(), ids.begin(), ids.end());
        p.committed += ids.size();
    } catch (...) {
        p.failed = true;
        try {
            present(p.queue).drain();
        } catch (...) { // NOLINT(bugprone-empty-catch)
            // Preserve the active exception; buffer owners perform a final no-throw drain.
        }
        throw;
    }
    return result;
}

} // namespace dolittle::inference
