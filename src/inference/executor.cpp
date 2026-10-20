/*
 * Axis X1: prepare model recipes, bind storage and execute full or cached
 passes.
 * Contract: verified BoundCheckpoint and Device outlive borrowed plans. A
 * ResidentModel owns a checkpoint, uploaded weights and ordered queue; at most
 * one cached plan leases it. Destroy the plan before its resident model and
 * retain Device until all exported capture leases die. Failed plans are
 poisoned
 * and drained; they cannot resume. Allocation, shape/provenance/budget checks
 * happen before live launch. Model-provided observation sites determine hooks.
 *
 * Algorithm: transformer-execution.md full model/block flow and
 * execution-boundary.md sections 3-6.
 * 1. Validate the recipe and each operand. Assign immutable weights, contiguous
 *    per-layer K/V, persistent outputs and transient scratch. Extend source
 *    lifetimes through capture copies. Bind independent attention-scale slots;
 *    initialize 1/sqrt(head_width) on device before their first reader.
 * 2. For live generation, admit P prompt and G continuation positions, upload
 *    all prompt IDs and allocate independent retained device capture storage.
 *    Build fixed full-chunk, remainder, first-selection and decode graph
 bodies.
 *    Device position selects IDs, RoPE position, KV write/read windows and the
 *    destination row for every requested hook.
 * 3. Each body embeds IDs and runs the model's ordered block recipe: RMSNorm,
 *    Q/K/V projections, Q/K head norms, RoPE, causal grouped-query attention,
 *    output projection/residual, post-attention norm, gated SiLU MLP/residual.
 *    Capture raw advertised values after their producer and before scratch
 *    reuse. The head gathers the last row, normalizes and projects vocabulary
 *    logits; device sampling feeds the next embedding without readback.
 * 4. An AV override leases a completed source row and declares type/scale.
 *    DeviceInjection validates and normalizes it inside the root graph. Status
 *    gates all inference bodies. Copy its output over the marker embedding
 *    after lookup and before block 0, only in the containing prompt chunk.
 * 5. Start once; collection updates host metadata from immutable events. The
 *    GPU owns all progress decisions. finish drains and seals the plan. Capture
 *    leases outlive transient storage and target-to-AV weight replacement.
 *
 * FullForwardPlan is an explicit diagnostic path: execute an exact fixed token
 * trace with optional host injection, capture selected rows and batch readbacks
 * at one completion point. Cached consume/consume_view similarly replay known
 * IDs for oracle comparison; borrowed views expire at the next consume. These
 * APIs reject use after live preparation. Host injection cannot enter a live
 * sequence; device injection cannot silently enter a host diagnostic pass.
 * Tensor storage/rounding is model-selected BF16/FP32 with FP32 accumulation;
 * this executor does not alter arithmetic to match a reference.
 *
 * Resident upload algorithm (checkpoint loading and binding): pack parameter
 * bytes into aligned device offsets, then split the admitted host staging
 * allowance into two disjoint slots. For each parameter chunk, wait for that
 * slot's previous upload, copy source bytes into it, enqueue the H2D copy,
 * record its completion point, and advance to the other slot. CPU staging can
 * overlap the previous slot's DMA on the same ordered queue. Drain the final
 * uploads before returning the resident model. Staging and weights remain
 * owned until completion, including exception cleanup through their RAII
 * handles. No tensor conversion or arithmetic occurs in this byte pipeline.
 * Cost: total staging <= min(weight arena bytes, host allowance, 2 MiB),
 * two completion slots, one CPU copy and one H2D copy per parameter byte.
 * A one-byte allowance uses one slot; odd allowances leave at most one byte
 * unused. Optimization: wait on slot reuse, not immediately after submission;
 * allocate staging and completion resources once before the upload loop.
 * Upload design/implementation consultation, 2026-10-20 (both phases queried
 * both servers and retrieved the rules): cpp-guidelines R.1 (#rr-raii)
 * retains in-flight resources on failure; I.7 (#ri-post) makes ready weights
 * a return postcondition. cpp-perf-guidelines GPU.7 requires multi-buffered
 * upload storage, per-resource completion and measured overlap. Source:
 * guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md.
 *
 * Design/cost: one prepared graph O(recipe size), not O(sequence length).
 * Weights O(parameter bytes); KV O(layers*capacity*kv_heads*head_width*2*b);
 * capture payload O((P+G)*sum(row_bytes)); scratch O(chunk*model widths).
 * The immutable journal is O(P+G) small metadata. Separate budgets cover host
 * staging/diagnostic buffers and device state, constants, sampler and captures.
 * Optimization: direct device-positioned KV writes avoid append copies;
 * graph execution and GPU NLA conversion allocate nothing or read tensors back.
 * CPU-only oracle replay is explicit and never chosen as a live fallback.
 *
 * Corpus consultation, header design and implementation, 2026-10-20: both local
 * servers queried separately in each phase and the cited rules read in full.
 * cpp-guidelines I.1 (#ri-explicit), R.1 (#rr-raii): explicit residency,
 * completion, ownership and failure contracts. cpp-perf-guidelines GPU.1,
 * GPU.6, GPU.7, GPU.9: keep tensor intermediates/decisions on device, batch
 * dispatch, decouple observation and separate persistent from transient arenas.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Related GPU.6/7/9 entries describe dispatch, staging and storage.
 */
#include "inference/executor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "bounds.h"
#include "gpu/device.h"
#include "inference/cache/contiguous.h"
#include "inference/decoding/device.h"
#include "inference/injection/device.h"
#include "inference/storage_plan.h"
#include "models/checkpoint.h"
#include "tensors/representations/bf16.h"

namespace dolittle::inference {
namespace {
template <class Optional>
decltype(auto) present(Optional& value) {
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
  std::size_t workspace_offset = 0;
};
void validate_injection(const models::ForwardRecipe&, const EmbeddingOverride&,
                        std::size_t, tensors::ComputationMode);
struct ResultCapture {
  models::ObservationSite site;
  std::vector<models::TokenPosition> positions;
  std::size_t offset, bytes;
};
}  // namespace

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
ForwardResult::ForwardResult(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ForwardResult::~ForwardResult() = default;
ForwardResult::ForwardResult(ForwardResult&&) noexcept = default;
ForwardResult& ForwardResult::operator=(ForwardResult&&) noexcept = default;
ExecutionId ForwardResult::execution_id() const noexcept { return impl_->id; }
std::string_view ForwardResult::checkpoint_identity() const noexcept {
  return impl_->identity;
}
std::span<const std::uint32_t> ForwardResult::token_ids() const noexcept {
  return impl_->tokens;
}
tensors::ComputationMode ForwardResult::numerical_mode() const noexcept {
  return impl_->mode;
}
LogitRows ForwardResult::logits() const noexcept {
  return {impl_->logit_positions, impl_->vocabulary,
          impl_->mode.activation_storage, impl_->logits};
}
std::span<const CapturedRows> ForwardResult::captures() const noexcept {
  return impl_->capture_views;
}

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
  std::size_t injection_staging_offset = 0;
  std::optional<std::size_t> injection_step;

  Impl(const models::BoundCheckpoint& checkpoint_in,
       models::SequenceLength length_in, tensors::ComputationMode mode_in,
       models::ForwardRecipe recipe_in, ObservationPlan observations_in,
       std::vector<models::TokenPosition> positions)
      : checkpoint(&checkpoint_in),
        length(length_in),
        mode(mode_in),
        recipe(std::move(recipe_in)),
        observations(std::move(observations_in)),
        logit_positions(std::move(positions)) {}
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    if (queue) {
      try {
        present(queue).drain();
      } catch (...) {  // NOLINT(bugprone-empty-catch)
        // Preserve the active exception; buffer owners perform a final
        // no-throw drain.
      }
    }
  }
  gpu::DeviceBuffer* buffer(Arena arena) {
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
FullForwardPlan::FullForwardPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
FullForwardPlan::~FullForwardPlan() = default;
FullForwardPlan::FullForwardPlan(FullForwardPlan&&) noexcept = default;
FullForwardPlan& FullForwardPlan::operator=(FullForwardPlan&&) noexcept =
    default;
models::SequenceLength FullForwardPlan::sequence_length() const noexcept {
  return impl_->length;
}
std::size_t FullForwardPlan::estimated_peak_device_bytes() const noexcept {
  return impl_->peak;
}
std::size_t FullForwardPlan::host_output_bytes() const noexcept {
  return impl_->host_bytes;
}

FullForwardPlan prepare_full_forward(gpu::Device& device,
                                     const models::BoundCheckpoint& checkpoint,
                                     models::SequenceLength length,
                                     const ForwardOptions& options) {
  const auto& model = checkpoint.model();
  if (!length.value || length.value > model.context_limit().value) {
    throw std::invalid_argument("forward length outside context");
  }
  std::vector<models::TokenPosition> positions;
  if (options.logit_positions.valueless_by_exception()) {
    throw std::invalid_argument("missing logit selection");
  }
  const auto all =
      std::holds_alternative<AllPositions>(options.logit_positions);
  const auto count = all ? length.value
                         : std::get<std::vector<models::TokenPosition>>(
                               options.logit_positions)
                               .size();
  const auto logit_bytes = multiply(
      multiply(count, model.vocabulary_size()),
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
    positions =
        std::get<std::vector<models::TokenPosition>>(options.logit_positions);
  }
  auto observation_plan = prepare_observations(
      model.observation_sites(options.numerical_mode), length,
      options.observations, options.observation_limits);
  auto recipe = model.full_forward(length, options.numerical_mode, positions);
  models::validate_recipe(model, recipe, length, options.numerical_mode,
                          positions);
  auto plan = std::make_unique<FullForwardPlan::Impl>(
      checkpoint, length, options.numerical_mode, std::move(recipe),
      std::move(observation_plan), std::move(positions));
  plan->logit_bytes = logit_bytes;
  const auto captures = plan->observations.entries();
  plan->capture_slots.resize(captures.size());
  std::vector<std::vector<kernels::Operation>> copies(
      plan->recipe.tensors.size());
  for (std::size_t c = 0; c < captures.size(); ++c) {
    const auto& capture = dolittle::at(captures, c);
    if (capture.positions.empty()) {
      continue;
    }
    const auto found =
        std::find_if(plan->recipe.observations.begin(),
                     plan->recipe.observations.end(), [&](const auto& value) {
                       return same_point(value.site.point, capture.site.point);
                     });
    if (found == plan->recipe.observations.end()) {
      throw std::logic_error("unresolved observation");
    }
    const kernels::TensorId target{plan->recipe.tensors.size()};
    plan->capture_slots.at(c) = target;
    plan->recipe.tensors.push_back(tensors::contiguous_layout(
        capture.site.storage_type,
        std::array{capture.positions.size(), capture.site.vector_width}));
    std::vector<std::size_t> rows;
    rows.reserve(capture.positions.size());
    for (auto position : capture.positions) {
      rows.push_back(position.value);
    }
    copies.at(found->tensor.value)
        .emplace_back(
            kernels::GatherRows{found->tensor, target, std::move(rows)});
  }
  // Resolve every operation, including captures, before allocating tensor
  // storage.
  for (std::size_t operation_index = 0;
       operation_index < plan->recipe.operations.size(); ++operation_index) {
    const auto& operation = plan->recipe.operations.at(operation_index);
    auto prepared =
        kernels::prepare(device, operation, plan->recipe.tensors, plan->mode);
    std::vector<kernels::TensorId> produced;
    for (const auto& use : prepared.operands()) {
      if (use.access == kernels::OperandAccess::write) {
        produced.push_back(use.tensor);
      }
    }
    if (plan->recipe.embedding_boundary &&
        plan->recipe.embedding_boundary->producer_step == operation_index) {
      plan->injection_step = plan->steps.size();
    }
    plan->steps.push_back({std::move(prepared), {}});
    for (auto output : produced) {
      for (const auto& copy : copies.at(output.value)) {
        plan->steps.push_back(
            {kernels::prepare(device, copy, plan->recipe.tensors, plan->mode),
             {}});
      }
    }
  }
  plan->slots.resize(plan->recipe.tensors.size());
  std::vector<std::size_t> first(plan->slots.size(),
                                 std::numeric_limits<std::size_t>::max());
  std::vector<std::size_t> last(plan->slots.size(), 0);
  for (std::size_t i = 0; i < plan->slots.size(); ++i) {
    plan->slots.at(i).bytes =
        tensors::required_bytes(plan->recipe.tensors.at(i));
    plan->slots.at(i).alignment =
        tensors::element_bytes(plan->recipe.tensors.at(i).element_type);
  }
  std::size_t workspace_bytes = 0, workspace_alignment = 1;
  for (std::size_t s = 0; s < plan->steps.size(); ++s) {
    const auto& op = plan->steps.at(s).operation;
    for (const auto& use : op.operands()) {
      auto& slot = plan->slots.at(use.tensor.value);
      slot.alignment = std::max(slot.alignment, use.alignment);
      first.at(use.tensor.value) = std::min(first.at(use.tensor.value), s);
      last.at(use.tensor.value) = s;
    }
    plan->steps.at(s).workspace_offset = workspace_bytes;
    workspace_bytes = add(workspace_bytes, op.workspace().bytes);
    workspace_alignment =
        std::max(workspace_alignment, op.workspace().alignment);
  }
  std::size_t weight_bytes = 0, weight_alignment = 1;
  for (const auto& input : plan->recipe.parameters) {
    auto& slot = plan->slots.at(input.tensor.value);
    slot.arena = Arena::weight;
    const auto view = checkpoint.parameter(input.parameter);
    tensors::validate_view(view);
    const auto& expected = plan->recipe.tensors.at(input.tensor.value);
    if (view.layout->shape != expected.shape ||
        view.layout->element_type != expected.element_type ||
        view.bytes.size() < slot.bytes) {
      throw std::invalid_argument("checkpoint parameter differs from recipe");
    }
    for (std::size_t axis = 0; axis < expected.shape.size(); ++axis) {
      if (expected.shape.at(axis) > 1 && view.layout->byte_strides.at(axis) !=
                                             expected.byte_strides.at(axis)) {
        throw std::invalid_argument(
            "checkpoint parameter strides differ from recipe");
      }
    }
    // Empty logit selections need no final projection residency.
    if (first.at(input.tensor.value) ==
        std::numeric_limits<std::size_t>::max()) {
      continue;
    }
    weight_bytes = align_up(weight_bytes, slot.alignment);
    slot.offset = weight_bytes;
    weight_bytes = add(weight_bytes, slot.bytes);
    weight_alignment = std::max(weight_alignment, slot.alignment);
  }
  std::size_t output_alignment = 1;
  const auto retain = [&](kernels::TensorId id) {
    auto& slot = plan->slots.at(id.value);
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
    const auto& slot = plan->slots.at(i);
    if (slot.arena == Arena::scratch) {
      if (first.at(i) == std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("scratch slot has no use or producer");
      }
      resources.push_back(
          {{i}, slot.bytes, slot.alignment, first.at(i), last.at(i)});
    }
  }
  const auto fixed =
      add(add(weight_bytes, plan->output_bytes), workspace_bytes);
  if (fixed > options.max_device_bytes) {
    throw std::length_error("forward fixed arenas exceed budget");
  }
  const auto storage =
      plan_storage(resources, options.max_device_bytes - fixed);
  plan->peak = add(fixed, storage.arena_bytes);
  for (const auto& placement : storage.placements) {
    plan->slots.at(placement.id.value).offset = placement.offset;
  }
  const auto token_bytes = multiply(length.value, sizeof(std::uint32_t));
  plan->host_bytes =
      add(add(logit_bytes, plan->observations.payload_bytes()), token_bytes);
  plan->host_bytes = add(
      plan->host_bytes,
      multiply(plan->logit_positions.size(), sizeof(models::TokenPosition)));
  for (const auto& capture : captures) {
    plan->host_bytes =
        add(plan->host_bytes,
            multiply(capture.positions.size(), sizeof(models::TokenPosition)));
    plan->host_bytes =
        add(plan->host_bytes, sizeof(ResultCapture) + sizeof(CapturedRows));
  }
  plan->host_bytes = add(plan->host_bytes, checkpoint.identity().size());
  plan->queue.emplace(device.create_queue());
  if (weight_bytes) {
    plan->weights.emplace(device.allocate({weight_bytes, weight_alignment}));
  }
  if (storage.arena_bytes) {
    plan->scratch.emplace(
        device.allocate({storage.arena_bytes, storage.arena_alignment}));
  }
  if (plan->output_bytes) {
    plan->output.emplace(
        device.allocate({plan->output_bytes, output_alignment}));
    plan->readback.emplace(
        device.allocate_staging({plan->output_bytes, output_alignment}));
  }
  if (workspace_bytes) {
    plan->workspace.emplace(
        device.allocate({workspace_bytes, workspace_alignment}));
    for (const auto& step : plan->steps) {
      if (step.operation.workspace().bytes) {
        step.operation.initialize(present(plan->queue),
                                  {&*plan->workspace, step.workspace_offset,
                                   step.operation.workspace().bytes});
      }
    }
  }
  // Bounded weight staging is reused serially during setup; token staging may
  // require more bytes for an explicitly admitted long sequence.
  const auto staging_bytes = std::max(
      token_bytes, std::min(weight_bytes, std::size_t{16} * 1024 * 1024));
  plan->injection_staging_offset = staging_bytes;
  std::size_t injection_bytes = 0;
  if (plan->recipe.embedding_boundary) {
    const auto& layout =
        plan->recipe.tensors.at(plan->recipe.embedding_boundary->tensor.value);
    injection_bytes = multiply(layout.shape.at(1),
                               tensors::element_bytes(layout.element_type));
  }
  plan->staging.emplace(device.allocate_staging(
      {add(staging_bytes, injection_bytes), alignof(std::uint32_t)}));
  for (const auto& input : plan->recipe.parameters) {
    if (first.at(input.tensor.value) ==
        std::numeric_limits<std::size_t>::max()) {
      continue;
    }
    const auto view = checkpoint.parameter(input.parameter);
    const auto& slot = plan->slots.at(input.tensor.value);
    for (std::size_t offset = 0; offset < slot.bytes;) {
      const auto bytes = std::min(staging_bytes, slot.bytes - offset);
      std::memcpy(present(plan->staging).bytes().data(),
                  view.bytes.subspan(offset).data(), bytes);
      present(plan->queue)
          .upload({&present(plan->weights), slot.offset + offset, bytes},
                  {&present(plan->staging), 0, bytes});
      present(plan->queue).wait(present(plan->queue).mark());
      offset += bytes;
    }
  }
  for (auto& step : plan->steps) {
    for (const auto& use : step.operation.operands()) {
      const auto& slot = plan->slots.at(use.tensor.value);
      auto* owner = plan->buffer(slot.arena);
      if (use.access == kernels::OperandAccess::read) {
        step.bindings.push_back({use.tensor,
                                 static_cast<const gpu::DeviceBuffer*>(owner),
                                 slot.offset, slot.bytes});
      } else {
        step.bindings.push_back({use.tensor, owner, slot.offset, slot.bytes});
      }
    }
  }
  return FullForwardPlan(std::move(plan));
}

ForwardResult FullForwardPlan::execute(std::span<const std::uint32_t> token_ids,
                                       ExecutionId id) {
  return execute_impl(token_ids, id, nullptr);
}
ForwardResult FullForwardPlan::execute(std::span<const std::uint32_t> token_ids,
                                       ExecutionId id,
                                       const EmbeddingOverride& injection) {
  return execute_impl(token_ids, id, &injection);
}
ForwardResult FullForwardPlan::execute_impl(
    std::span<const std::uint32_t> token_ids, ExecutionId id,
    const EmbeddingOverride* injection) {
  auto& plan = *impl_;
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
  if (injection && injection->device) {
    throw std::invalid_argument(
        "diagnostic full forward requires an exported host injection");
  }
  if (injection) {
    validate_injection(plan.recipe, *injection, plan.length.value, plan.mode);
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
  for (const auto& capture : captures) {
    result->capture_metadata.push_back(
        {capture.site,
         {capture.positions.begin(), capture.positions.end()},
         capture.byte_offset,
         capture.byte_count});
  }
  for (const auto& capture : result->capture_metadata) {
    result->capture_views.push_back(
        {capture.site, capture.positions,
         std::span<const std::byte>(result->capture_bytes)
             .subspan(capture.offset, capture.bytes)});
  }
  try {
    const auto& tokens = plan.slots.at(plan.recipe.token_input.value);
    std::memcpy(present(plan.staging).bytes().data(), token_ids.data(),
                tokens.bytes);
    present(plan.queue)
        .upload({plan.buffer(tokens.arena), tokens.offset, tokens.bytes},
                {&present(plan.staging), 0, tokens.bytes});
    for (std::size_t step_index = 0; step_index < plan.steps.size();
         ++step_index) {
      const auto& step = plan.steps.at(step_index);
      const auto bytes = step.operation.workspace().bytes;
      step.operation.submit(present(plan.queue), step.bindings,
                            {bytes ? &present(plan.workspace) : nullptr,
                             step.workspace_offset, bytes});
      if (injection && plan.injection_step &&
          step_index == *plan.injection_step) {
        const auto& slot =
            plan.slots.at(present(plan.recipe.embedding_boundary).tensor.value);
        const auto row_bytes = injection->bytes.size();
        std::memcpy(present(plan.staging)
                        .bytes()
                        .subspan(plan.injection_staging_offset)
                        .data(),
                    injection->bytes.data(), row_bytes);
        present(plan.queue)
            .upload({plan.buffer(slot.arena),
                     slot.offset + injection->position.value * row_bytes,
                     row_bytes},
                    {&present(plan.staging), plan.injection_staging_offset,
                     row_bytes});
      }
    }
    if (plan.output_bytes) {
      present(plan.queue)
          .download({&present(plan.readback), 0, plan.output_bytes},
                    {&present(plan.output), 0, plan.output_bytes});
    }
    present(plan.queue).wait(present(plan.queue).mark());
    if (plan.recipe.logits) {
      const auto& slot = plan.slots.at(plan.recipe.logits->value);
      std::memcpy(result->logits.data(),
                  present(plan.readback).bytes().subspan(slot.offset).data(),
                  slot.bytes);
    }
    for (std::size_t c = 0; c < captures.size(); ++c) {
      if (!plan.capture_slots.at(c)) {
        continue;
      }
      const auto& slot = plan.slots.at(present(plan.capture_slots.at(c)).value);
      std::memcpy(std::span(result->capture_bytes)
                      .subspan(dolittle::at(captures, c).byte_offset)
                      .data(),
                  present(plan.readback).bytes().subspan(slot.offset).data(),
                  slot.bytes);
    }
  } catch (...) {
    plan.failed = true;
    try {
      present(plan.queue).drain();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Preserve the active exception; buffer owners perform a final no-throw
      // drain.
    }
    throw;
  }
  return ForwardResult(std::move(result));
}

namespace {
constexpr std::size_t resident_alignment = 256;
struct ResidentState {
  gpu::Device* device{};
  std::unique_ptr<models::BoundCheckpoint> checkpoint;
  tensors::ComputationMode mode{};
  std::vector<std::size_t> offsets;
  std::size_t bytes{};
  bool leased = false, failed = false;
  std::optional<gpu::Queue> queue;
  std::optional<gpu::DeviceBuffer> weights;
  ResidentState() = default;
  ResidentState(const ResidentState&) = delete;
  ResidentState& operator=(const ResidentState&) = delete;
  ResidentState(ResidentState&&) = delete;
  ResidentState& operator=(ResidentState&&) = delete;
  ~ResidentState() {
    if (queue) {
      try {
        present(queue).drain();
      } catch (...) {  // NOLINT(bugprone-empty-catch)
        // No-throw teardown preserves an active device exception.
      }
    }
  }
};
void validate_injection(const models::ForwardRecipe& recipe,
                        const EmbeddingOverride& row, std::size_t capacity,
                        tensors::ComputationMode mode) {
  if (!recipe.embedding_boundary || row.position.value >= capacity ||
      !row.width || row.storage_type != mode.activation_storage ||
      (!row.device &&
       row.bytes.size() !=
           multiply(row.width, tensors::element_bytes(row.storage_type)))) {
    throw std::invalid_argument("invalid embedding replacement");
  }
  const auto& layout =
      recipe.tensors.at(recipe.embedding_boundary->tensor.value);
  if (layout.shape.size() != 2 || layout.shape.at(1) != row.width) {
    throw std::invalid_argument(
        "replacement width differs from embedding boundary");
  }
  if (row.device) {
    const auto& input = *row.device;
    if (!row.bytes.empty() || !input.source.owner ||
        input.source.bytes !=
            multiply(row.width, tensors::element_bytes(input.source_type)) ||
        !std::isfinite(input.scale) || input.scale <= 0) {
      throw std::invalid_argument("invalid device injection descriptor");
    }
    return;  // Values are validated on-device; no CPU dereference is
             // possible.
  }
  for (std::size_t i = 0; i < row.width; ++i) {
    float value{};
    if (row.storage_type == tensors::ElementType::float32) {
      std::memcpy(&value,
                  std::span(row.bytes).subspan(i * sizeof(float)).data(),
                  sizeof(float));
    } else {
      value = tensors::to_float(tensors::read_le(std::span<const std::byte, 2>(
          std::span(row.bytes).subspan(i * 2).data(), 2)));
    }
    if (!std::isfinite(value)) {
      throw std::invalid_argument("nonfinite embedding replacement");
    }
  }
}
}  // namespace
struct ResidentModel::Impl {
  std::shared_ptr<ResidentState> state;
};
ResidentModel::ResidentModel(std::unique_ptr<Impl> p) : impl_(std::move(p)) {}
ResidentModel::~ResidentModel() = default;
std::string_view ResidentModel::checkpoint_identity() const noexcept {
  return impl_->state->checkpoint->identity();
}
std::size_t ResidentModel::resident_device_bytes() const noexcept {
  return impl_->state->bytes;
}
bool ResidentModel::usable() const noexcept { return !impl_->state->failed; }
std::unique_ptr<ResidentModel> load_resident_model(
    gpu::Device& device, std::unique_ptr<models::BoundCheckpoint> checkpoint,
    const ResidentOptions& options) {
  if (!checkpoint || !options.max_host_staging_bytes) {
    throw std::invalid_argument("invalid resident inputs");
  }
  const auto recipe =
      checkpoint->model().incremental_forward({1}, options.numerical_mode);
  models::validate_incremental_recipe(checkpoint->model(), recipe, {1},
                                      options.numerical_mode);
  auto state = std::make_shared<ResidentState>();
  state->device = &device;
  state->mode = options.numerical_mode;
  state->checkpoint = std::move(checkpoint);
  const auto parameters = state->checkpoint->model().parameters();
  state->offsets.reserve(parameters.size());
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    state->bytes = align_up(state->bytes, resident_alignment);
    state->offsets.push_back(state->bytes);
    state->bytes =
        add(state->bytes,
            tensors::required_bytes(dolittle::at(parameters, i).layout));
  }
  if (state->bytes > options.max_device_bytes) {
    throw std::length_error("resident weights exceed budget");
  }
  const auto staging_bytes =
      std::min({state->bytes, options.max_host_staging_bytes,
                std::size_t{2} * 1024 * 1024});
  constexpr std::size_t max_slots = 2;
  const std::size_t slot_count = staging_bytes >= max_slots ? max_slots : 1;
  const auto slot_bytes = staging_bytes / slot_count;
  std::array<std::optional<gpu::CompletionPoint>, max_slots> completions{};
  std::size_t slot = 0;
  state->queue.emplace(device.create_queue());
  auto& queue = present(state->queue);
  queue.reserve_completion_points(slot_count);
  state->weights.emplace(device.allocate({state->bytes, resident_alignment}));
  auto staging = device.allocate_staging({staging_bytes, 4});
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    const auto view = state->checkpoint->parameter({i});
    tensors::validate_view(view);
    for (std::size_t offset = 0; offset < view.bytes.size();) {
      auto& completion = completions.at(slot);
      if (completion) {
        queue.wait(*completion);
      }
      const auto bytes = std::min(slot_bytes, view.bytes.size() - offset);
      const auto staging_offset = slot * slot_bytes;
      std::memcpy(staging.bytes().subspan(staging_offset, bytes).data(),
                  view.bytes.subspan(offset, bytes).data(), bytes);
      queue.upload(
          {&present(state->weights), state->offsets.at(i) + offset, bytes},
          {&staging, staging_offset, bytes});
      completion = queue.mark();
      slot = (slot + 1) % slot_count;
      offset += bytes;
    }
  }
  queue.drain();
  auto impl = std::make_unique<ResidentModel::Impl>();
  impl->state = std::move(state);
  return std::unique_ptr<ResidentModel>(new ResidentModel(std::move(impl)));
}
struct CachedForwardPlan::Impl {
  struct Capture {
    models::ObservationSite site;
    kernels::TensorId source;
    std::size_t offset, row_bytes;
    std::size_t decode_offset = 0;
  };
  std::shared_ptr<ResidentState> resident;
  std::optional<EmbeddingOverride> injection;
  std::optional<gpu::HostBuffer> injection_staging;
  bool injected = false;
  gpu::Queue& work_queue() {
    return resident ? present(resident->queue) : present(queue);
  }
  const models::BoundCheckpoint* checkpoint;
  CachedForwardOptions options;
  models::IncrementalRecipe recipe;
  std::vector<Slot> slots;
  std::vector<Step> steps;
  std::vector<Capture> captures;
  gpu::Device* device{};
  std::unique_ptr<sampling::DeviceCategorical> sampler;
  std::optional<gpu::HostBuffer> selection_readback;
  std::unique_ptr<DeviceSequence> decode;
  std::unique_ptr<DeviceInjection> device_injection;
  std::optional<gpu::DeviceBuffer> prompt_ids;
  std::optional<gpu::HostBuffer> prompt_staging;
  std::shared_ptr<gpu::DeviceBuffer> retained_capture_storage;
  std::vector<DeviceCapture> retained_captures;
  std::size_t collected = 0;
  bool device_generation = false, decode_started = false;
  std::vector<models::TokenPosition> positions;
  std::vector<CapturedRows> views;
  std::vector<models::ObservationSite> sites;
  std::vector<std::vector<std::size_t>> copies_after;
  std::vector<std::optional<std::size_t>> cache_at;
  std::vector<std::optional<gpu::DeviceRegion>> cache_destinations;
  std::vector<std::vector<std::size_t>> cache_binding_indices;
  std::vector<std::uint32_t> history;
  std::size_t committed = 0, peak = 0, host_bytes = 0, output_bytes = 0,
              logit_bytes = 0;
  bool failed = false, finished = false;
  std::optional<gpu::Queue> queue;
  std::optional<gpu::DeviceBuffer> weights, scratch, output, constants;
  std::optional<gpu::HostBuffer> upload, readback;
  std::unique_ptr<cache::Contiguous> cache;
  Impl(const models::BoundCheckpoint& c, CachedForwardOptions o,
       models::IncrementalRecipe r)
      : checkpoint(&c), options(std::move(o)), recipe(std::move(r)) {}
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    if (decode) {
      decode->request_stop();
    }
    if (queue || resident) {
      try {
        work_queue().drain();
      } catch (...) {  // NOLINT(bugprone-empty-catch)
        // Preserve the active exception; buffer owners perform a final
        // no-throw drain.
      }
    }
    // Release graph borrowers while their queue and bound arenas still exist.
    decode.reset();
    device_injection.reset();
    if (resident) {
      resident->leased = false;
    }
  }
  gpu::DeviceBuffer* buffer(Arena arena) {
    switch (arena) {
      case Arena::weight:
        return resident ? &present(resident->weights) : &present(weights);
      case Arena::scratch:
        return &present(scratch);
      case Arena::output:
        return &present(output);
    }
    throw std::logic_error("unknown cached arena");
  }
  gpu::ConstDeviceRegion source(kernels::TensorId id, std::size_t bytes) {
    const auto& cached = cache_destinations.at(id.value);
    if (cached) {
      return {cached->owner, cached->offset, bytes};
    }
    const auto& slot = slots.at(id.value);
    return {buffer(slot.arena), slot.offset, bytes};
  }
  void submit_chunk(std::span<const std::uint32_t> ids,
                    std::size_t first_position, bool need_logits,
                    std::optional<sampling::DeviceDraw> draw,
                    gpu::HostBuffer& readback, gpu::HostBuffer* selection) {
    auto& p = *this;
    const kernels::ExecutionExtent extent{first_position, ids.size()};
    const bool direct_cache =
        p.options.max_prefill_chunk <= p.options.capacity - first_position;
    for (const auto& append : p.recipe.cache_appends) {
      if (direct_cache) {
        const auto window = p.cache->writable_window(
            p.work_queue(), append.block.value, first_position,
            p.options.max_prefill_chunk);
        p.cache_destinations.at(append.rotated_key.value) = window.keys;
        p.cache_destinations.at(append.value.value) = window.values;
      } else {
        p.cache_destinations.at(append.rotated_key.value).reset();
        p.cache_destinations.at(append.value.value).reset();
      }
    }
    const auto& tokens = p.slots.at(p.recipe.forward.token_input.value);
    const auto bytes = ids.size_bytes();
    std::memcpy(present(p.upload).bytes().data(), ids.data(), bytes);
    p.work_queue().upload({&present(p.scratch), tokens.offset, bytes},
                          {&present(p.upload), 0, bytes});
    for (std::size_t i = 0; i < p.steps.size(); ++i) {
      if (!need_logits && i == p.recipe.logits_begin) {
        break;
      }
      std::optional<cache::ContiguousLayerView> view;
      if (p.cache_at.at(i)) {
        const auto block = present(p.cache_at.at(i));
        const auto& append = p.recipe.cache_appends.at(block);
        view = p.cache->layer(block);
        const auto bytes_per_row =
            multiply(multiply(view->shape.kv_heads, view->shape.head_width),
                     tensors::element_bytes(
                         p.options.numerical_mode.activation_storage));
        if (!direct_cache) {
          p.cache->append(
              p.work_queue(), block, first_position, ids.size(),
              p.source(append.rotated_key, multiply(ids.size(), bytes_per_row)),
              p.source(append.value, multiply(ids.size(), bytes_per_row)));
        }
      }
      auto& step = p.steps.at(i);
      for (auto index : p.cache_binding_indices.at(i)) {
        auto& binding = step.bindings.at(index);
        const auto& destination = p.cache_destinations.at(binding.tensor.value);
        const auto& slot = p.slots.at(binding.tensor.value);
        auto* buffer = destination ? destination->owner : p.buffer(slot.arena);
        if (std::holds_alternative<gpu::DeviceBuffer*>(binding.storage)) {
          binding.storage = buffer;
        } else {
          binding.storage = static_cast<const gpu::DeviceBuffer*>(buffer);
        }
        binding.offset = destination ? destination->offset : slot.offset;
      }
      step.operation.submit(
          p.work_queue(), step.bindings,
          {step.operation.workspace().bytes ? &*p.constants : nullptr,
           step.workspace_offset, step.operation.workspace().bytes},
          i <= p.recipe.logits_begin ? &extent : nullptr,
          view ? &*view : nullptr);
      if (p.injection && !p.injected &&
          i == present(p.recipe.forward.embedding_boundary).producer_step &&
          p.injection->position.value >= first_position &&
          p.injection->position.value < first_position + ids.size()) {
        const auto& slot = p.slots.at(
            present(p.recipe.forward.embedding_boundary).tensor.value);
        const auto row_bytes = p.injection->bytes.size();
        const auto offset =
            slot.offset +
            (p.injection->position.value - first_position) * row_bytes;
        p.work_queue().upload({p.buffer(slot.arena), offset, row_bytes},
                              {&present(p.injection_staging), 0, row_bytes});
      }
      for (auto c : p.copies_after.at(i)) {
        const auto& capture = p.captures.at(c);
        const auto bytes = multiply(ids.size(), capture.row_bytes);
        p.work_queue().copy({&present(p.output), capture.offset, bytes},
                            p.source(capture.source, bytes));
      }
    }
    if (draw) {
      const gpu::ConstDeviceRegion logits{&present(p.output), 0, p.logit_bytes};
      const auto selected = draw ? p.sampler->submit(logits, *draw)
                                 : p.sampler->submit_next(logits);
      p.work_queue().download({selection, 0, selected.bytes}, selected);
    } else if (need_logits) {
      p.work_queue().download({&readback, 0, p.logit_bytes},
                              {&present(p.output), 0, p.logit_bytes});
    }
    for (const auto& capture : p.captures) {
      const auto bytes = multiply(ids.size(), capture.row_bytes);
      p.work_queue().download({&readback, capture.offset, bytes},
                              {&present(p.output), capture.offset, bytes});
    }
  }
  void poison() noexcept {
    if (decode) {
      decode->request_stop();
    }
    failed = true;
    if (resident) {
      resident->failed = true;
    }
    try {
      work_queue().drain();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Preserve the original error; destruction attempts final cleanup.
    }
  }
};
CachedForwardPlan::CachedForwardPlan(std::unique_ptr<Impl> p)
    : impl_(std::move(p)) {}
CachedForwardPlan::~CachedForwardPlan() = default;
CachedForwardPlan::CachedForwardPlan(CachedForwardPlan&&) noexcept = default;
CachedForwardPlan& CachedForwardPlan::operator=(CachedForwardPlan&&) noexcept =
    default;
std::string_view CachedForwardPlan::checkpoint_identity() const noexcept {
  return impl_->checkpoint->identity();
}
cache::SequenceId CachedForwardPlan::sequence_id() const noexcept {
  return impl_->options.sequence;
}
tensors::ComputationMode CachedForwardPlan::numerical_mode() const noexcept {
  return impl_->options.numerical_mode;
}
cache::Prefix CachedForwardPlan::prefix() const noexcept {
  return {impl_->committed, !impl_->failed && !impl_->finished};
}
std::size_t CachedForwardPlan::capacity() const noexcept {
  return impl_->options.capacity;
}
std::size_t CachedForwardPlan::max_prefill_chunk() const noexcept {
  return impl_->options.max_prefill_chunk;
}
std::size_t CachedForwardPlan::estimated_peak_device_bytes() const noexcept {
  return impl_->peak;
}
std::size_t CachedForwardPlan::estimated_peak_host_bytes() const noexcept {
  return impl_->host_bytes;
}
std::size_t CachedForwardPlan::vocabulary_size() const noexcept {
  return impl_->checkpoint->model().vocabulary_size();
}
std::span<const models::ObservationSite> CachedForwardPlan::capture_sites()
    const noexcept {
  return impl_->sites;
}
std::optional<models::TokenPosition>
CachedForwardPlan::embedding_override_position() const noexcept {
  return impl_->injection ? std::optional(impl_->injection->position)
                          : std::nullopt;
}
void CachedForwardPlan::finish() {
  if (impl_->injection && !impl_->injected && !impl_->device_generation) {
    throw std::logic_error("replacement not consumed");
  }
  if (impl_->failed) {
    throw std::runtime_error("failed cached execution plan");
  }
  if (impl_->decode_started) {
    impl_->decode->request_stop();
    try {
      impl_->decode->drain();
    } catch (...) {
      impl_->poison();
      throw;
    }
  }
  impl_->finished = true;
}
CachedForwardPlan prepare_cached_forward(
    gpu::Device& device, const models::BoundCheckpoint& checkpoint,
    const CachedForwardOptions& options) {
  return CachedForwardPlan::prepare_impl(device, checkpoint, options, nullptr,
                                         std::nullopt);
}
CachedForwardPlan prepare_cached_forward(
    ResidentModel& resident, const CachedForwardOptions& options,
    std::optional<EmbeddingOverride> injection) {
  auto& state = *resident.impl_->state;
  if (state.failed || state.leased ||
      state.mode.activation_storage !=
          options.numerical_mode.activation_storage ||
      state.mode.reduction_accumulation !=
          options.numerical_mode.reduction_accumulation) {
    throw std::invalid_argument("resident owner unavailable or mode mismatch");
  }
  return CachedForwardPlan::prepare_impl(*state.device, *state.checkpoint,
                                         options, &resident,
                                         std::move(injection));
}
CachedForwardPlan CachedForwardPlan::prepare_impl(
    gpu::Device& device, const models::BoundCheckpoint& checkpoint,
    const CachedForwardOptions& options, ResidentModel* resident,
    std::optional<EmbeddingOverride> injection) {
  const auto& model = checkpoint.model();
  const auto count = options.max_prefill_chunk;
  if (!count || !options.capacity || count > options.capacity ||
      options.capacity > model.context_limit().value) {
    throw std::invalid_argument("invalid cached capacity/chunk");
  }
  auto recipe = model.incremental_forward({count}, options.numerical_mode);
  models::validate_incremental_recipe(model, recipe, {count},
                                      options.numerical_mode);
  std::vector<ObservationRequest> requests;
  requests.reserve(options.capture_sites.size());
  for (auto point : options.capture_sites) {
    requests.push_back({point, AllPositions{}});
  }
  const auto observations = prepare_observations(
      model.observation_sites(options.numerical_mode), {options.capacity},
      requests, options.observation_limits);
  auto p = std::make_unique<CachedForwardPlan::Impl>(checkpoint, options,
                                                     std::move(recipe));
  auto& forward = p->recipe.forward;
  if (injection) {
    validate_injection(forward, *injection, options.capacity,
                       options.numerical_mode);
  }
  if (resident) {
    p->resident = resident->impl_->state;
    p->resident->leased = true;
  }
  p->injection = std::move(injection);
  cache::Specification spec{std::string(checkpoint.identity()),
                            options.sequence,
                            options.numerical_mode,
                            options.capacity,
                            {}};
  p->cache_at.resize(forward.operations.size());
  p->cache_destinations.resize(forward.tensors.size());
  p->cache_binding_indices.resize(forward.operations.size());
  for (const auto& append : p->recipe.cache_appends) {
    const auto& shape = forward.tensors.at(append.rotated_key.value).shape;
    spec.layers.push_back({shape.at(1), shape.at(2)});
    p->cache_at.at(append.attention_step) = append.block.value;
  }
  const auto cache_bytes = cache::estimate_contiguous(spec).allocation_bytes;
  p->slots.resize(forward.tensors.size());
  p->copies_after.resize(forward.operations.size());
  std::vector<std::size_t> first(p->slots.size(),
                                 std::numeric_limits<std::size_t>::max()),
      last(p->slots.size(), 0);
  for (std::size_t i = 0; i < p->slots.size(); ++i) {
    p->slots.at(i).bytes = tensors::required_bytes(forward.tensors.at(i));
    p->slots.at(i).alignment =
        tensors::element_bytes(forward.tensors.at(i).element_type);
  }
  for (std::size_t i = 0; i < forward.operations.size(); ++i) {
    kernels::RowPolicy rows;
    if (i <= p->recipe.logits_begin) {
      rows = {count, i == p->recipe.logits_begin};
    }
    auto operation =
        kernels::prepare(device, forward.operations.at(i), forward.tensors,
                         options.numerical_mode, rows);
    for (const auto& use : operation.operands()) {
      auto& slot = p->slots.at(use.tensor.value);
      slot.alignment = std::max(slot.alignment, use.alignment);
      first.at(use.tensor.value) = std::min(first.at(use.tensor.value), i);
      last.at(use.tensor.value) = i;
    }
    p->steps.push_back({std::move(operation), {}});
  }
  // Retain one vocabulary row; captures are chunk-sized, not capacity-sized.
  const auto logits = present(forward.logits);
  auto& logit_slot = p->slots.at(logits.value);
  logit_slot.arena = Arena::output;
  logit_slot.offset = 0;
  p->logit_bytes = logit_slot.bytes;
  p->output_bytes = p->logit_bytes;
  for (const auto& observation : observations.entries()) {
    const auto it = std::find_if(
        forward.observations.begin(), forward.observations.end(),
        [&](const auto& value) {
          return same_point(value.site.point, observation.site.point);
        });
    if (it == forward.observations.end()) {
      throw std::invalid_argument("unresolved cached observation");
    }
    const auto row =
        multiply(observation.site.vector_width,
                 tensors::element_bytes(observation.site.storage_type));
    p->sites.push_back(observation.site);
    p->captures.push_back({observation.site, it->tensor, p->output_bytes, row});
    p->output_bytes = add(p->output_bytes, multiply(count, row));
    bool found = false;
    for (std::size_t i = 0; i < p->steps.size(); ++i) {
      for (const auto& use : p->steps.at(i).operation.operands()) {
        if (use.tensor.value == it->tensor.value &&
            use.access == kernels::OperandAccess::write) {
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
  for (const auto& input : forward.parameters) {
    auto& slot = p->slots.at(input.tensor.value);
    slot.arena = Arena::weight;
    const auto view = checkpoint.parameter(input.parameter);
    tensors::validate_view(view);
    const auto& expected = forward.tensors.at(input.tensor.value);
    if (view.layout->shape != expected.shape ||
        view.layout->element_type != expected.element_type ||
        view.bytes.size() != slot.bytes) {
      throw std::invalid_argument("cached parameter shape/extent mismatch");
    }
    for (std::size_t axis = 0; axis < expected.shape.size(); ++axis) {
      if (expected.shape.at(axis) > 1 &&
          expected.byte_strides.at(axis) !=
              view.layout->byte_strides.at(axis)) {
        throw std::invalid_argument("cached parameter strides mismatch");
      }
    }
    if (p->resident) {
      slot.offset = p->resident->offsets.at(input.parameter.value);
      if (slot.offset % slot.alignment) {
        throw std::invalid_argument("resident weight alignment mismatch");
      }
      weight_bytes = p->resident->bytes;
    } else {
      weight_bytes = align_up(weight_bytes, slot.alignment);
      slot.offset = weight_bytes;
      weight_bytes = add(weight_bytes, slot.bytes);
    }
  }
  first.at(forward.token_input.value) = 0;
  std::vector<ResourceLifetime> lifetimes;
  for (std::size_t i = 0; i < p->slots.size(); ++i) {
    const auto& slot = p->slots.at(i);
    if (slot.arena == Arena::scratch) {
      if (first.at(i) == std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("unused cached scratch");
      }
      lifetimes.push_back(
          {{i}, slot.bytes, slot.alignment, first.at(i), last.at(i)});
    }
  }
  std::size_t constants_bytes = 0;
  for (auto& step : p->steps) {
    step.workspace_offset = constants_bytes;
    constants_bytes = add(constants_bytes, step.operation.workspace().bytes);
  }
  const auto fixed = add(add(add(weight_bytes, cache_bytes), p->output_bytes),
                         constants_bytes);
  if (fixed > options.max_device_bytes) {
    throw std::length_error("cached fixed arenas exceed device budget");
  }
  const auto storage =
      plan_storage(lifetimes, options.max_device_bytes - fixed);
  p->peak = add(fixed, storage.arena_bytes);
  for (const auto& placement : storage.placements) {
    p->slots.at(placement.id.value).offset = placement.offset;
  }
  const auto upload_bytes = std::max(
      multiply(count, sizeof(std::uint32_t)),
      p->resident ? 0 : std::min(weight_bytes, std::size_t{16} * 1024 * 1024));
  auto host_bytes = add(upload_bytes, multiply(2, p->output_bytes));
  host_bytes = add(host_bytes, observations.payload_bytes());
  host_bytes =
      add(host_bytes, multiply(options.capacity, 2 * sizeof(std::uint32_t)));
  host_bytes = add(host_bytes, multiply(count, sizeof(models::TokenPosition)));
  host_bytes =
      add(host_bytes, multiply(p->captures.size(), sizeof(CapturedRows)));
  host_bytes = add(host_bytes, multiply(multiply(count, p->captures.size()),
                                        sizeof(models::TokenPosition)));
  host_bytes =
      add(host_bytes, multiply(multiply(options.capacity, p->captures.size()),
                               sizeof(models::TokenPosition)));
  host_bytes = add(host_bytes, multiply(model.vocabulary_size(),
                                        sizeof(float) + sizeof(double) +
                                            2 * sizeof(std::uint32_t)));
  if (p->injection) {
    host_bytes = add(host_bytes, multiply(2, p->injection->bytes.size()));
  }
  if (host_bytes > options.max_host_bytes) {
    throw std::length_error("cached host outputs exceed budget");
  }
  p->host_bytes = host_bytes;
  p->history.reserve(options.capacity);
  if (!p->resident) {
    p->queue.emplace(device.create_queue());
    p->weights.emplace(device.allocate({weight_bytes, 4}));
  }
  if (constants_bytes) {
    p->constants.emplace(device.allocate({constants_bytes, alignof(float)}));
    for (const auto& step : p->steps) {
      if (step.operation.workspace().bytes) {
        step.operation.initialize(p->work_queue(),
                                  {&*p->constants, step.workspace_offset,
                                   step.operation.workspace().bytes});
      }
    }
  }
  if (p->injection && !p->injection->device) {
    p->injection_staging.emplace(
        device.allocate_staging({p->injection->bytes.size(), 4}));
    std::memcpy(present(p->injection_staging).bytes().data(),
                p->injection->bytes.data(), p->injection->bytes.size());
  }
  p->scratch.emplace(
      device.allocate({storage.arena_bytes, storage.arena_alignment}));
  p->output.emplace(device.allocate({p->output_bytes, 4}));
  p->upload.emplace(device.allocate_staging({upload_bytes, 4}));
  p->readback.emplace(device.allocate_staging({p->output_bytes, 4}));
  p->cache =
      cache::allocate_contiguous(device, p->work_queue(), spec, cache_bytes);
  if (!p->resident) {
    for (const auto& input : forward.parameters) {
      const auto view = checkpoint.parameter(input.parameter);
      const auto& slot = p->slots.at(input.tensor.value);
      for (std::size_t offset = 0; offset < slot.bytes;) {
        const auto bytes = std::min(upload_bytes, slot.bytes - offset);
        std::memcpy(present(p->upload).bytes().data(),
                    view.bytes.subspan(offset).data(), bytes);
        p->work_queue().upload(
            {&present(p->weights), slot.offset + offset, bytes},
            {&present(p->upload), 0, bytes});
        p->work_queue().wait(p->work_queue().mark());
        offset += bytes;
      }
    }
  }
  for (auto& step : p->steps) {
    for (const auto& use : step.operation.operands()) {
      const auto& slot = p->slots.at(use.tensor.value);
      auto* buffer = p->buffer(slot.arena);
      if (use.access == kernels::OperandAccess::read) {
        step.bindings.push_back({use.tensor,
                                 static_cast<const gpu::DeviceBuffer*>(buffer),
                                 slot.offset, slot.bytes});
      } else {
        step.bindings.push_back({use.tensor, buffer, slot.offset, slot.bytes});
      }
    }
  }
  for (const auto& append : p->recipe.cache_appends) {
    for (std::size_t i = 0; i < p->steps.size(); ++i) {
      const auto& bindings = p->steps.at(i).bindings;
      for (std::size_t b = 0; b < bindings.size(); ++b) {
        const auto id = bindings.at(b).tensor.value;
        if (id == append.rotated_key.value || id == append.value.value) {
          p->cache_binding_indices.at(i).push_back(b);
        }
      }
    }
  }
  p->device = &device;
  p->positions.resize(p->options.max_prefill_chunk);
  p->views.reserve(p->captures.size());
  for (const auto& capture : p->captures) {
    p->views.push_back({capture.site, {}, {}});
  }
  return CachedForwardPlan(std::move(p));
}
ConsumedView CachedForwardPlan::consume_view(
    std::span<const std::uint32_t> ids, bool need_logits,
    std::optional<sampling::DeviceDraw> draw) {
  auto& p = *impl_;
  if (p.injection && p.injection->device) {
    throw std::invalid_argument(
        "device injection requires autonomous execution");
  }
  if (p.failed || p.finished || p.device_generation) {
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
  if (draw && (!need_logits || !p.sampler)) {
    throw std::invalid_argument("device selection not prepared or requested");
  }
  ConsumedView result{};
  result.first_position = {p.committed};
  result.tokens = ids;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    p.positions.at(i) = {p.committed + i};
  }
  const auto positions = std::span(p.positions).first(ids.size());
  for (std::size_t c = 0; c < p.captures.size(); ++c) {
    const auto& capture = p.captures.at(c);
    p.views.at(c) = {
        capture.site, positions,
        present(p.readback)
            .bytes()
            .subspan(capture.offset, multiply(ids.size(), capture.row_bytes))};
  }
  result.captures = p.views;
  if (need_logits && !draw && !p.device_generation) {
    result.next_logits =
        LogitRows{positions.last(1), vocabulary_size(),
                  p.options.numerical_mode.activation_storage,
                  present(p.readback).bytes().first(p.logit_bytes)};
  }
  try {
    p.submit_chunk(ids, p.committed, need_logits, draw, present(p.readback),
                   p.selection_readback ? &*p.selection_readback : nullptr);
    p.work_queue().wait(p.work_queue().mark());
    if (draw) {
      std::array<std::uint32_t, 2> selected{};
      std::memcpy(selected.data(), present(p.selection_readback).bytes().data(),
                  sizeof(selected));
      if (selected.at(1) || selected.at(0) >= vocabulary_size()) {
        throw std::invalid_argument("invalid device sampling distribution");
      }
      result.selected_token = selected.at(0);
    }
    p.history.insert(p.history.end(), ids.begin(), ids.end());
    result.tokens = std::span(p.history).subspan(p.committed, ids.size());
    p.committed += ids.size();
    if (p.injection && p.injection->position.value < p.committed) {
      p.injected = true;
    }
  } catch (...) {
    p.failed = true;
    if (p.resident) {
      p.resident->failed = true;
    }
    try {
      p.work_queue().drain();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Preserve the active exception; buffer owners perform a final no-throw
      // drain.
    }
    throw;
  }
  return result;
}

void CachedForwardPlan::prepare_device_generation(
    const DeviceGenerationRequest& request) {
  const auto& [settings, seed, prompt, stop_ids, generated] = request;
  auto& p = *impl_;
  if (p.device_generation || p.committed || p.failed || p.finished ||
      prompt.empty() || prompt.size() > p.options.capacity ||
      generated > p.options.capacity - prompt.size()) {
    throw std::logic_error("sequence must be prepared once before execution");
  }
  if (p.injection && !p.injection->device) {
    throw std::invalid_argument(
        "live injection requires a device normalization input");
  }
  prepare_sampling();
  p.sampler->start(settings, seed, stop_ids);
  const auto capacity = prompt.size() + generated;
  const DeviceSequence::Shape shape{
      prompt.size(), generated,
      std::min(prompt.size(), p.options.max_prefill_chunk)};
  std::size_t capture_bytes = 0;
  for (auto& c : p.captures) {
    c.decode_offset = capture_bytes;
    capture_bytes = add(capture_bytes, multiply(capacity, c.row_bytes));
  }
  auto extra_device = add(
      add(DeviceSequence::device_bytes(), prompt.size_bytes()), capture_bytes);
  if (p.injection) {
    extra_device =
        add(extra_device, DeviceInjection::required_bytes(
                              p.injection->width, p.injection->storage_type));
  }
  const auto extra_host =
      add(DeviceSequence::host_bytes(shape), prompt.size_bytes());
  if (extra_host > p.options.max_host_bytes - p.host_bytes ||
      extra_device > p.options.max_device_bytes - p.peak) {
    throw std::length_error("sequence resources exceed admitted budget");
  }
  p.prompt_ids.emplace(
      p.device->allocate({prompt.size_bytes(), alignof(std::uint32_t)}));
  p.prompt_staging.emplace(p.device->allocate_staging(
      {prompt.size_bytes(), alignof(std::uint32_t)}));
  std::memcpy(p.prompt_staging->bytes().data(), prompt.data(),
              prompt.size_bytes());
  p.work_queue().upload({&present(p.prompt_ids), 0, prompt.size_bytes()},
                        {&present(p.prompt_staging), 0, prompt.size_bytes()});
  if (capture_bytes) {
    p.retained_capture_storage = std::make_shared<gpu::DeviceBuffer>(
        p.device->allocate({capture_bytes, 4}));
    p.retained_captures.reserve(p.captures.size());
    for (const auto& c : p.captures) {
      p.retained_captures.push_back({p.retained_capture_storage,
                                     c.decode_offset,
                                     multiply(capacity, c.row_bytes)});
    }
  }
  if (p.injection) {
    const auto& injection = present(p.injection);
    const auto& input = present(injection.device);
    p.device_injection = std::make_unique<DeviceInjection>(
        *p.device, p.work_queue(), input.source.region(), input.source_type,
        injection.storage_type, injection.width, input.scale);
  }
  p.decode = std::make_unique<DeviceSequence>(*p.device, p.work_queue(), shape);
  const auto position = p.decode->position();
  const auto& tokens = p.slots.at(p.recipe.forward.token_input.value);
  try {
    p.decode->begin(p.sampler->result_storage(), p.sampler->terminal_state(),
                    {&present(p.scratch), tokens.offset, tokens.bytes},
                    {&present(p.prompt_ids), 0, prompt.size_bytes()},
                    p.device_injection ? p.device_injection->status()
                                       : gpu::ConstDeviceRegion{},
                    p.device_injection ? p.device_injection->norm()
                                       : gpu::ConstDeviceRegion{});
    if (p.device_injection) {
      p.device_injection->submit();
    }
    for (const auto& append : p.recipe.cache_appends) {
      const auto window = p.cache->writable_window(
          p.work_queue(), append.block.value, 0, p.options.max_prefill_chunk);
      p.cache_destinations.at(append.rotated_key.value) = window.keys;
      p.cache_destinations.at(append.value.value) = window.values;
    }
    const auto submit = [&](std::size_t rows, std::size_t first,
                            std::size_t end, bool captures,
                            bool prompt_body = false) {
      const kernels::ExecutionExtent extent{0, rows, &position,
                                            p.options.capacity};
      for (std::size_t i = first; i < end; ++i) {
        std::optional<cache::ContiguousLayerView> view;
        if (p.cache_at.at(i)) {
          view = p.cache->layer(*p.cache_at.at(i));
        }
        auto& step = p.steps.at(i);
        for (auto index : p.cache_binding_indices.at(i)) {
          auto& binding = step.bindings.at(index);
          const auto& destination =
              p.cache_destinations.at(binding.tensor.value);
          if (std::holds_alternative<gpu::DeviceBuffer*>(binding.storage)) {
            binding.storage = destination->owner;
            binding.positioned_output = true;
          } else {
            binding.storage =
                static_cast<const gpu::DeviceBuffer*>(destination->owner);
          }
          binding.offset = destination->offset;
        }
        step.operation.submit(
            p.work_queue(), step.bindings,
            {step.operation.workspace().bytes ? &*p.constants : nullptr,
             step.workspace_offset, step.operation.workspace().bytes},
            i <= p.recipe.logits_begin ? &extent : nullptr,
            view ? &*view : nullptr);
        if (prompt_body && p.device_injection &&
            i == present(p.recipe.forward.embedding_boundary).producer_step) {
          const auto& slot = p.slots.at(
              present(p.recipe.forward.embedding_boundary).tensor.value);
          p.decode->replace(p.device_injection->output(),
                            {p.buffer(slot.arena), slot.offset, slot.bytes},
                            p.injection->position.value, rows);
        }
        if (captures) {
          for (auto c : p.copies_after.at(i)) {
            const auto& capture = p.captures.at(c);
            if (p.cache_destinations.at(capture.source.value)) {
              throw std::invalid_argument("capture aliases cache producer");
            }
            p.decode->capture(
                p.source(capture.source, multiply(rows, capture.row_bytes)),
                {p.retained_capture_storage.get(), capture.decode_offset,
                 multiply(capacity, capture.row_bytes)},
                capture.row_bytes, rows);
          }
        }
      }
    };
    p.decode->begin_stage(DeviceSequence::Stage::prefill);
    submit(shape.chunk, 0, p.recipe.logits_begin, true, true);
    p.decode->end_stage();
    p.decode->begin_stage(DeviceSequence::Stage::tail);
    const auto remainder = prompt.size() % shape.chunk;
    if (remainder) {
      submit(remainder, 0, p.recipe.logits_begin, true, true);
    }
    p.decode->end_stage();
    p.decode->begin_stage(DeviceSequence::Stage::first_selection);
    submit(remainder ? remainder : shape.chunk, p.recipe.logits_begin,
           p.steps.size(), false);
    (void)p.sampler->submit_next({&present(p.output), 0, p.logit_bytes});
    p.decode->end_stage();
    p.decode->begin_stage(DeviceSequence::Stage::decode);
    submit(1, 0, p.recipe.logits_begin, true);
    p.decode->begin_selection();
    submit(1, p.recipe.logits_begin, p.steps.size(), false);
    (void)p.sampler->submit_next({&present(p.output), 0, p.logit_bytes});
    p.decode->end_stage();
    p.decode->complete();
  } catch (...) {
    p.decode.reset();
    p.poison();
    throw;
  }
  p.host_bytes = add(p.host_bytes, extra_host);
  p.peak = add(p.peak, extra_device);
  p.device_generation = true;
}
void CachedForwardPlan::request_stop() noexcept {
  if (impl_->decode) {
    impl_->decode->request_stop();
  }
}
void CachedForwardPlan::start_sequence() {
  auto& p = *impl_;
  if (!p.device_generation || !p.decode || p.failed || p.finished ||
      p.decode_started) {
    throw std::logic_error("sequence unavailable");
  }
  try {
    p.decode->start();
    p.decode_started = true;
  } catch (...) {
    p.poison();
    throw;
  }
}
DeviceSequence::Event CachedForwardPlan::collect_sequence(
    const std::stop_token& stop) {
  auto& p = *impl_;
  if (!p.decode_started || p.failed || p.finished) {
    throw std::logic_error("sequence collection unavailable");
  }
  try {
    const auto event = p.decode->collect(p.collected++, stop);
    if (event.kind == DeviceSequence::Kind::consumed) {
      if (event.position != p.committed ||
          event.count > p.options.capacity - p.committed) {
        throw std::runtime_error("invalid sequence consumed range");
      }
      p.committed += event.count;
    }
    return event;
  } catch (...) {
    p.poison();
    throw;
  }
}
std::span<const DeviceCapture> CachedForwardPlan::device_captures()
    const noexcept {
  return impl_->retained_captures;
}

void CachedForwardPlan::prepare_sampling() {
  auto& p = *impl_;
  if (p.sampler) {
    return;
  }
  if (p.committed || p.failed || p.finished) {
    throw std::logic_error("selection must be prepared before consumption");
  }
  constexpr auto result_bytes = 2 * sizeof(std::uint32_t);
  const auto bytes =
      sampling::DeviceCategorical::required_bytes(vocabulary_size());
  if (p.options.max_host_bytes - p.host_bytes < result_bytes ||
      bytes > p.options.max_device_bytes - p.peak) {
    throw std::length_error("selection exceeds admitted budget");
  }
  auto sampler = std::make_unique<sampling::DeviceCategorical>(
      *p.device, p.work_queue(), vocabulary_size(),
      p.options.numerical_mode.activation_storage, bytes);
  auto readback =
      p.device->allocate_staging({result_bytes, alignof(std::uint32_t)});
  p.sampler = std::move(sampler);
  p.selection_readback.emplace(std::move(readback));
  p.host_bytes = add(p.host_bytes, result_bytes);
  p.peak = add(p.peak, bytes);
}
ConsumedChunk CachedForwardPlan::consume(std::span<const std::uint32_t> ids,
                                         bool need_logits) {
  const auto view = consume_view(ids, need_logits);
  ConsumedChunk result{};
  result.first_position = view.first_position;
  result.tokens.assign(ids.begin(), ids.end());
  if (view.next_logits) {
    const auto& row = *view.next_logits;
    result.next_logits = OwnedLogitRow{row.positions.front(),
                                       row.vocabulary_size,
                                       row.storage_type,
                                       {row.bytes.begin(), row.bytes.end()}};
  }
  result.captures.reserve(view.captures.size());
  for (const auto& capture : view.captures) {
    result.captures.push_back(
        {capture.site,
         {capture.positions.begin(), capture.positions.end()},
         {capture.bytes.begin(), capture.bytes.end()}});
  }
  return result;
}

}  // namespace dolittle::inference
