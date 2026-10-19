/*
 * Shared model lifetime and structural recipe validation.
 * Axis M6: pure structural validation of a model-produced forward recipe.
 * Contract: implement validate_recipe and Model lifetime definitions. Validation
 * does not execute operations, load artifacts or query a device. It follows
 * model.h's invalid-input/overflow/allocation failures, with no partial side effects.
 * Algorithm: transformer overlay, "Preparing the first execution recipe":
 * validate token shape, parameter assignments and layouts; walk ordered operations
 * using validate_operation; reject read-before-produce, multiple producers and
 * writes to external input; validate catalog/site values and selected logit shape.
 * Adjacent block output/input sites may share a tensor or name distinct values;
 * their semantic equality requires numerical integration, not tensor-ID equality.
 * Check every slot, including unused slots, has exactly one source; no tensor
 * values are examined and selected-logit numerical row order is not proved.
 * Layout matching ignores singleton-axis stride differences: those strides
 * address no additional elements. All stride units and extents remain checked.
 * Incremental validation then checks one CacheAppend per attention operation,
 * with matching K/V producer IDs and consecutive block identities, before the
 * last-row gather. Its output tail permits only norm/projection operations;
 * cache completion and initialized prefix length are runtime responsibilities.
 * Design and cost: O(tensors+operand uses+sites) expected work with indexed state,
 * plus model metadata creation. Optimization: validate once before resource planning.
 * Tests check producer, shape, site and ownership contracts using in-memory data;
 * they do not snapshot slot numbers, operation counts or a private recipe layout.
 * Mathematical equivalence still requires independent numerical integration.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, implementation (full rules read), 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): enforce the stated results and failure boundaries.
 * cpp-guidelines I.4 (#ri-typed): distinct IDs and operation access roles.
 * cpp-perf-guidelines CDSA.32: validate indexed metadata once before execution.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 *
 * Embedding capability validation: if declared, the boundary must identify the
 * token lookup operation's dense [N,D] activation output. Existing producer/read
 * validation proves all consumers follow it. Runtime replacement occurs after
 * that producer, before its consumers; validation does not perform injection.
 * Corpus consultation, implementation (both servers queried, full rules read),
 * 2026-10-19: cpp-guidelines R.1 (#rr-raii) ties resources to their owning phase;
 * I.7 (#ri-post) makes completion, provenance and failure effects explicit.
 * cpp-perf-guidelines GPU.1 keeps intermediate embeddings on device and budgets
 * required transfers; source guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.9 separates persistent weights from sequence storage;
 * source guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "bounds.h"
#include "models/model.h"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace dolittle::models {
namespace {

bool same_layout(const tensors::Layout &left, const tensors::Layout &right) {
    if (left.element_type != right.element_type || left.shape != right.shape) {
        return false;
    }
    // Both layouts have already passed extent/rank/stride-unit validation.
    for (std::size_t axis = 0; axis < left.shape.size(); ++axis) {
        if (left.shape.at(axis) > 1 && left.byte_strides.at(axis) != right.byte_strides.at(axis)) {
            return false;
        }
    }
    return true;
}

bool same_point(ObservationPoint left, ObservationPoint right) {
    return left.boundary == right.boundary && left.block.value == right.block.value;
}

struct PointHash {
    std::size_t operator()(ObservationPoint point) const noexcept {
        return std::hash<std::size_t>{}(point.block.value) ^
               std::hash<int>{}(static_cast<int>(point.boundary));
    }
};

struct PointEqual {
    bool operator()(ObservationPoint left, ObservationPoint right) const noexcept {
        return same_point(left, right);
    }
};

struct SiteState {
    ObservationSite site;
    std::optional<kernels::TensorId> tensor;
};

enum class Source : std::uint8_t { absent, external, operation };

void validate_site(const ObservationSite &site, tensors::ComputationMode mode) {
    switch (site.point.boundary) {
    case ObservationBoundary::block_input:
    case ObservationBoundary::block_output:
        break;
    default:
        throw std::invalid_argument("unknown observation boundary");
    }
    if (site.vector_width == 0 || site.storage_type != mode.activation_storage) {
        throw std::invalid_argument("observation width or storage disagrees with mode");
    }
}

} // namespace

Model::Model() = default;
Model::~Model() = default;

void validate_recipe(const Model &model, const ForwardRecipe &recipe, SequenceLength length,
                     tensors::ComputationMode mode, std::span<const TokenPosition> logit_positions) {
    if (length.value == 0 || length.value > model.context_limit().value || model.vocabulary_size() == 0 ||
        model.vocabulary_size() - 1 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid model or sequence dimensions");
    }
    if ((mode.activation_storage != tensors::ElementType::bfloat16 &&
         mode.activation_storage != tensors::ElementType::float32) ||
        mode.reduction_accumulation != tensors::ElementType::float32) {
        throw std::invalid_argument("unsupported recipe computation mode");
    }
    for (std::size_t i = 0; i < logit_positions.size(); ++i) {
        if (dolittle::at(logit_positions, i).value >= length.value ||
            (i != 0 &&
             dolittle::at(logit_positions, i - 1).value >= dolittle::at(logit_positions, i).value)) {
            throw std::invalid_argument("logit positions must increase in range");
        }
    }
    for (const auto &layout : recipe.tensors) {
        (void)tensors::required_bytes(layout);
    }
    const auto layout_at = [&](kernels::TensorId id) -> const tensors::Layout & {
        if (id.value >= recipe.tensors.size()) {
            throw std::invalid_argument("recipe tensor ID is out of range");
        }
        return recipe.tensors.at(id.value);
    };
    const auto expected_tokens =
        tensors::contiguous_layout(tensors::ElementType::uint32, std::array{length.value});
    if (!same_layout(layout_at(recipe.token_input), expected_tokens)) {
        throw std::invalid_argument("token input must be dense uint32 sequence");
    }
    if (recipe.embedding_boundary) {
        const auto boundary = *recipe.embedding_boundary;
        if (boundary.producer_step >= recipe.operations.size()) {
            throw std::invalid_argument("embedding producer out of range");
        }
        const auto *embedding = std::get_if<kernels::Embedding>(&recipe.operations.at(boundary.producer_step));
        if (!embedding || embedding->output.value != boundary.tensor.value ||
            embedding->tokens.value != recipe.token_input.value) {
            throw std::invalid_argument("embedding boundary must name token lookup output");
        }
        const auto &layout = layout_at(boundary.tensor);
        if (layout.shape.size() != 2 || layout.shape.at(0) != length.value ||
            layout.element_type != mode.activation_storage ||
            !same_layout(layout, tensors::contiguous_layout(layout.element_type, layout.shape))) {
            throw std::invalid_argument("embedding boundary must be dense activation rows");
        }
    }
    std::vector<Source> sources(recipe.tensors.size(), Source::absent);
    sources.at(recipe.token_input.value) = Source::external;

    const auto specs = model.parameters();
    if (recipe.parameters.size() != specs.size()) {
        throw std::invalid_argument("recipe parameter membership is incomplete");
    }
    std::unordered_set<std::string_view> roles;
    for (const auto &spec : specs) {
        (void)tensors::required_bytes(spec.layout);
        if (spec.role.empty() || !roles.insert(spec.role).second) {
            throw std::invalid_argument("parameter roles must be nonempty and unique");
        }
    }
    std::vector<bool> bound_parameters(specs.size(), false);
    for (const auto &parameter : recipe.parameters) {
        if (parameter.parameter.value >= specs.size() || bound_parameters.at(parameter.parameter.value)) {
            throw std::invalid_argument("invalid or duplicate parameter assignment");
        }
        const auto &layout = layout_at(parameter.tensor);
        if (sources.at(parameter.tensor.value) != Source::absent ||
            !same_layout(layout, dolittle::at(specs, parameter.parameter.value).layout)) {
            throw std::invalid_argument("parameter slot or layout mismatch");
        }
        bound_parameters.at(parameter.parameter.value) = true;
        sources.at(parameter.tensor.value) = Source::external;
    }

    for (const auto &operation : recipe.operations) {
        const auto uses = kernels::validate_operation(operation, recipe.tensors, mode);
        for (const auto &use : uses) {
            if (use.access == kernels::OperandAccess::read &&
                sources.at(use.tensor.value) == Source::absent) {
                throw std::invalid_argument("recipe reads an unproduced tensor");
            }
            if (use.access == kernels::OperandAccess::write &&
                sources.at(use.tensor.value) != Source::absent) {
                throw std::invalid_argument("recipe overwrites a tensor source");
            }
        }
        for (const auto &use : uses) {
            if (use.access == kernels::OperandAccess::write) {
                sources.at(use.tensor.value) = Source::operation;
            }
        }
    }
    for (const auto source : sources) {
        if (source == Source::absent) {
            throw std::invalid_argument("recipe contains a tensor without a source");
        }
    }

    std::unordered_map<ObservationPoint, SiteState, PointHash, PointEqual> sites;
    const auto catalog = model.observation_sites(mode);
    sites.reserve(catalog.size());
    for (const auto &site : catalog) {
        validate_site(site, mode);
        if (!sites.emplace(site.point, SiteState{site, std::nullopt}).second) {
            throw std::invalid_argument("duplicate observation catalog point");
        }
    }
    if (recipe.observations.size() != catalog.size()) {
        throw std::invalid_argument("recipe observation membership is incomplete");
    }
    for (const auto &value : recipe.observations) {
        validate_site(value.site, mode);
        const auto found = sites.find(value.site.point);
        if (found == sites.end() || found->second.tensor.has_value() ||
            found->second.site.vector_width != value.site.vector_width ||
            found->second.site.storage_type != value.site.storage_type) {
            throw std::invalid_argument("recipe observation differs from model catalog");
        }
        const auto &layout = layout_at(value.tensor);
        const auto expected = tensors::contiguous_layout(value.site.storage_type,
                                                         std::array{length.value, value.site.vector_width});
        if (sources.at(value.tensor.value) != Source::operation || !same_layout(layout, expected)) {
            throw std::invalid_argument("observation must name a produced full-sequence value");
        }
        found->second.tensor = value.tensor;
    }
    if (logit_positions.empty()) {
        if (recipe.logits) {
            throw std::invalid_argument("unrequested logits must be absent");
        }
    } else {
        if (!recipe.logits) {
            throw std::invalid_argument("requested logits are absent");
        }
        const auto &layout = layout_at(*recipe.logits);
        const auto expected = tensors::contiguous_layout(
            mode.activation_storage, std::array{logit_positions.size(), model.vocabulary_size()});
        if (sources.at(recipe.logits->value) != Source::operation || !same_layout(layout, expected)) {
            throw std::invalid_argument("selected logits layout or producer mismatch");
        }
    }
}

IncrementalRecipe Model::incremental_forward(SequenceLength, tensors::ComputationMode) const {
    throw std::invalid_argument("model does not support incremental execution");
}
void validate_incremental_recipe(const Model &model, const IncrementalRecipe &incremental,
                                 SequenceLength length, tensors::ComputationMode mode) {
    if (!length.value) {
        throw std::invalid_argument("empty incremental chunk");
    }
    const std::array<TokenPosition, 1> positions{{{length.value - 1}}};
    const auto &recipe = incremental.forward;
    validate_recipe(model, recipe, length, mode, positions);
    if (incremental.cache_appends.empty() || incremental.logits_begin >= recipe.operations.size()) {
        throw std::invalid_argument("incomplete incremental recipe");
    }
    const auto *gather = std::get_if<kernels::GatherRows>(&recipe.operations.at(incremental.logits_begin));
    if (!gather || gather->rows != std::vector<std::size_t>{length.value - 1}) {
        throw std::invalid_argument("incremental output must select the last new row");
    }
    std::size_t next = 0;
    for (std::size_t i = 0; i < recipe.operations.size(); ++i) {
        const auto *attention = std::get_if<kernels::CausalAttention>(&recipe.operations.at(i));
        if (!attention) {
            continue;
        }
        if (i >= incremental.logits_begin || next >= incremental.cache_appends.size()) {
            throw std::invalid_argument("unbound incremental attention");
        }
        const auto &append = incremental.cache_appends.at(next);
        if (append.block.value != next || append.attention_step != i ||
            append.rotated_key.value != attention->key.value ||
            append.value.value != attention->value.value) {
            throw std::invalid_argument("invalid incremental cache binding");
        }
        ++next;
    }
    if (next != incremental.cache_appends.size()) {
        throw std::invalid_argument("unused cache binding");
    }
    for (std::size_t i = incremental.logits_begin; i < recipe.operations.size(); ++i) {
        for (const auto &use : kernels::validate_operation(recipe.operations.at(i), recipe.tensors, mode)) {
            if (use.access != kernels::OperandAccess::write) {
                continue;
            }
            for (const auto &observation : recipe.observations) {
                if (observation.tensor.value == use.tensor.value) {
                    throw std::invalid_argument("capture producer lies in optional logit tail");
                }
            }
        }
    }
    for (std::size_t i = incremental.logits_begin + 1; i < recipe.operations.size(); ++i) {
        if (!std::holds_alternative<kernels::RmsNorm>(recipe.operations.at(i)) &&
            !std::holds_alternative<kernels::Projection>(recipe.operations.at(i))) {
            throw std::invalid_argument("unsupported incremental logit branch");
        }
    }
}

} // namespace dolittle::models
