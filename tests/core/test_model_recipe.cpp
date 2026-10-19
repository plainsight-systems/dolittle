/*
 * Axis D3: isolated structural verification of the M6 recipe validator.
 * Contract: use only tiny in-memory layouts and a declarative Model catalog;
 * never call a concrete model's recipe generator or execute a forward pass.
 * Test valid schedules, unique sources, immutable externals, exact metadata,
 * selected-logit dimensions and rejection boundaries, not internal slot counts.
 * Algorithm: docs/architecture/transformer-execution.md, "Preparing the first
 * execution recipe": describe a small schedule, validate it, then independently
 * violate one public structural invariant per case. No mathematical equivalence
 * or GPU correctness is inferred from accepting a schedule.
 * Design and cost: bounded fixture metadata, no tensor payloads, files, services
 * or device resources. Optimization: dimension overflow tests allocate no tensors.
 * Corpus consultation, design and implementation, 2026-10-18 (full rules read):
 * cpp-guidelines I.7 (#ri-post): assert public validation outcomes and errors.
 * cpp-guidelines I.4 (#ri-typed): typed tensor, parameter and observation IDs.
 * cpp-perf-guidelines CDSA.32: test the resolved consumer metadata contracts.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#include "models/model.h"
#include "tests/support/check.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using namespace dolittle;
using tests::expect_throws;
using tensors::ElementType;
using models::ObservationBoundary;

constexpr tensors::ComputationMode mode{ElementType::float32, ElementType::float32};
constexpr std::array selected{models::TokenPosition{2}};
constexpr models::SequenceLength length{3};
constexpr kernels::TensorId tokens{0}, embedding_weight{1}, output_weight{2},
    embedded{3}, residual{4}, gathered{5}, logits{6};

tensors::Layout matrix(std::size_t rows, std::size_t width) {
    return tensors::contiguous_layout(ElementType::float32, std::array{rows, width});
}

class DeclaredModel final : public models::Model {
public:
    std::size_t vocabulary = 5;
    models::SequenceLength limit{10};
    std::vector<models::ParameterSpec> specs{
        {"embedding", matrix(5, 2)}, {"output", matrix(5, 2)}};
    std::vector<models::ObservationSite> catalog{
        {{ObservationBoundary::block_input, {0}}, 2, ElementType::float32},
        {{ObservationBoundary::block_output, {0}}, 2, ElementType::float32},
        {{ObservationBoundary::block_input, {1}}, 2, ElementType::float32}};

    std::string_view identity() const noexcept override { return "declarative-test"; }
    std::size_t vocabulary_size() const noexcept override { return vocabulary; }
    models::SequenceLength context_limit() const noexcept override { return limit; }
    std::span<const models::ParameterSpec> parameters() const noexcept override { return specs; }
    std::vector<models::ObservationSite> observation_sites(
        tensors::ComputationMode) const override { return catalog; }
    models::ForwardRecipe full_forward(models::SequenceLength, tensors::ComputationMode,
        std::span<const models::TokenPosition>) const override {
        throw std::logic_error("recipe generation is outside this unit test");
    }
};

models::ForwardRecipe declared_recipe(const DeclaredModel& model) {
    return {
        {tensors::contiguous_layout(ElementType::uint32, std::array{length.value}),
         matrix(5, 2), matrix(5, 2), matrix(3, 2), matrix(3, 2), matrix(1, 2), matrix(1, 5)},
        tokens,
        {{{0}, embedding_weight}, {{1}, output_weight}},
        {kernels::Embedding{tokens, embedding_weight, embedded},
         kernels::Add{embedded, embedded, residual},
         kernels::GatherRows{residual, gathered, {2}},
         kernels::Projection{gathered, output_weight, logits}},
        {{model.catalog[0], embedded}, {model.catalog[1], residual},
         {model.catalog[2], residual}},
        logits};
}

models::ForwardRecipe without_logits(const DeclaredModel& model) {
    auto recipe = declared_recipe(model);
    recipe.operations.resize(2);
    recipe.tensors.resize(5);
    recipe.logits.reset();
    return recipe;
}

template<class Mutation>
void reject_recipe(Mutation mutation) {
    const DeclaredModel model;
    auto recipe = declared_recipe(model);
    mutation(recipe);
    expect_throws<std::invalid_argument>([&] {
        models::validate_recipe(model, recipe, length, mode, selected);
    });
}

template<class Mutation>
void reject_model(Mutation mutation) {
    DeclaredModel model;
    const auto recipe = declared_recipe(model);
    mutation(model);
    expect_throws<std::invalid_argument>([&] {
        models::validate_recipe(model, recipe, length, mode, selected);
    });
}

void valid_schedules() {
    const DeclaredModel model;
    auto recipe = declared_recipe(model);
    models::validate_recipe(model, recipe, length, mode, selected);
    // A leading singleton axis has only index zero, so changing its stride
    // cannot change selected-logit addresses or the logical layout.
    auto singleton_stride = recipe;
    singleton_stride.tensors[logits.value].byte_strides[0] = 0;
    models::validate_recipe(model, singleton_stride, length, mode, selected);
    DeclaredModel singleton_parameter_model;
    singleton_parameter_model.specs.push_back({"singleton-parameter", matrix(1, 2)});
    auto singleton_parameter = declared_recipe(singleton_parameter_model);
    auto parameter_layout = matrix(1, 2);
    parameter_layout.byte_strides[0] = 0;
    singleton_parameter.tensors.push_back(std::move(parameter_layout));
    singleton_parameter.parameters.push_back(
        {{2}, {singleton_parameter.tensors.size() - 1}});
    models::validate_recipe(singleton_parameter_model, singleton_parameter,
                            length, mode, selected);
    // Adjacent boundaries can use distinct completed tensors. Validation proves
    // their structure only; numerical integration must establish equal values.
    auto distinct_boundaries = declared_recipe(model);
    distinct_boundaries.tensors.push_back(matrix(3, 2));
    const kernels::TensorId copied_boundary{distinct_boundaries.tensors.size() - 1};
    distinct_boundaries.operations.push_back(
        kernels::GatherRows{residual, copied_boundary, {0, 1, 2}});
    distinct_boundaries.observations[2].tensor = copied_boundary;
    models::validate_recipe(model, distinct_boundaries, length, mode, selected);
    // Parameter and observation list order is not a contract.
    std::swap(recipe.parameters[0], recipe.parameters[1]);
    std::swap(recipe.observations[0], recipe.observations[2]);
    models::validate_recipe(model, recipe, length, mode, selected);
    models::validate_recipe(model, without_logits(model), length, mode, {});
    // An unobserved intermediate can be unused while still having one source.
    recipe.tensors.push_back(matrix(3, 2));
    recipe.operations.push_back(kernels::Add{embedded, residual,
                                           {recipe.tensors.size() - 1}});
    models::validate_recipe(model, recipe, length, mode, selected);
}

void source_and_layout_contracts() {
    reject_recipe([](auto& r) { std::swap(r.operations[0], r.operations[1]); });
    reject_recipe([](auto& r) { r.operations.push_back(r.operations[0]); });
    reject_recipe([](auto& r) { r.tensors.push_back(matrix(3, 2)); });
    reject_recipe([](auto& r) {
        r.operations.push_back(kernels::Add{embedding_weight, embedding_weight, output_weight});
    });
    reject_recipe([](auto& r) { r.token_input = {r.tensors.size()}; });
    reject_recipe([](auto& r) { r.tensors[tokens.value] = matrix(1, 3); });
    reject_recipe([](auto& r) { r.tensors[tokens.value].shape[0] = 2; });
    reject_recipe([](auto& r) { r.tensors[tokens.value].byte_strides[0] = 8; });
    reject_recipe([](auto& r) { r.tensors.push_back({ElementType::float32, {1}, {3}}); });
    const DeclaredModel model;
    auto overflowing = declared_recipe(model);
    overflowing.tensors.push_back({ElementType::float32,
        {std::numeric_limits<std::size_t>::max()}, {4}});
    expect_throws<std::length_error>([&] {
        models::validate_recipe(model, overflowing, length, mode, selected);
    });
}

void parameter_contracts() {
    reject_recipe([](auto& r) { r.parameters.pop_back(); });
    reject_recipe([](auto& r) { r.parameters.push_back(r.parameters[0]); });
    reject_recipe([](auto& r) { r.parameters[1].parameter = r.parameters[0].parameter; });
    reject_recipe([](auto& r) { r.parameters[1].parameter = {99}; });
    reject_recipe([](auto& r) { r.parameters[1].tensor = r.parameters[0].tensor; });
    reject_recipe([](auto& r) { r.parameters[1].tensor = {r.tensors.size()}; });
    reject_recipe([](auto& r) { r.parameters[0].tensor = tokens; });
    reject_recipe([](auto& r) { r.tensors[output_weight.value] = matrix(2, 5); });
    reject_model([](auto& m) { m.specs[0].role.clear(); });
    reject_model([](auto& m) { m.specs[1].role = m.specs[0].role; });
    reject_model([](auto& m) { m.specs[1].layout.byte_strides.clear(); });
}

void observation_contracts() {
    reject_recipe([](auto& r) { r.observations.pop_back(); });
    reject_recipe([](auto& r) { r.observations[1] = r.observations[0]; });
    reject_recipe([](auto& r) { r.observations[0].site.point.block = {7}; });
    reject_recipe([](auto& r) { r.observations[0].site.vector_width = 3; });
    reject_recipe([](auto& r) { r.observations[0].site.storage_type = ElementType::bfloat16; });
    reject_recipe([](auto& r) { r.observations[0].tensor = gathered; });
    reject_recipe([](auto& r) { r.observations[0].tensor = {r.tensors.size()}; });
    reject_model([](auto& m) { m.catalog[1] = m.catalog[0]; });
    reject_model([](auto& m) { m.catalog[0].vector_width = 0; });
    reject_model([](auto& m) { m.catalog[0].storage_type = ElementType::uint32; });
    reject_model([](auto& m) {
        m.catalog[0].point.boundary = static_cast<ObservationBoundary>(99);
    });
    DeclaredModel model;
    auto recipe = declared_recipe(model);
    model.specs.push_back({"external-shaped-like-activation", matrix(3, 2)});
    recipe.tensors.push_back(matrix(3, 2));
    recipe.parameters.push_back({{2}, {recipe.tensors.size() - 1}});
    recipe.observations[0].tensor = recipe.parameters.back().tensor;
    expect_throws<std::invalid_argument>([&] {
        models::validate_recipe(model, recipe, length, mode, selected);
    });
}

void request_contracts() {
    const DeclaredModel model;
    const auto recipe = declared_recipe(model);
    for (const auto invalid_length : std::array{models::SequenceLength{0},
                                                models::SequenceLength{11}}) {
        expect_throws<std::invalid_argument>([&] {
            models::validate_recipe(model, recipe, invalid_length, mode, selected);
        });
    }
    for (const auto invalid_mode : std::array{
             tensors::ComputationMode{ElementType::uint32, ElementType::float32},
             tensors::ComputationMode{static_cast<ElementType>(99), ElementType::float32},
             tensors::ComputationMode{ElementType::float32, ElementType::bfloat16}}) {
        expect_throws<std::invalid_argument>([&] {
            models::validate_recipe(model, recipe, length, invalid_mode, selected);
        });
    }
    for (const auto& rows : std::array{
             std::vector<models::TokenPosition>{{0}, {0}},
             std::vector<models::TokenPosition>{{2}, {1}},
             std::vector<models::TokenPosition>{{3}}}) {
        expect_throws<std::invalid_argument>([&] {
            models::validate_recipe(model, recipe, length, mode, rows);
        });
    }
    expect_throws<std::invalid_argument>([&] {
        models::validate_recipe(model, recipe, length, mode, {});
    });
    reject_recipe([](auto& r) { r.logits.reset(); });
    reject_recipe([](auto& r) { r.logits = {r.tensors.size()}; });
    reject_recipe([](auto& r) { r.logits = gathered; });
    reject_model([](auto& m) { m.vocabulary = 6; });
    reject_model([](auto& m) { m.vocabulary = 0; });
    reject_model([](auto& m) { m.limit = {0}; });
}

}  // namespace

int main() {
    valid_schedules();
    source_and_layout_contracts();
    parameter_contracts();
    observation_contracts();
    request_contracts();
}
