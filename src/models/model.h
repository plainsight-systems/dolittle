/*
 * Header design; recipes and model implementations are pending.
 * Axis M6: model equations, parameter schema and observable values. No backend
 * handles, filenames, tokenizer behavior or NLA-pair constants.
 * Contract: each Model owns immutable parameter specs indexed by ParameterId.
 * Names are semantic model-local roles, not serialized tensor names. Identity
 * names architecture and recipe version, not a particular checkpoint's bytes.
 * full_forward receives positive N within context_limit, supported numerical
 * mode, and strictly increasing in-range selected logit positions (possibly
 * empty). Return an owned, finite ordered recipe; never retain caller spans.
 * Each tensor slot has a unique producer or one external input binding.
 * Parameters and token input may not be written. Reads refer only to external
 * values or earlier outputs. All slots have valid layouts and checked extents.
 * Observation values cover exactly the advertised sites and identify their
 * completed tensors; output for block i equals input for i+1 where applicable.
 * These two site names may refer to the same tensor. If logits are unrequested,
 * logits is absent, but every block still executes. Advertised sites vary with
 * the requested storage mode; the recipe/catalog must agree exactly.
 * Invalid configuration/request -> invalid_argument; size overflow ->
 * length_error; host allocation failure propagates. No GPU work occurs here.
 * validate_recipe checks those structural contracts against the model and
 * requested shape/mode/rows without executing the recipe. It checks token
 * input type/length, parameter layouts, read-before-write, unique producers,
 * exact advertised observation sites and selected-logit shape. It cannot prove
 * the model equations: independent numerical integration provides that evidence.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Full model flow" and
 * "Preparing the first execution recipe": emit embedding, the model's ordered
 * block operations and observation values, then requested output computation.
 * Executor validates recipe dependencies, resolves copies, assigns storage and
 * selects kernels. It does not derive equations from architecture names.
 * Cost: O(blocks + requested logit rows) recipe metadata, independent of token
 * values. Optimization: prepare once per shape/selection; no virtual calls per
 * element; the Model virtual interface is used only during preparation.
 * This is not a general graph compiler.
 *
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.4 (#ri-typed): parameter/tensor/observation IDs are distinct.
 * cpp-guidelines I.7 (#ri-post): single producers and complete sites explicit.
 * cpp-perf-guidelines CDSA.32: model metadata is resolved into execution order.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include "models/observation.h"
#include "kernels/operations.h"
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dolittle::models {

struct ParameterId { std::size_t value; };
struct ParameterSpec { std::string role; tensors::Layout layout; };
struct ParameterInput { ParameterId parameter; kernels::TensorId tensor; };
struct ObservationValue { ObservationSite site; kernels::TensorId tensor; };
struct ForwardRecipe {
    std::vector<tensors::Layout> tensors;
    kernels::TensorId token_input;
    std::vector<ParameterInput> parameters;
    std::vector<kernels::Operation> operations;
    std::vector<ObservationValue> observations;
    std::optional<kernels::TensorId> logits;
};

class Model {
public:
    virtual ~Model();
    [[nodiscard]] virtual std::string_view identity() const noexcept = 0;
    [[nodiscard]] virtual std::size_t vocabulary_size() const noexcept = 0;
    [[nodiscard]] virtual SequenceLength context_limit() const noexcept = 0;
    [[nodiscard]] virtual std::span<const ParameterSpec> parameters() const noexcept = 0;
    [[nodiscard]] virtual std::vector<ObservationSite> observation_sites(
        tensors::ComputationMode mode) const = 0;
    [[nodiscard]] virtual ForwardRecipe full_forward(
        SequenceLength length, tensors::ComputationMode mode,
        std::span<const TokenPosition> logit_positions) const = 0;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
protected:
    Model();
};

void validate_recipe(
    const Model& model, const ForwardRecipe& recipe, SequenceLength length,
    tensors::ComputationMode mode, std::span<const TokenPosition> logit_positions);

}  // namespace dolittle::models
