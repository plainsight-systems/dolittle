/*
 * Axis M6, Qwen3 kind: immutable target-model description and forward recipe.
 * Contract: implement model.h for the declared Qwen3 subset only. Validate positive
 * dimensions, checked products, supported dtype, context bound and positional
 * constants before constructing parameter metadata. Invalid modes never fall back.
 * Algorithm: construct a recipe for the following forward computation.
 * Inputs to recipe construction are the sequence length N, computation mode
 * and requested logit positions. Output is owned tensor/parameter metadata,
 * ordered operations and observation sites; this file does not execute them.
 * At execution, token IDs and bound weights supply the recipe's values.
 *
 * Let D be the residual width, d the head width, Hq and Hkv the query and
 * KV head counts, and F the MLP width. A projection P(X,W) means X * transpose(W);
 * weights are stored [output_features,input_features]. RMSNorm normalizes
 * the final feature dimension and applies its learned scale.
 * 1. Look up each token's embedding to obtain residual X[N,D].
 * 2. For every transformer block, record X as its block_input observation
 *    and construct these operations in dependency order:
 *    a. Normalize X with input_norm to obtain U[N,D].
 *    b. Project U separately into Q[N,Hq,d], K[N,Hkv,d], V[N,Hkv,d].
 *       Normalize each Q and K head with query_norm/key_norm, then apply
 *       rotary position embedding to Q and K at positions [0,N).
 *    c. Apply causal grouped-query attention to Q, K and V to obtain
 *       A[N,Hq,d]. Flatten its head axes for attention_projection, then
 *       add the projection to X: R = X + P(A, attention_projection).
 *    d. Normalize R with post_attention_norm to obtain S[N,D]. Project S
 *       into gate G[N,F] and up B[N,F]. Compute M = SiLU(G) * B elementwise.
 *    e. Project M back to D features and add the second residual:
 *       X_next = R + P(M, down_projection). Record X_next as block_output
 *       and use it as X for the following block.
 * 3. After all blocks, gather the requested rows of X, apply final_norm,
 *    then vocabulary_projection. With C requested rows and vocabulary
 *    size Vocab, logits have shape [C,Vocab]. An empty selection omits
 *    this branch.
 * Observation requests never stop the block loop. Operation contracts in
 * kernels/operations.h define storage rounding at each numerical boundary.
 * Overlay: docs/architecture/transformer-execution.md,
 * "One Qwen3 transformer block" and "First forward pass".
 *
 * Incremental construction uses the same block equations for Bmax new rows.
 * Each attention declares its rotated-K and projected-V append sources. The
 * last-row gather marks the optional logit tail. Runtime S/B supplies absolute
 * RoPE positions and causal cache visibility; no old Q/K/V is recomputed.
 * Design and cost: O(blocks+selected rows) metadata; widths come from Config, not
 * the selected checkpoint's constants. Optimization: reuse the second norm, keep
 * KV heads grouped, and do not project unused logit rows. Store each mathematical
 * value once in the recipe; physical aliasing is the executor's responsibility.
 * Unit checks cover advertised shapes/sites/configuration contracts. Full output
 * equivalence uses the pinned reference in an integration scenario; exact private
 * instruction ordering/counts are not unit assertions.
 * Overlay: docs/architecture/transformer-execution.md.
 * Parameter roles are documented in model.h; no checkpoint names appear here.
 * Projection retains leading N and flattens trailing axes, allowing Q/K/V
 * tensors to have [N,H,d] while their weights remain [H*d,D].
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit): explicit inputs and no hidden execution state.
 * cpp-guidelines I.4 (#ri-typed): distinct parameter/tensor IDs and typed ops.
 * cpp-guidelines I.7 (#ri-post): complete block sites and owned recipes.
 * cpp-guidelines ES.103 (#res-overflow): guard metadata/dimension products.
 * cpp-perf-guidelines GPU.9: prepare storage once and preserve last-use lifetimes.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines CDSA.32: resolve semantic roles into ordered recipe inputs.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 *
 * The lookup in step 1 now declares its output and producer as EmbeddingBoundary
 * in full and incremental recipes. Executor replacement precedes block-0 input
 * observations and RMSNorm; all block equations and absolute positions are shared.
 * Corpus consultation, implementation (both servers queried, full rules read),
 * 2026-10-19: cpp-guidelines R.1 (#rr-raii) ties resources to their owning phase;
 * I.7 (#ri-post) makes completion, provenance and failure effects explicit.
 * cpp-perf-guidelines GPU.1 keeps intermediate embeddings on device and budgets
 * required transfers; source guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.9 separates persistent weights from sequence storage;
 * source guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "bounds.h"
#include "models/qwen3/model.h"

#include <cmath>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>

namespace dolittle::models::qwen3 {
namespace {
inline constexpr std::size_t parameters_per_block = 11;
inline constexpr std::size_t tensors_per_block = 32;
inline constexpr std::size_t global_tensor_capacity = 8;
inline constexpr std::size_t operations_per_block = 19;

using kernels::TensorId;
using tensors::ElementType;

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::length_error("Qwen3 dimension product overflows");
    }
    return a * b;
}

struct MetadataCounts {
    std::size_t per_block;
    std::size_t other;
};
std::size_t metadata_count(std::size_t blocks, MetadataCounts counts) {
    const auto [per_block, other] = counts;
    const auto count = checked_product(blocks, per_block);
    if (other > std::numeric_limits<std::size_t>::max() - count) {
        throw std::length_error("Qwen3 metadata count overflows");
    }
    return count + other;
}

tensors::Layout layout(ElementType type, std::initializer_list<std::size_t> shape) {
    return tensors::contiguous_layout(type, {shape.begin(), shape.size()});
}

void validate_config(const Config &config) {
    if (config.block_count == 0 || config.residual_width == 0 || config.query_heads == 0 ||
        config.kv_heads == 0 || config.head_width == 0 || config.mlp_width == 0 ||
        config.vocabulary_size == 0 || config.context_limit.value == 0 ||
        config.query_heads % config.kv_heads != 0 || config.head_width % 2 != 0 ||
        !std::isfinite(config.rms_epsilon) || config.rms_epsilon <= 0 || !std::isfinite(config.rope_theta) ||
        config.rope_theta <= 0 ||
        (config.weight_type != ElementType::bfloat16 && config.weight_type != ElementType::float32) ||
        config.vocabulary_size - 1 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("unsupported Qwen3 configuration");
    }
    (void)checked_product(config.query_heads, config.head_width);
    (void)checked_product(config.kv_heads, config.head_width);
    (void)metadata_count(config.block_count, {.per_block = parameters_per_block, .other = 3});
    (void)metadata_count(config.block_count, {.per_block = 2, .other = 0});
}

struct BlockParameters {
    ParameterId input_norm;
    ParameterId query_projection;
    ParameterId key_projection;
    ParameterId value_projection;
    ParameterId query_norm;
    ParameterId key_norm;
    ParameterId attention_projection;
    ParameterId post_attention_norm;
    ParameterId gate_projection;
    ParameterId up_projection;
    ParameterId down_projection;
};

class Qwen3Model final : public Model {
  public:
    explicit Qwen3Model(Config config) : config_(config) {
        validate_config(config_);
        const auto q_width = checked_product(config_.query_heads, config_.head_width);
        const auto kv_width = checked_product(config_.kv_heads, config_.head_width);
        // Check representative weight extents before allocating per-block metadata.
        (void)layout(config_.weight_type, {config_.vocabulary_size, config_.residual_width});
        (void)layout(config_.weight_type, {q_width, config_.residual_width});
        (void)layout(config_.weight_type, {kv_width, config_.residual_width});
        (void)layout(config_.weight_type, {config_.mlp_width, config_.residual_width});
        parameters_.reserve(
            metadata_count(config_.block_count, {.per_block = parameters_per_block, .other = 3}));
        blocks_.reserve(config_.block_count);
        embedding_ = parameter("token_embedding", {config_.vocabulary_size, config_.residual_width});
        final_norm_ = parameter("final_norm", {config_.residual_width});
        vocabulary_ = parameter("vocabulary_projection", {config_.vocabulary_size, config_.residual_width});
        for (std::size_t i = 0; i < config_.block_count; ++i) {
            const auto prefix = "blocks." + std::to_string(i) + ".";
            blocks_.push_back({
                parameter(prefix + "input_norm", {config_.residual_width}),
                parameter(prefix + "query_projection", {q_width, config_.residual_width}),
                parameter(prefix + "key_projection", {kv_width, config_.residual_width}),
                parameter(prefix + "value_projection", {kv_width, config_.residual_width}),
                parameter(prefix + "query_norm", {config_.head_width}),
                parameter(prefix + "key_norm", {config_.head_width}),
                parameter(prefix + "attention_projection", {config_.residual_width, q_width}),
                parameter(prefix + "post_attention_norm", {config_.residual_width}),
                parameter(prefix + "gate_projection", {config_.mlp_width, config_.residual_width}),
                parameter(prefix + "up_projection", {config_.mlp_width, config_.residual_width}),
                parameter(prefix + "down_projection", {config_.residual_width, config_.mlp_width}),
            });
        }
    }

    std::string_view identity() const noexcept override { return "qwen3-target/full-forward-v1"; }
    std::size_t vocabulary_size() const noexcept override { return config_.vocabulary_size; }
    SequenceLength context_limit() const noexcept override { return config_.context_limit; }
    std::span<const ParameterSpec> parameters() const noexcept override { return parameters_; }

    std::vector<ObservationSite> observation_sites(tensors::ComputationMode mode) const override {
        validate_mode(mode);
        std::vector<ObservationSite> sites;
        sites.reserve(metadata_count(config_.block_count, {.per_block = 2, .other = 0}));
        for (std::size_t i = 0; i < config_.block_count; ++i) {
            sites.push_back(site(ObservationBoundary::block_input, i));
            sites.push_back(site(ObservationBoundary::block_output, i));
        }
        return sites;
    }

    ForwardRecipe full_forward(SequenceLength length, tensors::ComputationMode mode,
                               std::span<const TokenPosition> positions) const override {
        return build_forward(length, mode, positions, nullptr);
    }
    IncrementalRecipe incremental_forward(SequenceLength length,
                                          tensors::ComputationMode mode) const override {
        if (!length.value) {
            throw std::invalid_argument("empty incremental chunk");
        }
        IncrementalRecipe result{};
        const std::array<TokenPosition, 1> positions{{{length.value - 1}}};
        result.forward = build_forward(length, mode, positions, &result);
        return result;
    }

  private:
    ForwardRecipe build_forward(SequenceLength length, tensors::ComputationMode mode,
                                std::span<const TokenPosition> positions,
                                IncrementalRecipe *incremental) const {
        validate_mode(mode);
        if (length.value == 0 || length.value > config_.context_limit.value) {
            throw std::invalid_argument("Qwen3 sequence outside context limit");
        }
        for (std::size_t i = 0; i < positions.size(); ++i) {
            if (dolittle::at(positions, i).value >= length.value ||
                (i != 0 && dolittle::at(positions, i - 1).value >= dolittle::at(positions, i).value)) {
                throw std::invalid_argument("Qwen3 logit positions must increase within sequence");
            }
        }
        ForwardRecipe recipe{};
        recipe.tensors.reserve(metadata_count(
            config_.block_count, {.per_block = tensors_per_block, .other = global_tensor_capacity}));
        recipe.operations.reserve(
            metadata_count(config_.block_count, {.per_block = operations_per_block, .other = 4}));
        recipe.parameters.reserve(parameters_.size());
        recipe.observations.reserve(metadata_count(config_.block_count, {.per_block = 2, .other = 0}));
        const auto tensor = [&](tensors::Layout value) {
            const TensorId id{recipe.tensors.size()};
            recipe.tensors.push_back(std::move(value));
            return id;
        };
        const auto activation = [&](std::initializer_list<std::size_t> shape) {
            return tensor(layout(mode.activation_storage, shape));
        };
        recipe.token_input = tensor(layout(ElementType::uint32, {length.value}));
        std::vector<TensorId> weights;
        weights.reserve(parameters_.size());
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            const auto id = tensor(parameters_.at(i).layout);
            recipe.parameters.push_back({{i}, id});
            weights.push_back(id);
        }
        const auto norm = [&](TensorId input, ParameterId weight) {
            const auto output = tensor(recipe.tensors.at(input.value));
            recipe.operations.emplace_back(
                kernels::RmsNorm{input, weights.at(weight.value), output, config_.rms_epsilon});
            return output;
        };
        const auto project = [&](TensorId input, ParameterId weight,
                                 std::initializer_list<std::size_t> shape) {
            const auto output = activation(shape);
            recipe.operations.emplace_back(kernels::Projection{input, weights.at(weight.value), output});
            return output;
        };
        const auto rotary = [&](TensorId input) {
            const auto output = tensor(recipe.tensors.at(input.value));
            recipe.operations.emplace_back(kernels::Rotary{input, output, 0, config_.rope_theta});
            return output;
        };
        const auto add = [&](TensorId left, TensorId right) {
            const auto output = tensor(recipe.tensors.at(left.value));
            recipe.operations.emplace_back(kernels::Add{left, right, output});
            return output;
        };

        const auto n = length.value;
        const auto d = config_.residual_width;
        const auto h = config_.head_width;
        auto residual = activation({n, d});
        recipe.embedding_boundary = EmbeddingBoundary{residual, recipe.operations.size()};
        recipe.operations.emplace_back(
            kernels::Embedding{recipe.token_input, weights.at(embedding_.value), residual});
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            const auto &block = blocks_.at(i);
            recipe.observations.push_back({site(ObservationBoundary::block_input, i), residual});
            const auto normalized = norm(residual, block.input_norm);
            auto query = project(normalized, block.query_projection, {n, config_.query_heads, h});
            auto key = project(normalized, block.key_projection, {n, config_.kv_heads, h});
            const auto value = project(normalized, block.value_projection, {n, config_.kv_heads, h});
            query = rotary(norm(query, block.query_norm));
            key = rotary(norm(key, block.key_norm));
            const auto attention = activation({n, config_.query_heads, h});
            if (incremental) {
                incremental->cache_appends.push_back({{i}, recipe.operations.size(), key, value});
            }
            recipe.operations.emplace_back(kernels::CausalAttention{query, key, value, attention});
            residual = add(residual, project(attention, block.attention_projection, {n, d}));
            const auto post_attention = norm(residual, block.post_attention_norm);
            const auto gate = project(post_attention, block.gate_projection, {n, config_.mlp_width});
            const auto up = project(post_attention, block.up_projection, {n, config_.mlp_width});
            const auto activated = activation({n, config_.mlp_width});
            recipe.operations.emplace_back(kernels::SiluMultiply{gate, up, activated});
            residual = add(residual, project(activated, block.down_projection, {n, d}));
            recipe.observations.push_back({site(ObservationBoundary::block_output, i), residual});
        }
        if (incremental) {
            incremental->logits_begin = recipe.operations.size();
        }
        if (!positions.empty()) {
            std::vector<std::size_t> rows;
            rows.reserve(positions.size());
            for (const auto position : positions) {
                rows.push_back(position.value);
            }
            const auto selected = activation({positions.size(), d});
            recipe.operations.emplace_back(kernels::GatherRows{residual, selected, std::move(rows)});
            const auto normalized = norm(selected, final_norm_);
            recipe.logits = project(normalized, vocabulary_, {positions.size(), config_.vocabulary_size});
        }
        return recipe;
    }

  private:
    ParameterId parameter(std::string role, std::initializer_list<std::size_t> shape) {
        const ParameterId id{parameters_.size()};
        parameters_.push_back({std::move(role), layout(config_.weight_type, shape)});
        return id;
    }

    void validate_mode(tensors::ComputationMode mode) const {
        if (mode.activation_storage != config_.weight_type ||
            mode.reduction_accumulation != ElementType::float32) {
            throw std::invalid_argument("unsupported Qwen3 computation mode");
        }
    }

    ObservationSite site(ObservationBoundary boundary, std::size_t block) const {
        return {{boundary, {block}}, config_.residual_width, config_.weight_type};
    }

    Config config_;
    std::vector<ParameterSpec> parameters_;
    std::vector<BlockParameters> blocks_;
    ParameterId embedding_{};
    ParameterId final_norm_{};
    ParameterId vocabulary_{};
};
} // namespace

std::unique_ptr<Model> make_model(const Config &config) { return std::make_unique<Qwen3Model>(config); }
} // namespace dolittle::models::qwen3
