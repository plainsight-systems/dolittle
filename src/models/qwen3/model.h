/*
 * Axis M6, Qwen3 kind: target-style full causal forward mathematics.
 * Contract: Config describes the supported dense, biasless, untied-vocabulary
 * output kind with Q/K head RMSNorm, SiLU MLP and unscaled split-half RoPE.
 * Dimensions/context/vocabulary are positive, Hq % Hkv == 0, head width is
 * even, eps/theta positive finite, products checked, vocabulary fits uint32
 * IDs. Head projection width is Hq*d, not implicitly residual_width. Weight
 * type is BF16 or FP32; execution activation type must equal it, with FP32
 * reductions. Reject unsupported modes rather than silently converting weights.
 * Other Qwen variants, embedding injection and AR's output path require later
 * explicit extensions. This factory does not claim support for them.
 * make_model owns copied config and model metadata; invalid config ->
 * invalid_argument, overflow -> length_error, allocation failure propagates.
 * Model::full_forward contracts apply, including empty selected logits.
 * Parameter roles are token_embedding, final_norm, vocabulary_projection and
 * blocks.<zero-based index>.{input_norm,query_projection,key_projection,
 * value_projection,query_norm,key_norm,attention_projection,
 * post_attention_norm,gate_projection,up_projection,down_projection}.
 * These semantic roles are the binder contract; numeric slot IDs are not.
 * Vocabulary count minus one must fit uint32, so every possible ID fits.
 *
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
 * Cost: O(L) operations/parameter slots for L transformer blocks, plus
 * selected-row metadata. Dense FLOPs follow the overlay; no expanded KV heads or required N*N score tensor.
 * Optimization: one post-attention normalization reused by both MLP branches;
 * only requested rows reach vocabulary projection. Physical reuse belongs to
 * executor planning, not to changing model equations or observation values.
 *
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.4 (#ri-typed): model dimensions are separate from tensor IDs.
 * cpp-guidelines I.7 (#ri-post): ordered equations and capture semantics explicit.
 * cpp-guidelines ES.103 (#res-overflow): check dimension and metadata extents.
 * cpp-perf-guidelines CDSA.32: generate the immutable recipe once per workload.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 *
 * Incremental Qwen3 recipe: use the same block equations for
 * B new rows beginning at absolute position S. Q and K RoPE use S+i rather
 * than restarting at zero. In each block, expose rotated K and unrotated V as
 * append sources; causal attention uses the persistent prefix plus current rows.
 * Only queries/residuals for the new B rows are computed. The selected final
 * new row produces next-token logits; captures label absolute positions S+i.
 * The cache description comes from block_count, kv_heads and head_width. No
 * assumption about block 24, fixed Hkv or a contiguous device pointer belongs
 * in the mathematical recipe. Backend preparation selects a supported view.
 * The zero-prefix case is ordinary causal prefill; chunked prefill and decode
 * must agree with full_forward on identical consumed tokens within fixed
 * numerical bounds, including BF16 intermediate rounding.
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): absolute position and unchanged block equations.
 * cpp-perf-guidelines CDSA.32: prepare chunk/decode recipe shapes once.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include "models/model.h"
#include <cstddef>
#include <memory>

namespace dolittle::models::qwen3 {

struct Config {
    std::size_t block_count;
    std::size_t residual_width;
    std::size_t query_heads;
    std::size_t kv_heads;
    std::size_t head_width;
    std::size_t mlp_width;
    std::size_t vocabulary_size;
    SequenceLength context_limit;
    float rms_epsilon;
    double rope_theta;
    tensors::ElementType weight_type;
};

[[nodiscard]] std::unique_ptr<Model> make_model(const Config& config);

}  // namespace dolittle::models::qwen3
