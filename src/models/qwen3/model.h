/*
 * Header design; Qwen3 recipe generation is not implemented.
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
 *
 * Algorithm: docs/architecture/transformer-execution.md, "One Qwen3 transformer
 * block" and "First forward pass": lookup embeddings; for each block emit
 * input RMSNorm, Q/K/V projections, Q/K per-head RMSNorm, Q/K RoPE, causal GQA,
 * output projection/residual sum, post-attention RMSNorm once, gate/up
 * projections, SiLU*up, down projection and second residual sum. Advertise the
 * entering and final residual of every block. Gather selected final rows,
 * apply final RMSNorm and vocabulary projection; omit only that output branch
 * when no logits are requested. Never stop at an observation block.
 *
 * Cost: O(L) operations/parameter slots plus selected-row metadata. Dense FLOPs
 * follow the overlay; no expanded KV heads or required N*N score tensor.
 * Optimization: one post-attention normalization reused by both MLP branches;
 * only requested rows reach vocabulary projection. Physical reuse belongs to
 * executor planning, not to changing model equations or observation values.
 *
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.4 (#ri-typed): model dimensions are separate from tensor IDs.
 * cpp-guidelines I.7 (#ri-post): ordered equations and capture semantics explicit.
 * cpp-perf-guidelines CDSA.32: generate the immutable recipe once per workload.
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
