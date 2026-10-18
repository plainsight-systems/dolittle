/*
 * Header design; Qwen3 config parsing and parameter binding are pending.
 * Axis M3, Qwen3 kind. Only this binder knows serialized Qwen3 tensor names.
 * Contract: bind_target takes ownership of an already verified archive and
 * reads configuration from its named artifact, never an unverified side file.
 * Parse architecture, dimensions, context, epsilon, RoPE, activation, bias and
 * tying settings. Accept only qwen3/model.h semantics: reject unsupported
 * scaling, windowed attention, biases, tied heads, quantized weights or other
 * computation-affecting config. Validate all required tensors and account for
 * every supplied tensor; unexpected tensors fail for this target-only binder.
 * Informational config keys may be ignored explicitly; unknown behavioral
 * settings must not silently select an approximation. No tokenizer work here.
 *
 * Bind model.embed_tokens.weight and lm_head.weight as [V,D], model.norm.weight
 * as [D]. Per model.layers.i: input_layernorm/post_attention_layernorm [D];
 * self_attn q_proj [Hq*d,D], k_proj/v_proj [Hkv*d,D], o_proj [D,Hq*d], q_norm
 * and k_norm [d]; mlp gate_proj/up_proj [F,D], down_proj [D,F] (all .weight).
 * Shapes and dense orientation must match; all required weights use one
 * supported floating storage type. No transpose guessing or synthetic weights.
 * Return a stable owner of model/archive/bindings with identity including the
 * version of this binding convention. Invalid config/tensors -> invalid_argument;
 * checked overflow -> length_error; allocation failure propagates and releases
 * the consumed archive. A failed call does not return ownership to the caller.
 * parse_config consumes bounded in-memory JSON plus the actual weight dtype;
 * a declared dtype must agree. It returns owned Config without files/GPU work.
 * bind_parameters consumes Config and named layouts, validates the complete
 * catalog, and returns owned names in the parameter slots of make_model(config).
 * These pure helpers are shared with bind_target, so unit checks do not need
 * archives, model downloads or mock scripts reproducing loader call order.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Checkpoint loading
 * and binding": read verified config, construct the Qwen3 model description,
 * match and validate each role's payload, then publish BoundCheckpoint.
 * Cost: O(L) name/shape validation and metadata, no payload copies or GPU work.
 * Optimization: store parameter bindings in model-slot order; never resolve
 * serialized names inside forward execution. Full parameter bytes are counted
 * before the executor admits a resident plan.
 *
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): consumed archive released on binding failure.
 * cpp-guidelines I.7 (#ri-post): every required parameter is validated.
 * cpp-perf-guidelines CDSA.32: bind once in the model consumer's slot order.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include "models/checkpoint.h"
#include "models/qwen3/model.h"
#include "tensors/formats/format.h"
#include <memory>
#include <string_view>

namespace dolittle::models::qwen3 {
struct ParameterBinding {
    ParameterId parameter;
    std::string tensor_name;
};
[[nodiscard]] Config parse_config(
    std::span<const std::byte> json, tensors::ElementType actual_weight_type,
    std::size_t max_metadata_bytes);
[[nodiscard]] std::vector<ParameterBinding> bind_parameters(
    const Config& config, std::span<const tensors::formats::NamedTensor> catalog);
[[nodiscard]] std::unique_ptr<BoundCheckpoint> bind_target(
    std::unique_ptr<tensors::formats::TensorArchive> archive,
    std::string_view configuration_artifact);
}  // namespace dolittle::models::qwen3
