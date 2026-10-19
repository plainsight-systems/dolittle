/*
 * Qwen3 config decoding and target parameter binding.
 * Axis M3, Qwen3 kind. Only this binder knows serialized Qwen3 tensor names.
 * Contract: bind_target takes ownership of an already verified archive and
 * reads configuration from its named artifact, never an unverified side file.
 * Parse architecture, dimensions, context, epsilon, RoPE, activation, bias and
 * tying settings. Accept only qwen3/model.h semantics: reject unsupported
 * scaling, windowed attention, biases, tied heads, quantized weights or other
 * computation-affecting config. Validate all required tensors and account for
 * every supplied tensor; unexpected tensors fail for this target-only binder.
 * Informational provenance, initialization, token IDs and cache-preference keys
 * are explicitly allowlisted; unknown behavioral
 * settings must not silently select an approximation. No tokenizer work here.
 *
 * Bind model.embed_tokens.weight and lm_head.weight as [V,D], model.norm.weight
 * as [D]. Per model.layers.i: input_layernorm/post_attention_layernorm [D];
 * self_attn q_proj [Hq*d,D], k_proj/v_proj [Hkv*d,D], o_proj [D,Hq*d], q_norm
 * and k_norm [d]; mlp gate_proj/up_proj [F,D], down_proj [D,F] (all .weight).
 * Shapes and dense orientation must match; all required weights use one
 * supported floating storage type. No transpose guessing or synthetic weights.
 * Return a stable owner of model/archive/bindings with identity including the
 * version of this binding convention. Invalid config/tensors ->
 * invalid_argument; checked overflow -> length_error; allocation failure
 * propagates and releases the consumed archive. A failed call does not return
 * ownership to the caller. parse_config consumes bounded in-memory JSON plus
 * the actual weight dtype; a declared dtype must agree. It returns owned Config
 * without files/GPU work. Core dimensions, head_dim, activation, epsilon and
 * theta must be explicit; absent bias/tying/window flags mean false.
 * bind_target uses a 16 MiB config metadata budget including bounded
 * source/node/string and schema accounting. bind_parameters consumes Config and
 * named layouts, validates the complete catalog, and returns owned names in the
 * parameter slots of make_model(config). These pure helpers are shared with
 * bind_target, so unit checks do not need archives, model downloads or mock
 * scripts reproducing loader call order.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Checkpoint loading
 * and binding": read verified config, construct the Qwen3 model description,
 * match and validate each role's payload, then publish BoundCheckpoint.
 * Cost: O(L log L) indexed name validation and O(L) metadata; no payload copies
 * or GPU work.
 * Optimization: store parameter bindings in model-slot order; never resolve
 * serialized names inside forward execution. Full parameter bytes are counted
 * before the executor admits a resident plan.
 *
 * Corpus consultation, implementation (full rules read), 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): consumed archive released on binding failure.
 * cpp-guidelines I.7 (#ri-post): every required parameter is validated.
 * cpp-perf-guidelines CDSA.32: bind once in the model consumer's slot order.
 * Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 *
 * Full-AV binding (implemented, 2026-10-19): the selected AV's config
 * is byte-identical to the target config and uses the same complete parameter
 * schema. Use bind_causal_lm as the role-neutral entry point; bind_target
 * remains a compatibility entry with the same validation and binding identity.
 * Do not change existing target identity/fixtures merely to rename this entry
 * point. The caller selects only the full AV's shards, not ar/ shards. Both
 * paths validate every tensor as above. NLA source/prompt compatibility is
 * separately enforced by Pair before upload. AR/value-head binding is still
 * unsupported. Algorithm and costs remain the verified Qwen3 binder's steps
 * above; do not duplicate the loader or permit unknown tensors for the AV.
 * Optimization: both roles share one schema/binding implementation and
 * immutable payload views. Corpus consultation, header design and
 * implementation (full rules read), 2026-10-19: cpp-guidelines R.1 (#rr-raii),
 * I.7 (#ri-post): consumed archive owns all views; successful binding is
 * complete regardless of the caller's model role. cpp-perf-guidelines GPU.1:
 * retain mapped views for bounded weight upload; source
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include <memory>
#include <string_view>

#include "models/checkpoint.h"
#include "models/qwen3/model.h"
#include "tensors/formats/format.h"

namespace dolittle::models::qwen3 {
[[nodiscard]] std::unique_ptr<BoundCheckpoint> bind_causal_lm(
    std::unique_ptr<tensors::formats::TensorArchive> archive,
    std::string_view configuration_artifact);
struct ParameterBinding {
  ParameterId parameter;
  std::string tensor_name;
};
[[nodiscard]] Config parse_config(std::span<const std::byte> json,
                                  tensors::ElementType actual_weight_type,
                                  std::size_t max_metadata_bytes);
[[nodiscard]] std::vector<ParameterBinding> bind_parameters(
    const Config& config,
    std::span<const tensors::formats::NamedTensor> catalog);
[[nodiscard]] std::unique_ptr<BoundCheckpoint> bind_target(
    std::unique_ptr<tensors::formats::TensorArchive> archive,
    std::string_view configuration_artifact);
}  // namespace dolittle::models::qwen3
