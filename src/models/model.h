/*
 * Shared model contract and structural recipe validation; concrete equations
 * are separate. Axis M6: model equations, parameter schema and observable
 * values. No backend handles, filenames, tokenizer behavior or NLA-pair
 * constants. Contract: each Model owns immutable parameter specs indexed by
 * ParameterId. Names are semantic model-local roles, not serialized tensor
 * names. Identity names architecture and recipe version, not a particular
 * checkpoint's bytes. full_forward receives positive N within context_limit,
 * supported numerical mode, and strictly increasing in-range selected logit
 * positions (possibly empty). Return an owned, finite ordered recipe; never
 * retain caller spans. Each tensor slot has a unique producer or one external
 * input binding. Parameters and token input may not be written. Reads refer
 * only to external values or earlier outputs. All slots have valid layouts and
 * checked extents. Observation values cover exactly the advertised sites and
 * identify their completed tensors; output for block i equals input for i+1
 * where applicable. These two site names may refer to shared or distinct
 * tensors; value equality requires numerical integration and is not checked by
 * comparing tensor IDs. If logits are unrequested, logits is absent, but every
 * block still executes. Advertised sites vary with the requested storage mode;
 * the recipe/catalog must agree exactly. Invalid configuration/request ->
 * invalid_argument; size overflow -> length_error; host allocation failure
 * propagates. No GPU work occurs here. validate_recipe checks those structural
 * contracts against the model and requested shape/mode/rows without executing
 * the recipe. It checks token input type/length, parameter layouts,
 * read-before-write, unique producers, exact advertised observation sites and
 * selected-logit shape. Singleton-axis strides may differ when comparing
 * equivalent layouts; stride units and addressable extents are still validated.
 * It cannot prove the model equations or the numerical contents/order of
 * selected logit rows: independent numerical integration provides that
 * evidence. Structural success establishes a well-formed schedule, not a
 * correct transformer computation.
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
 * Corpus consultation, implementation (full rules read), 2026-10-18:
 * cpp-guidelines I.4 (#ri-typed): parameter/tensor/observation IDs are
 * distinct. cpp-guidelines I.7 (#ri-post): single producers and complete sites
 * explicit. cpp-perf-guidelines CDSA.32: model metadata is resolved into
 * execution order. Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 *
 * Incremental model contract: an incremental recipe is an
 * additional explicit capability, not full_forward on a repeatedly growing
 * sequence. The Model supplies per-block KV shapes and identifies produced
 * rotated-K/unrotated-V values, cache reads and semantic observation tensors.
 * Input shape is a bounded new-token span; runtime state supplies committed
 * prefix length S and active row count B. The executor binds cache resources,
 * but must not discover K/V producers by operation order or parameter names.
 *
 * Recipe validation must distinguish immutable parameters, old cache prefix,
 * new cache writes and ordinary single-producer intermediates. Each block
 * writes exactly [S,S+B), then attention may read [0,S+i] for query i. Reject
 * missing, duplicate, incompatible or reordered append/read declarations.
 * Validation alone does not commit cache validity; completion remains the
 * executor's job. IncrementalRecipe owns the ordinary bounded ForwardRecipe
 * plus CacheAppend entries naming each attention step and its
 * rotated-K/unrotated-V tensors. logits_begin marks the last-active-row gather
 * and optional output tail. The default incremental capability rejects
 * unsupported model kinds. All shape-dependent metadata is prepared once, not
 * rebuilt per token. Overlay: transformer-execution.md, "Prefill and decoding".
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): incremental capability and
 * external cache effects must be declared in the recipe and validated.
 * cpp-perf-guidelines CDSA.32: resolve producer/cache bindings into execution
 * order. Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 *
 * Embedding-input extension (implemented, 2026-10-19):
 * An optional EmbeddingBoundary names the embedding lookup's output tensor and
 * producer step explicitly. The tensor is contiguous [rows,D] in activation
 * storage; the named operation is its sole producer. Every consumer and any
 * block-input observation must occur after that step. A replacement is an
 * executor input applied after lookup and before those consumers, not another
 * model equation or an inferred "first operation" convention. Recipe validation
 * will reject an inconsistent boundary before allocation. Absence means this
 * recipe does not support embedding replacement. Qwen3 full and incremental
 * recipes publish and validate this capability. For chunk prefix S and active
 * rows B, a replacement at absolute position p affects local row p-S exactly
 * when S <= p < S+B. RoPE still uses absolute positions. Observation and cache
 * dependencies see the replaced residual. Cost is one O(1) descriptor per
 * recipe. Optimization: resolve its consumer boundary once, with no model-name
 * checks during execution. Overlay: transformer-execution.md, "NLA entry and
 * exit paths". Corpus consultation, header design and implementation (full
 * rules read), 2026-10-19: cpp-guidelines I.7 (#ri-post): explicit
 * producer/consumer ordering contract. cpp-perf-guidelines GPU.1: keep the
 * embedding matrix device-resident and replace only a row; source
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "kernels/operations.h"
#include "models/observation.h"

namespace dolittle::models {

struct ParameterId {
  std::size_t value;
};
struct ParameterSpec {
  std::string role;
  tensors::Layout layout;
};
struct ParameterInput {
  ParameterId parameter;
  kernels::TensorId tensor;
};
struct ObservationValue {
  ObservationSite site;
  kernels::TensorId tensor;
};
struct EmbeddingBoundary {
  kernels::TensorId tensor;
  std::size_t producer_step;
};
struct ForwardRecipe {
  std::vector<tensors::Layout> tensors;
  kernels::TensorId token_input;
  std::vector<ParameterInput> parameters;
  std::vector<kernels::Operation> operations;
  std::vector<ObservationValue> observations;
  std::optional<kernels::TensorId> logits;
  std::optional<EmbeddingBoundary> embedding_boundary{};
};

struct CacheAppend {
  BlockIndex block;
  std::size_t attention_step;
  kernels::TensorId rotated_key, value;
};
struct IncrementalRecipe {
  ForwardRecipe forward;
  std::vector<CacheAppend> cache_appends;
  std::size_t logits_begin;
};

class Model {
 public:
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;
  virtual ~Model();
  [[nodiscard]] virtual std::string_view identity() const noexcept = 0;
  [[nodiscard]] virtual std::size_t vocabulary_size() const noexcept = 0;
  [[nodiscard]] virtual SequenceLength context_limit() const noexcept = 0;
  [[nodiscard]] virtual std::span<const ParameterSpec> parameters()
      const noexcept = 0;
  [[nodiscard]] virtual std::vector<ObservationSite> observation_sites(
      tensors::ComputationMode mode) const = 0;
  [[nodiscard]] virtual ForwardRecipe full_forward(
      SequenceLength length, tensors::ComputationMode mode,
      std::span<const TokenPosition> logit_positions) const = 0;
  [[nodiscard]] virtual IncrementalRecipe incremental_forward(
      SequenceLength max_chunk, tensors::ComputationMode mode) const;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

 protected:
  Model();
};

void validate_recipe(const Model& model, const ForwardRecipe& recipe,
                     SequenceLength length, tensors::ComputationMode mode,
                     std::span<const TokenPosition> logit_positions);

void validate_incremental_recipe(const Model&, const IncrementalRecipe&,
                                 SequenceLength max_chunk,
                                 tensors::ComputationMode);
}  // namespace dolittle::models
