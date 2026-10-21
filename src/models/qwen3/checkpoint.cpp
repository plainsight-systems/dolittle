/*
 * AR binding, M3: verify an explicit embedding/block/value-head schema against
 * named tensor layouts, then retain the archive and ordered parameter views.
 * The pinned artifact identity alone permits its declared FP32/actual BF16
 * metadata mismatch; cached bytes and tensor precision stay unchanged. The AR
 * model supplies its vector endpoint and omits final norm/vocabulary weights.
 * O(L log L) name matching and O(L) owned metadata; no payload copies.
 * Overlay: steering.md, Checkpoint acquisition and binding. Implementation
 * corpora queried/read 2026-10-20: Core I.7 (#ri-post) for complete schemas;
 * GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md) for explicit upload
 * inputs. Implementation of the reviewed ownership and decoding contracts. Axis
 * M3, Qwen3 kind: configuration parsing and checkpoint parameter binding.
 * Contract: implement pure parse_config/bind_parameters plus bind_target's
 * ownership transfer. Reject unsupported mathematical settings and every
 * missing, duplicate, unexpected or mis-shaped parameter. Distinguish semantic
 * config errors from file integrity errors already handled by the archive.
 * Algorithm: transformer overlay, "Checkpoint loading and binding": obtain
 * verified config bytes; parse bounded JSON with explicit supported/defaulted
 * fields; validate the Qwen3 Config; derive parameter roles and serialized
 * names; validate the complete named layout catalog; bind immutable payload
 * views; publish the owner and versioned identity only after every check. Keep
 * archive/model owners alive until after all borrowed parameter metadata is
 * destroyed. Design and cost: O(blocks) metadata and no tensor copies.
 * Optimization: resolve names into parameter slots once rather than looking up
 * strings during inference. Unit checks use memory JSON/catalogs; real archive
 * opening is integration work. nlohmann/json 3.12.0 uses the format reader's
 * shared duplicate-key/depth/budget preflight. A fixed config allowlist rejects
 * unknown behavioral settings. Model configuration reference: Transformers
 * v4.57.3 configuration_qwen3.py. Overlay:
 * docs/architecture/transformer-execution.md. Corpus consultation,
 * implementation (full rules read), 2026-10-18: cpp-guidelines R.1 (#rr-raii):
 * resource ownership survives exceptions and teardown. cpp-perf-guidelines
 * GPU.1: keep transfers/conversions explicit and bounded. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 *
 * bind_causal_lm shares the strict schema with bind_target, which delegates to
 * it. Full AV weights use the same verified config/schema. Source identity
 * remains available directly from the owned archive; target binding identities
 * are stable. AR shards/value heads remain outside this causal-LM binding
 * contract. Corpus consultation, implementation (both servers queried, full
 * rules read), 2026-10-19: cpp-guidelines R.1 (#rr-raii) ties resources to
 * their owning phase; I.7 (#ri-post) makes completion, provenance and failure
 * effects explicit. cpp-perf-guidelines GPU.1 keeps intermediate embeddings on
 * device and budgets required transfers; source
 * guidelines/gpu/GPU.1-keep-data-on-device.md. cpp-perf-guidelines GPU.9
 * separates persistent weights from sequence storage; source
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 *
 * Archive reuse implementation, 2026-10-19: bind_causal_lm retains shared const
 * ownership of an already verified source. Binding still validates the full
 * schema and owns its model/parameter metadata, but neither reopens files nor
 * hashes or copies payloads. The service and multiple bindings can release
 * their references independently; the last owner releases the mapping.
 * cpp-guidelines F.27 (#rf-shared_ptr): shared_ptr expresses that lifetime
 * contract. cpp-perf-guidelines CDSA.32: reuse immutable preparation across
 * consumers;
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#include "models/qwen3/checkpoint.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include "bounds.h"
#include "tensors/formats/json_metadata.h"

namespace dolittle::models::qwen3 {
namespace {
using tensors::ElementType;
using tensors::formats::detail::Json;

const Json& required(const Json& object, std::string_view key) {
  const auto found = object.find(key);
  if (found == object.end()) {
    throw std::invalid_argument("required Qwen3 config field is missing");
  }
  return *found;
}

void require_string(const Json& value, std::string_view expected) {
  if (!value.is_string() || value.get_ref<const std::string&>() != expected) {
    throw std::invalid_argument("unsupported Qwen3 string setting");
  }
}

void optional_false(const Json& object, std::string_view key) {
  const auto found = object.find(key);
  if (found != object.end() && (!found->is_boolean() || found->get<bool>())) {
    throw std::invalid_argument("unsupported enabled Qwen3 option");
  }
}

double finite_number(const Json& value) {
  if (!value.is_number()) {
    throw std::invalid_argument("Qwen3 constant must be numeric");
  }
  const auto number = value.get<double>();
  if (!std::isfinite(number)) {
    throw std::invalid_argument("Qwen3 constant must be finite");
  }
  return number;
}

std::string serialized_name(std::string_view role) {
  if (role == "token_embedding") {
    return "model.embed_tokens.weight";
  }
  if (role == "reconstruction_head") {
    return "weight";
  }
  if (role == "final_norm") {
    return "model.norm.weight";
  }
  if (role == "vocabulary_projection") {
    return "lm_head.weight";
  }
  constexpr std::string_view prefix = "blocks.";
  if (!role.starts_with(prefix)) {
    throw std::invalid_argument("unsupported Qwen3 parameter role");
  }
  const auto separator = role.find('.', prefix.size());
  if (separator == std::string_view::npos) {
    throw std::invalid_argument("malformed Qwen3 parameter role");
  }
  const auto block = role.substr(prefix.size(), separator - prefix.size());
  const auto suffix = role.substr(separator + 1);
  using NameMapping = std::pair<std::string_view, std::string_view>;
  constexpr std::array mappings{
      NameMapping{"input_norm", "input_layernorm"},
      NameMapping{"query_projection", "self_attn.q_proj"},
      NameMapping{"key_projection", "self_attn.k_proj"},
      NameMapping{"value_projection", "self_attn.v_proj"},
      NameMapping{"query_norm", "self_attn.q_norm"},
      NameMapping{"key_norm", "self_attn.k_norm"},
      NameMapping{"attention_projection", "self_attn.o_proj"},
      NameMapping{"post_attention_norm", "post_attention_layernorm"},
      NameMapping{"gate_projection", "mlp.gate_proj"},
      NameMapping{"up_projection", "mlp.up_proj"},
      NameMapping{"down_projection", "mlp.down_proj"}};
  for (const auto& [semantic, serialized] : mappings) {
    if (suffix == semantic) {
      return "model.layers." + std::string(block) + "." +
             std::string(serialized) + ".weight";
    }
  }
  throw std::invalid_argument("unsupported Qwen3 parameter role");
}

bool same_layout(const tensors::Layout& left, const tensors::Layout& right) {
  (void)tensors::required_bytes(left);
  (void)tensors::required_bytes(right);
  if (left.element_type != right.element_type || left.shape != right.shape) {
    return false;
  }
  for (std::size_t i = 0; i < left.shape.size(); ++i) {
    if (left.shape.at(i) > 1 &&
        left.byte_strides.at(i) != right.byte_strides.at(i)) {
      return false;
    }
  }
  return true;
}

std::string identity_part(std::string_view value) {
  return std::to_string(value.size()) + ":" + std::string(value);
}

class TargetCheckpoint final : public BoundCheckpoint {
 public:
  TargetCheckpoint(
      std::shared_ptr<const tensors::formats::TensorArchive> archive,
      std::unique_ptr<Model> model,
      const std::vector<ParameterBinding>& bindings,
      std::string_view configuration_artifact)
      : archive_(std::move(archive)), model_(std::move(model)) {
    const auto& source = archive_->identity();
    identity_ = "qwen3-target-binding-v1/" + identity_part(source.repository) +
                identity_part(source.revision) +
                identity_part(source.artifact_digest) +
                identity_part(model_->identity()) +
                identity_part(configuration_artifact);
    parameters_.resize(bindings.size());
    std::set<std::string> distinct_sources;
    for (const auto& binding : bindings) {
      const auto view = archive_->tensor(binding.tensor_name);
      tensors::validate_view(view);
      if (!same_layout(*view.layout, dolittle::at(model_->parameters(),
                                                  binding.parameter.value)
                                         .layout)) {
        throw std::invalid_argument(
            "archive payload differs from named parameter layout");
      }
      parameters_.at(binding.parameter.value) = view;
      if (distinct_sources.insert(binding.tensor_name).second) {
        const auto count = tensors::required_bytes(*view.layout);
        if (count >
            std::numeric_limits<std::size_t>::max() - parameter_bytes_) {
          throw std::length_error("parameter bytes overflow");
        }
        parameter_bytes_ += count;
      }
    }
  }
  const Model& model() const noexcept override { return *model_; }
  const tensors::formats::SourceIdentity& source_identity()
      const noexcept override {
    return archive_->identity();
  }
  std::string_view identity() const noexcept override { return identity_; }
  tensors::HostView parameter(ParameterId id) const override {
    if (id.value >= parameters_.size()) {
      throw std::invalid_argument("unknown parameter ID");
    }
    return parameters_.at(id.value);
  }
  std::size_t parameter_bytes() const noexcept override {
    return parameter_bytes_;
  }

 private:
  std::shared_ptr<const tensors::formats::TensorArchive> archive_;
  std::unique_ptr<Model> model_;
  std::vector<tensors::HostView> parameters_;
  std::string identity_;
  std::size_t parameter_bytes_ = 0;
};

}  // namespace

Config parse_config(std::span<const std::byte> bytes,
                    ElementType actual_weight_type,
                    std::size_t max_metadata_bytes) {
  if (actual_weight_type != ElementType::bfloat16 &&
      actual_weight_type != ElementType::float32) {
    throw std::invalid_argument("Qwen3 weights require BF16 or FP32 storage");
  }
  const auto document =
      tensors::formats::detail::parse_metadata(bytes, max_metadata_bytes);
  const auto& root = document.value;
  static const std::set<std::string_view> supported{
      "model_type", "architectures", "num_hidden_layers", "hidden_size",
      "num_attention_heads", "num_key_value_heads", "head_dim",
      "intermediate_size", "vocab_size", "max_position_embeddings",
      "rms_norm_eps", "rope_theta", "hidden_act", "attention_bias", "mlp_bias",
      "tie_word_embeddings", "rope_scaling", "use_sliding_window",
      "sliding_window", "max_window_layers", "layer_types",
      "partial_rotary_factor", "attention_dropout", "torch_dtype", "dtype",
      // These describe provenance, training initialization, generation defaults
      // or caching preference, and do not change this full inference pass.
      "_name_or_path", "_commit_hash", "transformers_version",
      "initializer_range", "bos_token_id", "eos_token_id", "pad_token_id",
      "use_cache"};
  for (const auto& [key, value] : root.items()) {
    (void)value;
    if (!supported.contains(key)) {
      throw std::invalid_argument("unknown Qwen3 config setting: " + key);
    }
  }
  require_string(required(root, "model_type"), "qwen3");
  if (root.contains("architectures")) {
    const auto& architectures = root.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1) {
      throw std::invalid_argument("unsupported Qwen3 architecture list");
    }
    require_string(architectures.at(0), "Qwen3ForCausalLM");
  }
  require_string(required(root, "hidden_act"), "silu");
  for (const auto key : {"attention_bias", "mlp_bias", "tie_word_embeddings",
                         "use_sliding_window"}) {
    optional_false(root, key);
  }
  for (const auto key : {"rope_scaling", "sliding_window"}) {
    if (root.contains(key) && !root.at(key).is_null()) {
      throw std::invalid_argument(
          "scaled RoPE and sliding windows are unsupported");
    }
  }
  if (root.contains("partial_rotary_factor") &&
      finite_number(root.at("partial_rotary_factor")) != 1.0) {
    throw std::invalid_argument("partial rotary embeddings are unsupported");
  }
  if (root.contains("attention_dropout") &&
      finite_number(root.at("attention_dropout")) != 0.0) {
    throw std::invalid_argument(
        "nonzero attention dropout config is unsupported");
  }
  for (const auto key : {"dtype", "torch_dtype"}) {
    if (root.contains(key)) {
      require_string(root.at(key), actual_weight_type == ElementType::bfloat16
                                       ? "bfloat16"
                                       : "float32");
    }
  }
  const auto dimension = [&](std::string_view key) {
    return tensors::formats::detail::size_integer(required(root, key));
  };
  const auto epsilon = finite_number(required(root, "rms_norm_eps"));
  if (epsilon <= 0 || epsilon > std::numeric_limits<float>::max() ||
      static_cast<float>(epsilon) == 0) {
    throw std::invalid_argument("unsupported RMS epsilon");
  }
  Config config{dimension("num_hidden_layers"),
                dimension("hidden_size"),
                dimension("num_attention_heads"),
                dimension("num_key_value_heads"),
                dimension("head_dim"),
                dimension("intermediate_size"),
                dimension("vocab_size"),
                {dimension("max_position_embeddings")},
                static_cast<float>(epsilon),
                finite_number(required(root, "rope_theta")),
                actual_weight_type};
  if (root.contains("layer_types") && !root.at("layer_types").is_null()) {
    const auto& layers = root.at("layer_types");
    if (!layers.is_array() || layers.size() != config.block_count) {
      throw std::invalid_argument(
          "layer type count disagrees with block count");
    }
    for (const auto& layer : layers) {
      require_string(layer, "full_attention");
    }
  }
  if (root.contains("max_window_layers")) {
    (void)tensors::formats::detail::size_integer(root.at("max_window_layers"));
  }
  if (root.contains("use_cache") && !root.at("use_cache").is_boolean()) {
    throw std::invalid_argument("use_cache must be a boolean");
  }
  // Bound model schema expansion before make_model allocates per-block specs.
  constexpr std::size_t estimated_block_metadata =
      11 * (sizeof(ParameterSpec) + 128);
  if (config.block_count > (max_metadata_bytes - document.metadata_bytes) /
                               estimated_block_metadata) {
    throw std::length_error("Qwen3 parameter schema exceeds metadata budget");
  }
  (void)make_model(
      config);  // Same dimension/overflow contract as actual binding.
  return config;
}

namespace {
std::vector<ParameterBinding> bind_schema(
    const Config& config,
    std::span<const tensors::formats::NamedTensor> catalog,
    bool reconstruction) {
  const auto model =
      reconstruction ? make_reconstructor(config) : make_model(config);
  if (catalog.size() != model->parameters().size()) {
    throw std::invalid_argument("Qwen3 checkpoint parameter count mismatch");
  }
  std::map<std::string_view, const tensors::Layout*, std::less<>> named;
  for (const auto& tensor : catalog) {
    (void)tensors::required_bytes(tensor.layout);
    if (!named.emplace(tensor.name, &tensor.layout).second) {
      throw std::invalid_argument("duplicate checkpoint tensor name");
    }
  }
  std::vector<ParameterBinding> bindings;
  bindings.reserve(model->parameters().size());
  for (std::size_t i = 0; i < model->parameters().size(); ++i) {
    const auto& spec = dolittle::at(model->parameters(), i);
    auto name = serialized_name(spec.role);
    const auto found = named.find(name);
    if (found == named.end() || !same_layout(*found->second, spec.layout)) {
      throw std::invalid_argument("missing or incompatible Qwen3 parameter: " +
                                  name);
    }
    bindings.push_back({{i}, std::move(name)});
  }
  return bindings;
}

}  // namespace
std::vector<ParameterBinding> bind_parameters(
    const Config& config,
    std::span<const tensors::formats::NamedTensor> catalog) {
  return bind_schema(config, catalog, false);
}
std::unique_ptr<BoundCheckpoint> bind_reconstructor(
    std::shared_ptr<const tensors::formats::TensorArchive> archive,
    std::string_view configuration_artifact) {
  if (!archive || archive->tensors().empty()) {
    throw std::invalid_argument("empty reconstructor archive");
  }
  const auto actual = archive->tensors().front().layout.element_type;
  const auto& source = archive->identity();
  const bool pinned_dtype =
      source.repository == "marco-bazzani/Qwen3-8B-nla" &&
      source.revision == "12c6ed1b1e684bdb8a788c989e2f3d15f9790db3" &&
      source.artifact_digest ==
          "7eb232a434cb71c18c60e359901aa2ed4fb872d7e317c94bb0ff802e0b69f393" &&
      configuration_artifact == "ar/config.json" &&
      actual == ElementType::bfloat16;
  constexpr std::size_t metadata_limit = 16 * std::size_t{1024} * 1024;
  // This exact verified release serialized BF16 weights with an FP32 config.
  // Validate that declared config, then bind the actual BF16 storage. No bytes
  // or execution precision are converted; other archives require agreement.
  auto config = parse_config(archive->artifact(configuration_artifact),
                             pinned_dtype ? ElementType::float32 : actual,
                             metadata_limit);
  config.weight_type = actual;
  auto bindings = bind_schema(config, archive->tensors(), true);
  return std::make_unique<TargetCheckpoint>(std::move(archive),
                                            make_reconstructor(config),
                                            bindings, configuration_artifact);
}
std::unique_ptr<BoundCheckpoint> bind_causal_lm(
    std::shared_ptr<const tensors::formats::TensorArchive> archive,
    std::string_view configuration_artifact) {
  if (!archive || archive->tensors().empty()) {
    throw std::invalid_argument("empty checkpoint archive");
  }
  const auto actual_type = archive->tensors().front().layout.element_type;
  constexpr std::size_t config_metadata_limit =
      16 * std::size_t{1024} * std::size_t{1024};
  const auto config = parse_config(archive->artifact(configuration_artifact),
                                   actual_type, config_metadata_limit);
  auto bindings = bind_parameters(config, archive->tensors());
  auto model = make_model(config);
  return std::make_unique<TargetCheckpoint>(
      std::move(archive), std::move(model), std::move(bindings),
      configuration_artifact);
}
std::unique_ptr<BoundCheckpoint> bind_target(
    std::unique_ptr<tensors::formats::TensorArchive> archive,
    std::string_view configuration_artifact) {
  return bind_causal_lm(std::move(archive), configuration_artifact);
}
}  // namespace dolittle::models::qwen3
