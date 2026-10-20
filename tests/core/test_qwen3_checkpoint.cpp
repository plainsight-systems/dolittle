/*
 * Axis D3: isolated M3 configuration() and binding contracts for Qwen3 targets.
 * Contract: use bounded JSON, an explicitly named tiny tensor catalog, and an
 * in-memory archive owner. No filesystem, inference, subprocess or device work.
 * Check supported semantics, complete binding, layout/dtype rejection,
 * ownership transfer and borrowed payload lifetime; never assert parser calls
 * or slot order. Algorithm: transformer overlay, "Checkpoint loading and
 * binding": parse config, resolve declared tensor names, expose completed owned
 * bindings, and unwind failed ownership transfers. The in-memory archive does
 * not test artifact verification. Design/cost: one block and small byte
 * vectors; no pretrained model download. Optimization: exercise metadata
 * contracts without allocating model-sized data. Corpus consultation,
 * implementation, 2026-10-18 (full rules read): cpp-guidelines I.7 (#ri-post):
 * assert complete bindings and failure categories. cpp-guidelines R.1
 * (#rr-raii): count final owner destruction on success/failure.
 * cpp-perf-guidelines CDSA.32: resolve serialized names into semantic model
 * roles. Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 * Shared binding ownership, design/implementation 2026-10-19: two bindings and
 * their source owner can die independently; payloads remain valid until the
 * last binding releases the archive. Failed binding preserves other source
 * owners. cpp-guidelines F.27 (#rf-shared_ptr) defines this shared lifetime
 * invariant; cpp-perf-guidelines CDSA.32 supports reusing one immutable
 * prepared source.
 */
#include <algorithm>
#include <array>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>

#include "bounds.h"
#include "models/qwen3/checkpoint.h"
#include "tests/support/check.h"

namespace {
namespace tensors = dolittle::tensors;
namespace models = dolittle::models;
namespace tests = dolittle::tests;
namespace qwen = models::qwen3;
using tensors::ElementType;
using tests::check;
using tests::expect_throws;
constexpr std::size_t metadata_limit = std::size_t{1024} * std::size_t{1024};

const std::string& configuration() {
  static const std::string value =
      R"({"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_act":"silu","num_hidden_layers":1,"hidden_size":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"intermediate_size":6,"vocab_size":7,"max_position_embeddings":16,"rms_norm_eps":0.000001,"rope_theta":10000,"torch_dtype":"float32","attention_bias":false,"tie_word_embeddings":false,"rope_scaling":null,"sliding_window":null,"use_sliding_window":false,"use_cache":true})";
  return value;
}

std::span<const std::byte> bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}

qwen::Config config() {
  return qwen::parse_config(bytes(configuration()), ElementType::float32,
                            metadata_limit);
}

std::string replace(std::string text, std::string_view before,
                    std::string_view after) {
  const auto position = text.find(before);
  check(position != std::string::npos, "test fixture replacement exists");
  text.replace(position, before.size(), after);
  return text;
}

std::vector<tensors::formats::NamedTensor> catalog() {
  std::vector<tensors::formats::NamedTensor> result;
  const auto add = [&](std::string name,
                       std::initializer_list<std::size_t> shape) {
    result.push_back(
        {std::move(name),
         tensors::contiguous_layout(ElementType::float32,
                                    std::span(shape.begin(), shape.size()))});
  };
  add("model.embed_tokens.weight", {7, 4});
  add("model.norm.weight", {4});
  add("lm_head.weight", {7, 4});
  add("model.layers.0.input_layernorm.weight", {4});
  add("model.layers.0.post_attention_layernorm.weight", {4});
  add("model.layers.0.self_attn.q_proj.weight", {4, 4});
  add("model.layers.0.self_attn.k_proj.weight", {2, 4});
  add("model.layers.0.self_attn.v_proj.weight", {2, 4});
  add("model.layers.0.self_attn.o_proj.weight", {4, 4});
  add("model.layers.0.self_attn.q_norm.weight", {2});
  add("model.layers.0.self_attn.k_norm.weight", {2});
  add("model.layers.0.mlp.gate_proj.weight", {6, 4});
  add("model.layers.0.mlp.up_proj.weight", {6, 4});
  add("model.layers.0.mlp.down_proj.weight", {4, 6});
  return result;
}

void config_contracts() {
  const auto parsed = config();
  check(parsed.block_count == 1 && parsed.residual_width == 4 &&
            parsed.query_heads == 2 && parsed.kv_heads == 1 &&
            parsed.head_width == 2 && parsed.mlp_width == 6 &&
            parsed.vocabulary_size == 7 && parsed.context_limit.value == 16,
        "declared dimensions retained");
  const auto bf16 = replace(configuration(), "float32", "bfloat16");
  check(qwen::parse_config(bytes(bf16), ElementType::bfloat16, metadata_limit)
                .weight_type == ElementType::bfloat16,
        "actual weight representation retained");
  for (const auto& [before, after] :
       std::initializer_list<std::pair<std::string_view, std::string_view>>{
           std::pair<std::string_view, std::string_view>{"\"qwen3\"",
                                                         "\"qwen2\""},
           {"\"silu\"", "\"gelu\""},
           {"\"attention_bias\":false", "\"attention_bias\":true"},
           {"\"tie_word_embeddings\":false", "\"tie_word_embeddings\":true"},
           {"\"rope_scaling\":null",
            "\"rope_scaling\":{\"rope_type\":\"yarn\"}"},
           {"\"sliding_window\":null", "\"sliding_window\":8"},
           {"\"head_dim\":2", "\"head_dim\":3"},
           {"\"num_key_value_heads\":1", "\"num_key_value_heads\":3"},
           {"\"num_hidden_layers\":1", "\"num_hidden_layers\":-1"},
           {"\"hidden_size\":4", "\"hidden_size\":4.0"},
           {"\"rope_theta\":10000", "\"rope_theta\":0"},
           {"\"rms_norm_eps\":0.000001", "\"rms_norm_eps\":-1"},
           {"\"float32\"", "\"bfloat16\""},
           {"\"use_cache\":true",
            "\"use_cache\":true,\"unknown_behavior\":true"},
           {"\"use_cache\":true", "\"use_cache\":true,\"head_dim\":2"},
           {"\"use_cache\":true",
            "\"use_cache\":true,\"layer_types\":[\"sliding_attention\"]"},
           {"\"use_cache\":true",
            "\"use_cache\":true,\"_name_or_path\":{\"x\":1,\"x\":2}"}}) {
    const auto invalid = replace(configuration(), before, after);
    expect_throws<std::invalid_argument>([&] {
      (void)qwen::parse_config(bytes(invalid), ElementType::float32,
                               metadata_limit);
    });
  }
  expect_throws<std::length_error>([] {
    (void)qwen::parse_config(bytes(configuration()), ElementType::float32, 1);
  });
  const auto massive = replace(configuration(), "\"num_hidden_layers\":1",
                               "\"num_hidden_layers\":100000000");
  expect_throws<std::length_error>([&] {
    (void)qwen::parse_config(bytes(massive), ElementType::float32,
                             metadata_limit);
  });
}

void binding_contracts() {
  const auto settings = config();
  auto named = catalog();
  const auto bindings = qwen::bind_parameters(settings, named);
  check(bindings.size() == named.size(), "all supplied parameters bound");
  const auto model = qwen::make_model(settings);
  std::map<std::string, std::string> roles;
  for (const auto& binding : bindings) {
    roles.emplace(
        dolittle::at(model->parameters(), binding.parameter.value).role,
        binding.tensor_name);
  }
  check(roles.at("token_embedding") == "model.embed_tokens.weight" &&
            roles.at("vocabulary_projection") == "lm_head.weight" &&
            roles.at("blocks.0.query_norm") ==
                "model.layers.0.self_attn.q_norm.weight" &&
            roles.at("blocks.0.down_projection") ==
                "model.layers.0.mlp.down_proj.weight",
        "serialized names match semantic roles");
  std::reverse(named.begin(), named.end());
  check(qwen::bind_parameters(settings, named).size() == bindings.size(),
        "catalog order independent");
  for (int kind = 0; kind < 6; ++kind) {
    auto invalid = catalog();
    switch (kind) {
      default:
        throw std::logic_error("invalid malformed-metadata case");
      case 0:
        invalid.pop_back();
        break;
      case 1:
        invalid.push_back(invalid.front());
        break;
      case 2:
        invalid.at(1).name = invalid.at(0).name;
        break;
      case 3:
        invalid.at(0).name = "unexpected.weight";
        break;
      case 4:
        invalid.at(0).layout = tensors::contiguous_layout(
            ElementType::float32, std::array<std::size_t, 2>{4, 7});
        break;
      case 5:
        invalid.at(0).layout = tensors::contiguous_layout(
            ElementType::bfloat16, std::array<std::size_t, 2>{7, 4});
        break;
    }
    expect_throws<std::invalid_argument>(
        [&] { (void)qwen::bind_parameters(settings, invalid); });
  }
}

class MemoryArchive final : public tensors::formats::TensorArchive {
 public:
  explicit MemoryArchive(int& destroyed, bool missing = false)
      : destroyed_(&destroyed), named_(catalog()) {
    if (missing) {
      named_.pop_back();
    }
    for (const auto& tensor : named_) {
      payloads_.emplace(tensor.name, std::vector<std::byte>(
                                         tensors::required_bytes(tensor.layout),
                                         std::byte{0x3f}));
    }
  }
  MemoryArchive(const MemoryArchive&) = delete;
  MemoryArchive& operator=(const MemoryArchive&) = delete;
  MemoryArchive(MemoryArchive&&) = delete;
  MemoryArchive& operator=(MemoryArchive&&) = delete;
  ~MemoryArchive() override { ++*destroyed_; }
  const tensors::formats::SourceIdentity& identity() const noexcept override {
    return identity_;
  }
  std::span<const tensors::formats::NamedTensor> tensors()
      const noexcept override {
    return named_;
  }
  tensors::HostView tensor(std::string_view name) const override {
    for (const auto& item : named_) {
      if (item.name == name) {
        return {&item.layout, payloads_.at(item.name)};
      }
    }
    throw std::invalid_argument("unknown memory tensor");
  }
  std::span<const std::byte> artifact(std::string_view name) const override {
    if (name != "config.json") {
      throw std::invalid_argument("unknown memory artifact");
    }
    return bytes(configuration());
  }

 private:
  int* destroyed_;
  tensors::formats::SourceIdentity identity_{"test/repository", "revision",
                                             "digest"};
  std::vector<tensors::formats::NamedTensor> named_;
  std::map<std::string, std::vector<std::byte>> payloads_;
};

void ownership_contracts() {
  int destroyed = 0;
  {
    auto checkpoint = qwen::bind_target(
        std::make_unique<MemoryArchive>(destroyed), "config.json");
    check(destroyed == 0, "checkpoint retains archive owner");
    std::size_t total = 0;
    for (std::size_t i = 0; i < checkpoint->model().parameters().size(); ++i) {
      const auto view = checkpoint->parameter({i});
      tensors::validate_view(view);
      total += tensors::required_bytes(*view.layout);
      check(view.bytes.front() == std::byte{0x3f},
            "borrowed payload remains live");
    }
    check(checkpoint->parameter_bytes() == total, "parameter byte accounting");
    check(!checkpoint->identity().empty(), "bound provenance exists");
    expect_throws<std::invalid_argument>([&] {
      (void)checkpoint->parameter({checkpoint->model().parameters().size()});
    });
  }
  check(destroyed == 1, "checkpoint destroys archive exactly once");
  expect_throws<std::invalid_argument>([&] {
    (void)qwen::bind_target(std::make_unique<MemoryArchive>(destroyed, true),
                            "config.json");
  });
  check(destroyed == 2, "failed binding releases consumed archive");
  expect_throws<std::invalid_argument>(
      [] { (void)qwen::bind_target(nullptr, "config.json"); });
}
void shared_ownership_contracts() {
  int destroyed = 0;
  auto source = std::make_shared<MemoryArchive>(destroyed);
  auto first = qwen::bind_causal_lm(source, "config.json");
  auto second = qwen::bind_causal_lm(source, "config.json");
  source.reset();
  first.reset();
  check(destroyed == 0, "remaining binding retains shared source");
  const auto view = second->parameter({0});
  tensors::validate_view(view);
  check(view.bytes.front() == std::byte{0x3f}, "shared payload remains live");
  second.reset();
  check(destroyed == 1, "last binding releases shared source once");
  auto invalid = std::make_shared<MemoryArchive>(destroyed, true);
  expect_throws<std::invalid_argument>(
      [&] { (void)qwen::bind_causal_lm(invalid, "config.json"); });
  check(destroyed == 1 && !invalid->tensors().empty(),
        "failed binding preserves external archive owner");
  invalid.reset();
  check(destroyed == 2, "external owner releases failed binding source");
}
}  // namespace

int main() try {
  config_contracts();
  binding_contracts();
  ownership_contracts();
  shared_ownership_contracts();
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
