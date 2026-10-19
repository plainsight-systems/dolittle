/*
 * Axis D3: one real checkpoint archive loading/binding integrity workflow.
 * Contract: a private temporary directory owns tiny safetensors/config/index
 * fixtures. Exercise actual file verification, read-only mapped payloads, Qwen3
 * binding and owner lifetime; corrupt request integrity or index/path metadata
 * must prevent publication. No model inference, GPU or subprocess is launched.
 * All created paths are owned by the directory and removed during unwinding.
 * Algorithm: transformer overlay, "Checkpoint loading and binding": serialize a
 * complete tiny target, compute independent expected SHA-256 via OpenSSL EVP,
 * open and bind it, check source/payload ownership, then reject integrity and
 * source-containment failures through the same production loading entry point.
 * Design/cost: one bounded disk checkpoint workflow, O(fixture bytes) setup.
 * Optimization: tiny deterministic weights avoid pretrained downloads and still
 * exercise real mappings. This is one integration scenario, never a unit test.
 * Corpus consultation, implementation, 2026-10-18 (full rules read this phase):
 * cpp-guidelines R.1 (#rr-raii): temporary resources and archives own teardown.
 * cpp-guidelines I.7 (#ri-post): only verified, completely bound owners escape.
 * cpp-perf-guidelines CDSA.32: check consumer bindings after metadata resolution.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#include "bounds.h"
#include "models/qwen3/checkpoint.h"
#include "tensors/formats/safetensors.h"
#include "tests/support/check.h"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace tensors = dolittle::tensors;
namespace models = dolittle::models;
namespace tests = dolittle::tests;
namespace format = tensors::formats::safetensors;
using tests::check;
using tests::expect_throws;
using Json = nlohmann::json;

class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        std::random_device entropy;
        for (int attempt = 0; attempt < 100; ++attempt) {
            path_ = std::filesystem::temp_directory_path() /
                    ("dolittle-archive-" + std::to_string(entropy()) + "-" + std::to_string(entropy()));
            if (std::filesystem::create_directory(path_)) {
                return;
            }
        }
        throw std::runtime_error("cannot create a unique archive fixture directory");
    }
    TemporaryDirectory(TemporaryDirectory &&) = delete;
    TemporaryDirectory &operator=(TemporaryDirectory &&) = delete;
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;
    const std::filesystem::path &path() const noexcept { return path_; }

  private:
    std::filesystem::path path_;
};

std::vector<std::byte> bytes(std::string_view value) {
    const auto span = std::as_bytes(std::span(value.data(), value.size()));
    return {span.begin(), span.end()};
}

void write_file(const std::filesystem::path &path, std::span<const std::byte> data) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    // Native API byte/address boundary; no unrelated typed object is dereferenced.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    stream.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    stream.close();
}

std::string sha256(std::span<const std::byte> data) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int count = 0;
    check(EVP_Digest(data.data(), data.size(), digest.data(), &count, EVP_sha256(), nullptr) == 1 &&
              count == 32,
          "fixture digest succeeds");
    constexpr std::string_view hex = "0123456789abcdef";
    std::string result;
    for (unsigned int i = 0; i < count; ++i) {
        result += hex.at(digest.at(i) >> 4);
        result += hex.at(digest.at(i) & 15);
    }
    return result;
}

struct Fixture {
    std::vector<std::byte> shard;
    std::vector<std::byte> config;
    Json index;
    std::size_t parameter_bytes;
};

Fixture fixture() {
    Json header = Json::object();
    Json weight_map = Json::object();
    std::vector<std::byte> payload;
    const auto tensor = [&](std::string_view name, std::initializer_list<std::size_t> shape) {
        std::size_t elements = 1;
        for (const auto dimension : shape) {
            elements *= dimension;
        }
        const auto begin = payload.size();
        for (std::size_t i = 0; i < elements; ++i) {
            payload.insert(payload.end(), {std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0x3f}});
        }
        header.emplace(std::string(name), Json{{"dtype", "F32"},
                                               {"shape", std::vector<std::size_t>(shape)},
                                               {"data_offsets", {begin, payload.size()}}});
        weight_map.emplace(std::string(name), "weights.safetensors");
    };
    tensor("model.embed_tokens.weight", {7, 4});
    tensor("model.norm.weight", {4});
    tensor("lm_head.weight", {7, 4});
    tensor("model.layers.0.input_layernorm.weight", {4});
    tensor("model.layers.0.post_attention_layernorm.weight", {4});
    tensor("model.layers.0.self_attn.q_proj.weight", {4, 4});
    tensor("model.layers.0.self_attn.k_proj.weight", {2, 4});
    tensor("model.layers.0.self_attn.v_proj.weight", {2, 4});
    tensor("model.layers.0.self_attn.o_proj.weight", {4, 4});
    tensor("model.layers.0.self_attn.q_norm.weight", {2});
    tensor("model.layers.0.self_attn.k_norm.weight", {2});
    tensor("model.layers.0.mlp.gate_proj.weight", {6, 4});
    tensor("model.layers.0.mlp.up_proj.weight", {6, 4});
    tensor("model.layers.0.mlp.down_proj.weight", {4, 6});
    auto serialized_header = header.dump();
    while (serialized_header.size() % 8 != 0) {
        serialized_header += ' ';
    }
    std::vector<std::byte> shard(8);
    const auto header_size = static_cast<std::uint64_t>(serialized_header.size());
    for (std::size_t i = 0; i < 8; ++i) {
        shard.at(i) = std::byte((header_size >> (8 * i)) & 255);
    }
    const auto header_bytes = bytes(serialized_header);
    shard.insert(shard.end(), header_bytes.begin(), header_bytes.end());
    shard.insert(shard.end(), payload.begin(), payload.end());
    auto config = bytes(
        R"({"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_act":"silu","num_hidden_layers":1,"hidden_size":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"intermediate_size":6,"vocab_size":7,"max_position_embeddings":16,"rms_norm_eps":0.000001,"rope_theta":10000,"torch_dtype":"float32","attention_bias":false,"tie_word_embeddings":false,"rope_scaling":null,"use_sliding_window":false})");
    return {std::move(shard),
            std::move(config),
            {{"metadata", {{"total_size", payload.size()}}}, {"weight_map", std::move(weight_map)}},
            payload.size()};
}

format::OpenRequest request_for(const std::filesystem::path &root, const Fixture &data) {
    const auto index = bytes(data.index.dump());
    return {root,
            "test/tiny-qwen3",
            "fixture-v1",
            {{"weights.safetensors", data.shard.size(), sha256(data.shard)},
             {"config.json", data.config.size(), sha256(data.config)},
             {"model.safetensors.index.json", index.size(), sha256(index)}},
            {"weights.safetensors"},
            "model.safetensors.index.json",
            std::size_t{1024} * std::size_t{1024},
            std::size_t{1024} * std::size_t{1024},
            100};
}

void checkpoint_loading_workflow() {
    const TemporaryDirectory temporary;
    const auto root = temporary.path() / "checkpoint";
    std::filesystem::create_directory(root);
    auto data = fixture();
    write_file(root / "weights.safetensors", data.shard);
    write_file(root / "config.json", data.config);
    write_file(root / "model.safetensors.index.json", bytes(data.index.dump()));
    const auto valid_request = request_for(root, data);
    {
        // Request strings are temporary; source identity and mapped bytes belong
        // to the returned archive, which is then consumed by the bound owner.
        auto archive = [&] { return format::open(request_for(root, data)); }();
        check(archive->artifact("config.json").size() == data.config.size(), "verified metadata exposed");
        const auto retained = archive->tensor("model.embed_tokens.weight");
        check(retained.bytes.size() == std::size_t{7} * 4 * 4 &&
                  dolittle::at(retained.bytes, 3) == std::byte{0x3f},
              "mapped parameter bits match the file");
        auto reversed_request = valid_request;
        std::reverse(reversed_request.artifacts.begin(), reversed_request.artifacts.end());
        const auto reordered = format::open(reversed_request);
        check(archive->identity().artifact_digest == reordered->identity().artifact_digest,
              "artifact identity does not depend on request order");
        auto checkpoint = models::qwen3::bind_target(std::move(archive), "config.json");
        check(checkpoint->parameter_bytes() == data.parameter_bytes, "bound parameter bytes accounted");
        check(checkpoint->model().vocabulary_size() == 7, "verified config bound to model");
        check(dolittle::at(retained.bytes, 3) == std::byte{0x3f},
              "mapping remains live through archive ownership transfer");
        for (std::size_t i = 0; i < checkpoint->model().parameters().size(); ++i) {
            const auto view = checkpoint->parameter({i});
            tensors::validate_view(view);
            check(dolittle::at(view.bytes, 3) == std::byte{0x3f}, "bound views retain mapped payload");
        }
    }
    auto wrong_hash = valid_request;
    wrong_hash.artifacts.at(0).sha256.assign(64, '0');
    expect_throws<std::runtime_error>([&] { (void)format::open(wrong_hash); });
    auto wrong_size = valid_request;
    ++wrong_size.artifacts.at(0).byte_size;
    expect_throws<std::runtime_error>([&] { (void)format::open(wrong_size); });

    // The index is itself correctly hashed, but disagrees with the shard that
    // actually owns one tensor; it must fail the loading workflow.
    data.index.at("weight_map").at("model.norm.weight") = "other.safetensors";
    write_file(root / "model.safetensors.index.json", bytes(data.index.dump()));
    expect_throws<std::invalid_argument>([&] { (void)format::open(request_for(root, data)); });
    data.index.at("weight_map").at("model.norm.weight") = "weights.safetensors";
    write_file(root / "model.safetensors.index.json", bytes(data.index.dump()));

    auto escaping_path = valid_request;
    escaping_path.artifacts.at(0).relative_path = "../outside.safetensors";
    escaping_path.shards.at(0) = "../outside.safetensors";
    expect_throws<std::invalid_argument>([&] { (void)format::open(escaping_path); });
    write_file(temporary.path() / "outside.safetensors", data.shard);
    std::filesystem::create_symlink(temporary.path() / "outside.safetensors", root / "escape.safetensors");
    auto escaping_link = valid_request;
    escaping_link.artifacts.at(0).relative_path = "escape.safetensors";
    escaping_link.shards.at(0) = "escape.safetensors";
    expect_throws<std::invalid_argument>([&] { (void)format::open(escaping_link); });

    // Rejected attempts leave no partial published owner or damaged source;
    // the original checkpoint remains loadable through the production path.
    auto final = models::qwen3::bind_target(format::open(valid_request), "config.json");
    check(final->parameter_bytes() == data.parameter_bytes, "valid checkpoint remains loadable");
}
} // namespace

int main() try { checkpoint_loading_workflow(); } catch (const std::exception &error) {
    return dolittle::tests::report_failure(error.what());
} catch (...) {
    return dolittle::tests::report_failure("unexpected non-standard exception");
}
