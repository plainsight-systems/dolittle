/*
 * Axis M2/M3: internal bounded JSON decoding shared by archive and model metadata.
 * Contract: UTF-8 JSON input is bounded before tokenization. A SAX preflight
 * rejects duplicate keys at every depth, nesting beyond 64, malformed numbers,
 * and conservative node/string accounting beyond the supplied metadata budget.
 * The returned DOM owns its data. Invalid JSON -> invalid_argument; budget ->
 * length_error; allocation failures propagate. This is no exact allocator/RSS
 * limit: parser scratch is bounded by input length, containers by charged nodes.
 * Algorithm: transformer overlay, "Checkpoint loading and binding": bound input,
 * validate every key and charge metadata before constructing the DOM, then hand
 * the immutable object to the format/model-specific semantic decoder.
 * Design: two O(input bytes) parser passes with O(metadata) temporary ownership.
 * Optimization: parse only setup metadata, never tensor payload bytes.
 * Corpus consultation, implementation, 2026-10-18 (full rules read):
 * cpp-guidelines R.1 (#rr-raii): parser state unwinds through owning containers.
 * cpp-guidelines I.7 (#ri-post): successful documents have unique keys.
 * cpp-perf-guidelines CDSA.32: validate metadata once before consumer binding.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include <nlohmann/json.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace dolittle::tensors::formats::detail {
using Json = nlohmann::json;

inline void charge(std::size_t& used, std::size_t amount, std::size_t limit) {
    if (used > limit || amount > limit - used) {
        throw std::length_error("metadata budget exceeded");
    }
    used += amount;
}

class JsonPreflight final : public nlohmann::json_sax<Json> {
public:
    explicit JsonPreflight(std::size_t limit) : limit_(limit) {}
    bool null() override { return node(); }
    bool boolean(bool) override { return node(); }
    bool number_integer(number_integer_t) override { return node(); }
    bool number_unsigned(number_unsigned_t) override { return node(); }
    bool number_float(number_float_t value, const string_t&) override {
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite JSON number");
        return node();
    }
    bool string(string_t& value) override {
        node(); charge(used_, value.size(), limit_); return true;
    }
    bool binary(binary_t&) override { throw std::invalid_argument("binary is not JSON metadata"); }
    bool start_object(std::size_t) override {
        node(); push(true); return true;
    }
    bool key(string_t& value) override {
        charge(used_, sizeof(std::string) + sizeof(void*) * 4, limit_);
        charge(used_, value.size(), limit_);
        if (!stack_.back().keys.insert(value).second) {
            throw std::invalid_argument("duplicate JSON key");
        }
        return true;
    }
    bool end_object() override { stack_.pop_back(); return true; }
    bool start_array(std::size_t) override { node(); push(false); return true; }
    bool end_array() override { stack_.pop_back(); return true; }
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&) override {
        throw std::invalid_argument("malformed JSON metadata");
    }
    std::size_t used() const noexcept { return used_; }
private:
    struct Frame { bool object; std::unordered_set<std::string> keys; };
    bool node() { charge(used_, sizeof(Json) + sizeof(void*) * 4, limit_); return true; }
    void push(bool object) {
        if (stack_.size() >= 64) throw std::length_error("JSON nesting limit exceeded");
        stack_.push_back({object, {}});
    }
    std::size_t limit_;
    std::size_t used_ = 0;
    std::vector<Frame> stack_;
};

struct JsonDocument { Json value; std::size_t metadata_bytes; };

inline JsonDocument parse_metadata(std::span<const std::byte> input, std::size_t limit) {
    if (input.size() > limit) throw std::length_error("JSON input exceeds metadata budget");
    if (input.empty()) throw std::invalid_argument("empty JSON metadata");
    const auto* begin = reinterpret_cast<const char*>(input.data());
    JsonPreflight check(limit - input.size());
    if (!Json::sax_parse(begin, begin + input.size(), &check)) {
        throw std::invalid_argument("malformed JSON metadata");
    }
    auto value = Json::parse(begin, begin + input.size());
    if (!value.is_object()) throw std::invalid_argument("JSON metadata root must be an object");
    return {std::move(value), input.size() + check.used()};
}

inline std::uint64_t unsigned_integer(const Json& value) {
    if (!value.is_number_integer()) throw std::invalid_argument("expected unsigned JSON integer");
    if (value.is_number_unsigned()) return value.get<std::uint64_t>();
    const auto signed_value = value.get<std::int64_t>();
    if (signed_value < 0) throw std::invalid_argument("negative JSON dimension or offset");
    return static_cast<std::uint64_t>(signed_value);
}

inline std::size_t size_integer(const Json& value) {
    const auto number = unsigned_integer(value);
    if (number > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("JSON size exceeds host address range");
    }
    return static_cast<std::size_t>(number);
}
}  // namespace dolittle::tensors::formats::detail
