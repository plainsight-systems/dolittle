/*
 * Axis D3: isolated M2 safetensors header contract verification.
 * Contract: direct in-memory parsing only; no files, hashes, archives or devices.
 * Check typed extents, complete payload coverage, unique keys, owned metadata
 * and failure categories. No assertions about parser internals or diagnostics.
 * Algorithm: transformer overlay, "Checkpoint loading and binding": decode a
 * bounded header, inspect layouts/ranges, reject malformed or excessive metadata.
 * Design/cost: tiny JSON strings; huge dimensions appear only in rejection cases.
 * Optimization: no weight payload allocation is necessary to check header bounds.
 * Corpus consultation, implementation, 2026-10-18 (full rules read):
 * cpp-guidelines I.7 (#ri-post): assertions exercise the public parse contract.
 * cpp-guidelines R.1 (#rr-raii): outputs outlive caller-owned JSON bytes.
 * cpp-perf-guidelines CDSA.32: compare decoded logical shape/range metadata.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#include "tensors/formats/safetensors.h"
#include "tests/support/check.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace dolittle;
namespace format = tensors::formats::safetensors;
using tests::check;
using tests::expect_throws;
constexpr format::HeaderLimits limits{1024 * 1024, 1024 * 1024, 100};

std::span<const std::byte> bytes(std::string_view text) {
    return std::as_bytes(std::span(text.data(), text.size()));
}

void valid_headers() {
    std::string text = R"({"z":{"dtype":"BF16","shape":[2,3],"data_offsets":[4,16]},"a":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"empty":{"dtype":"U32","shape":[9,0],"data_offsets":[16,16]},"__metadata__":{"format":"pt"}}   )";
    auto parsed = format::parse_header(bytes(text), 16, limits);
    text.assign("changed caller bytes");
    check(parsed.size() == 3, "metadata is excluded from tensor catalog");
    std::size_t covered = 0;
    for (const auto& entry : parsed) {
        check(entry.byte_count == tensors::required_bytes(entry.tensor.layout), "shape matches extent");
        check(entry.data_offset <= 16 && entry.byte_count <= 16 - entry.data_offset, "range contained");
        covered += entry.byte_count;
        if (entry.tensor.name == "a") {
            check(entry.tensor.layout.shape.empty() && entry.byte_count == 4, "scalar tensor retained");
        } else if (entry.tensor.name == "z") {
            check(entry.tensor.layout.element_type == tensors::ElementType::bfloat16 &&
                  entry.tensor.layout.shape == std::vector<std::size_t>{2, 3} &&
                  entry.data_offset == 4, "BF16 shape and offset retained");
        } else {
            check(entry.tensor.name == "empty" && entry.byte_count == 0, "empty tensor retained");
        }
    }
    check(covered == 16, "whole payload covered");
    check(format::parse_header(bytes("{}"), 0, limits).empty(), "empty catalog for empty payload");
}

void invalid_headers() {
    for (const auto text : {
             R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
             R"({"x":{"dtype":"F32","dtype":"BF16","shape":[2],"data_offsets":[0,4]}})",
             R"({"__metadata__":{"a":"one","a":"two"}})",
             R"({"__metadata__":{"a":3}})",
             R"({"x":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}})",
             R"({"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}})",
             R"({"x":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}})",
             R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[-1,3]}})",
             R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}})",
             R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})",
             R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}})",
             R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":true}})",
             R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
             R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}} trailing)",
             " {}", "[]", "{", "{}"}) {
        expect_throws<std::invalid_argument>([&] { (void)format::parse_header(bytes(text), 4, limits); });
    }
    const auto gap = R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})";
    expect_throws<std::invalid_argument>([&] { (void)format::parse_header(bytes(gap), 8, limits); });
}

void limits_and_overflow() {
    const auto valid = R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    for (const auto restricted : {format::HeaderLimits{4, limits.max_metadata_bytes, 10},
                                 format::HeaderLimits{limits.max_header_bytes, 4, 10},
                                 format::HeaderLimits{limits.max_header_bytes, limits.max_metadata_bytes, 0}}) {
        expect_throws<std::length_error>([&] { (void)format::parse_header(bytes(valid), 4, restricted); });
    }
    const auto overflow = std::string(R"({"x":{"dtype":"F32","shape":[)") +
        std::to_string(std::numeric_limits<std::size_t>::max()) + R"(],"data_offsets":[0,4]}})";
    expect_throws<std::length_error>([&] { (void)format::parse_header(bytes(overflow), 4, limits); });
    std::string nested = "{\"x\":";
    for (int i = 0; i < 65; ++i) nested += '[';
    nested += '0';
    for (int i = 0; i < 65; ++i) nested += ']';
    nested += '}';
    expect_throws<std::length_error>([&] { (void)format::parse_header(bytes(nested), 0, limits); });
}
}  // namespace

int main() {
    valid_headers();
    invalid_headers();
    limits_and_overflow();
}
