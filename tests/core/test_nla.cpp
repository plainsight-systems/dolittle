/*
 * Axis M7 unit contracts: normalize small owned vectors and inspect raw tags.
 * No model loading, filesystem, GPU or generation. Exact 3-4-5 geometry checks
 * scaling; malformed/nonfinite inputs must reject without modifying the input.
 * Optimization: bounded in-memory arrays, no inference fixture dependency.
 * Overlay: logical-overview.md, "4 Interpret selected positions".
 * Corpus implementation consultation, 2026-10-19: cpp-guidelines I.7 (#ri-post)
 * tests observable conversion/postconditions; R.1 (#rr-raii) owns fixture
 * bytes. cpp-perf-guidelines GPU.1: conversion tests run on retained host
 * bytes; source guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include <array>
#include <cstring>
#include <limits>

#include "nla/protocol.h"
#include "tensors/representations/bf16.h"
#include "tests/support/check.h"

namespace tensors = dolittle::tensors;
namespace models = dolittle::models;
namespace tests = dolittle::tests;
namespace nla = dolittle::nla;
namespace inference = dolittle::inference;
namespace text = dolittle::text;

int main() try {
  const std::array<float, 2> values{3, 4};
  const nla::CaptureRow row{{{models::ObservationBoundary::block_output, {2}},
                             2,
                             tensors::ElementType::float32},
                            {7},
                            std::as_bytes(std::span(values))};
  const auto result =
      nla::normalize_activation(row, {3}, 10, tensors::ElementType::float32);
  std::array<float, 2> actual{};
  std::memcpy(actual.data(), result.embedding.bytes.data(),
              result.embedding.bytes.size());
  tests::check(
      result.source_l2_norm == 5 && actual == std::array<float, 2>{6, 8},
      "normalization law");
  tests::check(result.embedding.position.value == 3 &&
                   row.position.value == 7 &&
                   values == std::array<float, 2>{3, 4},
               "position separation and immutable input");
  const auto bf =
      nla::normalize_activation(row, {0}, 10, tensors::ElementType::bfloat16);
  tests::check(tensors::to_float(tensors::read_le(std::span<const std::byte, 2>(
                   bf.embedding.bytes.data(), 2))) == 6,
               "BF16 conversion");
  std::array<float, 2> zero{};
  auto invalid = row;
  invalid.bytes = std::as_bytes(std::span(zero));
  tests::expect_throws<std::invalid_argument>([&] {
    (void)nla::normalize_activation(invalid, {0}, 1,
                                    tensors::ElementType::float32);
  });
  zero.at(0) = std::numeric_limits<float>::infinity();
  tests::expect_throws<std::invalid_argument>([&] {
    (void)nla::normalize_activation(invalid, {0}, 1,
                                    tensors::ElementType::float32);
  });
  tests::expect_throws<std::invalid_argument>([&] {
    (void)nla::normalize_activation(row, {0}, -1,
                                    tensors::ElementType::float32);
  });
  const std::string raw = "before<explanation> \nhello\t</explanation>after";
  const auto tags = nla::explanation_tags(raw);
  tests::check(
      tags.format == nla::ExplanationFormat::complete && tags.content &&
          raw.substr(tags.content->begin,
                     tags.content->end - tags.content->begin) == "hello",
      "retained output range");
  tests::check(
      nla::explanation_tags("raw").format == nla::ExplanationFormat::missing,
      "missing tags");
  for (auto text :
       {"<explanation>x", "</explanation><explanation>",
        "<explanation><explanation>x</explanation>",
        "<explanation>x</explanation><explanation>y</explanation>"}) {
    tests::check(
        nla::explanation_tags(text).format == nla::ExplanationFormat::malformed,
        "malformed tags");
  }
} catch (const std::exception& error) {
  return dolittle::tests::report_failure(error.what());
} catch (...) {
  return dolittle::tests::report_failure("unexpected non-standard exception");
}
