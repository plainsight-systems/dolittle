/*
 * Axis D3: independent operation equations for target numerical integration.
 * Contract: implement operator_reference.h. All operands are finite stored
 * FP32/BF16 values; expected arithmetic is FP32, with the model's BF16 rounding
 * points. Native host exp/sqrt/sin/cos supply independent elementary functions.
 * No production reduction tree or GPU math helper supplies expected results.
 *
 * Algorithm: numerical-portability.md, Operation checks. Projection sums X*W
 * over features. RMS divides X by sqrt(mean(X^2)+epsilon), rounds, multiplies
 * weight and rounds. RoPE rotates split-half pairs at each absolute position,
 * rounding trig values/products. Attention computes causal QK/sqrt(d),
 * subtracts the maximum, normalizes FP32 exponentials and sums probability*V.
 * SwiGLU computes gate/(1+exp(-gate)), rounds, then multiplies up. Addition
 * rounds X+Y. All reductions use ordinary serial FP32 equations, not kernel
 * partitions.
 *
 * Design: shape cases cover tails, head/hidden widths, grouped-query attention
 * and prefill/decode-sized operations. Upload the same stored operands, poison
 * output, execute the production operation, drain and compare. Per-element
 * limits are selected by operation/dtype before execution from the pinned
 * sources in numerical-portability.md. Report norm error separately.
 * Cost: small bounded diagnostic allocations/readbacks outside inference.
 * Queue destruction precedes every borrowed buffer even on failure.
 * Both local corpora queried/read for header design and implementation,
 * 2026-10-21: Core I.1/I.7 (#ri-explicit/#ri-post) define comparison results;
 * performance GDSA.2 distinguishes numerical accuracy from replay identity
 * (guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md); GPU.1
 * keeps this reference/readback out of inference
 * (guidelines/gpu/GPU.1-keep-data-on-device.md).
 */
#include "tests/gpu/operator_reference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include "bounds.h"
#include "gpu/device.h"
#include "kernels/operations.h"
#include "tensors/representations/bf16.h"
#include "tests/support/check.h"
#include "tests/support/numerical.h"

namespace dolittle::tests {
namespace {
using Json = nlohmann::json;
using Type = tensors::ElementType;
struct Operand {
  tensors::Layout layout;
  std::vector<float> values;
};
float rounded(float value, Type type) {
  return type == Type::bfloat16 ? tensors::to_float(tensors::from_float(value))
                                : value;
}
Operand operand(Type type, std::initializer_list<std::size_t> shape,
                std::size_t seed) {
  Operand result{
      tensors::contiguous_layout(type, {shape.begin(), shape.size()}), {}};
  const auto count =
      tensors::required_bytes(result.layout) / tensors::element_bytes(type);
  result.values.reserve(count);
  // Integer construction is reproducible without sharing production RNG.
  for (std::size_t i = 0; i < count; ++i) {
    const auto signed_value =
        static_cast<int>((i * 73 + seed * 37) % 257) - 128;
    result.values.push_back(
        rounded(static_cast<float>(signed_value) / 64, type));
  }
  return result;
}
ElementwiseTolerance ordinary(Type type) {
  return type == Type::bfloat16 ? ElementwiseTolerance{1e-3, 1.6e-2}
                                : ElementwiseTolerance{1e-5, 1.3e-6};
}
void encode(const Operand& operand, std::span<std::byte> bytes, bool poison) {
  const auto type = operand.layout.element_type;
  for (std::size_t i = 0; i < operand.values.size(); ++i) {
    const float value =
        poison ? std::numeric_limits<float>::quiet_NaN() : operand.values.at(i);
    if (type == Type::float32) {
      std::memcpy(bytes.subspan(i * sizeof(float), sizeof(float)).data(),
                  &value, sizeof(float));
    } else {
      tensors::write_le(tensors::from_float(value),
                        bytes.subspan(i * 2, 2).first<2>());
    }
  }
}
Json evaluate(gpu::Device& device, const char* name,
              const kernels::Operation& operation,
              std::vector<Operand> operands, ElementwiseTolerance tolerance) {
  const auto type = operands.back().layout.element_type;
  std::vector<tensors::Layout> layouts;
  layouts.reserve(operands.size());
  for (const auto& tensor : operands) layouts.push_back(tensor.layout);
  auto prepared =
      kernels::prepare(device, operation, layouts, {type, Type::float32});
  std::vector<gpu::DeviceBuffer> buffers;
  std::vector<gpu::HostBuffer> staging;
  buffers.reserve(operands.size());
  staging.reserve(operands.size());
  for (std::size_t i = 0; i < operands.size(); ++i) {
    const auto bytes = tensors::required_bytes(operands.at(i).layout);
    buffers.push_back(device.allocate({bytes, 64}));
    staging.push_back(device.allocate_staging({bytes, 64}));
    encode(operands.at(i), staging.back().bytes(), i + 1 == operands.size());
  }
  const auto scratch = prepared.workspace();
  std::optional<gpu::DeviceBuffer> workspace;
  if (scratch.bytes) {
    workspace.emplace(device.allocate({scratch.bytes, scratch.alignment}));
  }
  const gpu::DeviceRegion region{workspace ? &*workspace : nullptr, 0,
                                 scratch.bytes};
  std::vector<kernels::BufferBinding> bindings;
  bindings.reserve(buffers.size());
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    bindings.push_back({{i}, &buffers.at(i), 0, buffers.at(i).size_bytes()});
  }
  auto queue = device.create_queue();
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    queue.upload({&buffers.at(i), 0, buffers.at(i).size_bytes()},
                 {&staging.at(i), 0, buffers.at(i).size_bytes()});
  }
  prepared.initialize(queue, region);
  prepared.submit(queue, bindings, region);
  queue.download({&staging.back(), 0, buffers.back().size_bytes()},
                 {&buffers.back(), 0, buffers.back().size_bytes()});
  queue.drain();
  std::vector<float> actual(operands.back().values.size());
  const auto bytes = staging.back().bytes();
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (type == Type::float32) {
      std::memcpy(&actual.at(i),
                  bytes.subspan(i * sizeof(float), sizeof(float)).data(),
                  sizeof(float));
    } else {
      actual.at(i) = tensors::to_float(
          tensors::read_le(bytes.subspan(i * 2, 2).first<2>()));
    }
  }
  const auto& expected = operands.back().values;
  const auto comparison = compare_elementwise(expected, actual, tolerance);
  const auto norm = compare_values(
      expected, actual,
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()});
  Json shapes = Json::array();
  for (const auto& layout : layouts) shapes.push_back(layout.shape);
  return {{"operation", name},
          {"dtype", type == Type::bfloat16 ? "bfloat16" : "float32"},
          {"shapes", shapes},
          {"atol", tolerance.absolute},
          {"rtol", tolerance.relative},
          {"elements", actual.size()},
          {"mismatched_elements", comparison.mismatched_elements},
          {"finite", comparison.all_values_finite},
          {"max_absolute_error", comparison.maximum_absolute_error},
          {"relative_l2",
           norm.relative_l2 ? Json(*norm.relative_l2) : Json(nullptr)},
          {"passed", comparison.passed}};
}
void normalization(gpu::Device& device, Type type, Json& cases) {
  for (const auto width : {std::size_t{128}, std::size_t{4096}}) {
    auto x = operand(type, {3, width}, 1);
    auto w = operand(type, {width}, 2);
    auto y = operand(type, {3, width}, 0);
    constexpr float epsilon = 1e-6F;
    for (std::size_t row = 0; row < 3; ++row) {
      float sum = 0;
      for (std::size_t i = 0; i < width; ++i) {
        const float value = x.values.at(row * width + i);
        sum += value * value;
      }
      const float inverse =
          1 / std::sqrt(sum / static_cast<float>(width) + epsilon);
      for (std::size_t i = 0; i < width; ++i) {
        y.values.at(row * width + i) =
            rounded(rounded(x.values.at(row * width + i) * inverse, type) *
                        w.values.at(i),
                    type);
      }
    }
    cases.push_back(evaluate(device, "rms_norm",
                             kernels::RmsNorm{{0}, {1}, {2}, epsilon},
                             {x, w, y}, {1e-2, 1e-2}));
  }
}
void elementwise(gpu::Device& device, Type type, Json& cases) {
  auto x = operand(type, {3, 259}, 3);
  auto up = operand(type, {3, 259}, 4);
  auto y = operand(type, {3, 259}, 0);
  // Exercise saturated positive and negative SiLU, as well as exact zero.
  x.values.at(0) = -80;
  x.values.at(1) = 80;
  x.values.at(2) = 0;
  for (std::size_t i = 0; i < y.values.size(); ++i) {
    const float gate = x.values.at(i);
    y.values.at(i) = rounded(
        rounded(gate / (1 + std::exp(-gate)), type) * up.values.at(i), type);
  }
  cases.push_back(evaluate(device, "silu_multiply",
                           kernels::SiluMultiply{{0}, {1}, {2}}, {x, up, y},
                           ordinary(type)));
  for (std::size_t i = 0; i < y.values.size(); ++i) {
    y.values.at(i) = rounded(x.values.at(i) + up.values.at(i), type);
  }
  cases.push_back(
      evaluate(device, "add", kernels::Add{{0}, {1}, {2}}, {x, up, y}, {0, 0}));
}
void projection(gpu::Device& device, Type type, Json& cases) {
  for (const auto rows : {std::size_t{1}, std::size_t{9}}) {
    for (const auto output : {std::size_t{33}, std::size_t{1025}}) {
      constexpr std::size_t input = 128;
      auto x = operand(type, {rows, input}, 5);
      auto w = operand(type, {output, input}, 6);
      auto y = operand(type, {rows, output}, 0);
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < output; ++out) {
          float sum = 0;
          for (std::size_t i = 0; i < input; ++i) {
            sum += x.values.at(row * input + i) * w.values.at(out * input + i);
          }
          y.values.at(row * output + out) = rounded(sum, type);
        }
      }
      cases.push_back(evaluate(device, "projection",
                               kernels::Projection{{0}, {1}, {2}}, {x, w, y},
                               ordinary(type)));
    }
  }
}
void rotary(gpu::Device& device, Type type, Json& cases) {
  constexpr std::size_t rows = 3, heads = 2, width = 128, first = 4093;
  constexpr double theta = 1000000;
  auto x = operand(type, {rows, heads, width}, 7);
  auto y = operand(type, {rows, heads, width}, 0);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t i = 0; i < width / 2; ++i) {
      // Frequency setup is binary64 by the model contract; tensor math stays
      // FP32.
      const float frequency = static_cast<float>(
          std::pow(theta, -2.0 * static_cast<double>(i) / width));
      const float angle = static_cast<float>(first + row) * frequency;
      const float cosine = rounded(std::cos(angle), type);
      const float sine = rounded(std::sin(angle), type);
      for (std::size_t head = 0; head < heads; ++head) {
        const auto a = (row * heads + head) * width + i, b = a + width / 2;
        y.values.at(a) = rounded(rounded(x.values.at(a) * cosine, type) -
                                     rounded(x.values.at(b) * sine, type),
                                 type);
        y.values.at(b) = rounded(rounded(x.values.at(b) * cosine, type) +
                                     rounded(x.values.at(a) * sine, type),
                                 type);
      }
    }
  }
  cases.push_back(evaluate(device, "rotary",
                           kernels::Rotary{{0}, {1}, first, theta}, {x, y},
                           ordinary(type)));
}
void attention(gpu::Device& device, Type type, Json& cases) {
  constexpr std::size_t heads = 4, kv_heads = 2, width = 128;
  const float scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(width)));
  for (const auto rows : {std::size_t{1}, std::size_t{37}}) {
    auto q = operand(type, {rows, heads, width}, 8);
    auto k = operand(type, {rows, kv_heads, width}, 9);
    auto v = operand(type, {rows, kv_heads, width}, 10);
    auto y = operand(type, {rows, heads, width}, 0);
    std::vector<float> probabilities(rows);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t head = 0; head < heads; ++head) {
        const auto kv = head / (heads / kv_heads);
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t key = 0; key <= row; ++key) {
          float dot = 0;
          for (std::size_t i = 0; i < width; ++i) {
            dot += q.values.at((row * heads + head) * width + i) *
                   k.values.at((key * kv_heads + kv) * width + i);
          }
          probabilities.at(key) = dot * scale;
          maximum = std::max(maximum, probabilities.at(key));
        }
        float denominator = 0;
        for (std::size_t key = 0; key <= row; ++key) {
          probabilities.at(key) = std::exp(probabilities.at(key) - maximum);
          denominator += probabilities.at(key);
        }
        for (std::size_t key = 0; key <= row; ++key) {
          probabilities.at(key) /= denominator;
        }
        for (std::size_t i = 0; i < width; ++i) {
          float sum = 0;
          for (std::size_t key = 0; key <= row; ++key) {
            sum += probabilities.at(key) *
                   v.values.at((key * kv_heads + kv) * width + i);
          }
          y.values.at((row * heads + head) * width + i) = rounded(sum, type);
        }
      }
    }
    const auto tolerance = type == Type::bfloat16
                               ? ElementwiseTolerance{3e-2, 2e-2}
                               : ElementwiseTolerance{1e-3, 1e-3};
    cases.push_back(evaluate(device, "causal_attention",
                             kernels::CausalAttention{{0}, {1}, {2}, {3}},
                             {q, k, v, y}, tolerance));
  }
}
}  // namespace
Json check_operator_reference(gpu::Device& device) {
  Json cases = Json::array();
  for (const auto type : {Type::float32, Type::bfloat16}) {
    normalization(device, type, cases);
    elementwise(device, type, cases);
    projection(device, type, cases);
    rotary(device, type, cases);
    attention(device, type, cases);
  }
  const bool passed = std::ranges::all_of(
      cases, [](const Json& item) { return item.at("passed").get<bool>(); });
  return {{"reference",
           "independent FP32 host equations with explicit storage rounding"},
          {"policy", "numerical-portability-v1"},
          {"cases", cases},
          {"passed", passed}};
}
}  // namespace dolittle::tests
