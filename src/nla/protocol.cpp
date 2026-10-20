/*
 * Pure activation normalization and explanation delimiter extraction.
 * Axis M7: explicit host reference conversion and text delimiter extraction.
 * Live device rows are rejected; their numerical path is DeviceInjection.
 * Contract: borrow one D-element FP32/BF16 row, a separate AV position and a
 * positive finite scale. Return owned replacement bytes and the original norm.
 * Reject empty/malformed/nonfinite rows, zero/overflowed norm, unsupported
 * storage and nonfinite converted output. Leave input bytes unchanged on every
 * outcome. Pair validates trained site/source compatibility before calling this
 * helper; this function checks the numerical row, not checkpoint provenance.
 *
 * Algorithm: docs/architecture/logical-overview.md, "4 Interpret selected
 * positions", and transformer-execution.md, "NLA entry and exit paths":
 * 1. Decode h[D] into an owned FP32 vector; BF16 widening is exact. Accumulate
 *    each FP32 product h[i]*h[i] into lane i modulo 8. Products and additions
 *    round separately; -ffp-contract=off forbids implicit fused operations.
 * 2. Add the eight lane sums left-to-right in FP32, then n=sqrt(sum). Require
 *    finite n>0. For each i compute u=h[i]/n, then v=u*scale in FP32. Do not
 *    replace these two operations with a precomputed reciprocal or ratio.
 * 3. Store v as FP32 or cast once to BF16 with round-to-nearest ties-to-even.
 *    Reject conversion overflow. Return n, scale and bytes addressed to the AV
 *    prompt position; the target consumed-token position is never rewritten.
 * 4. Independently scan raw AV output for <explanation> and </explanation>.
 *    Neither tag means missing; partial, reversed, nested or repeated tags mean
 *    malformed. For one ordered pair, trim ASCII whitespace within its bounds
 *    and return a range into the original bytes. Preserve all raw output.
 *
 * Cost: O(D) conversion, 4D FP32 + D*b output bytes and eight float sums;
 * delimiter scanning is O(output bytes), with no output-string copy.
 * Optimization: diagnostics normalize only the explicitly exported row.
 * No neighboring captures or full embedding matrix are copied here.
 * The eight-lane FP32 order matches the frozen CPU norm on the tested vectors;
 * reference arithmetic and acceptance limits remain unchanged.
 * Corpus consultation, implementation, 2026-10-19 (full rules read):
 * cpp-guidelines I.7 (#ri-post): finite owned output and explicit tag statuses.
 * cpp-guidelines R.1 (#rr-raii): local conversion storage survives exceptions.
 * cpp-perf-guidelines GPU.1: one selected-row transfer, no GPU embedding
 * readback; source guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Execution-boundary header/implementation consultation, 2026-10-20:
 * Core I.1/R.1 and Performance GPU.1/GPU.9 (both local corpora queried/read)
 * make this helper explicitly host-reference-only; device rows are rejected.
 */
#include "nla/protocol.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "bounds.h"
#include "tensors/representations/bf16.h"
#include "tensors/tensor.h"

namespace dolittle::nla {
Injection normalize_activation(const CaptureRow& row,
                               models::TokenPosition position, float scale,
                               tensors::ElementType output_storage) {
  using tensors::ElementType;
  const auto width = row.site.vector_width;
  if (row.device || !width ||
      width > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      (row.site.storage_type != ElementType::bfloat16 &&
       row.site.storage_type != ElementType::float32) ||
      (output_storage != ElementType::bfloat16 &&
       output_storage != ElementType::float32) ||
      !std::isfinite(scale) || scale <= 0 ||
      row.bytes.size() !=
          width * tensors::element_bytes(row.site.storage_type)) {
    throw std::invalid_argument("invalid activation normalization inputs");
  }
  constexpr std::size_t norm_lanes = 8;
  std::array<float, norm_lanes> sums{};
  std::vector<float> values(width);
  for (std::size_t i = 0; i < width; ++i) {
    if (row.site.storage_type == ElementType::float32) {
      std::memcpy(&values.at(i), row.bytes.subspan(i * sizeof(float)).data(),
                  sizeof(float));
    } else {
      values.at(i) = tensors::to_float(tensors::read_le(
          std::span<const std::byte, 2>(row.bytes.subspan(i * 2).data(), 2)));
    }
    if (!std::isfinite(values.at(i))) {
      throw std::invalid_argument("nonfinite capture");
    }
    const float square = values.at(i) * values.at(i);
    sums.at(i % norm_lanes) += square;
  }
  // Match the frozen CPU FP32 norm policy: eight interleaved accumulators,
  // then a left-to-right lane reduction. No FP64 or fused multiply/add.
  float sum = 0;
  for (float lane : sums) {
    sum += lane;
  }
  const float norm = std::sqrt(sum);
  if (!std::isfinite(norm) || norm == 0) {
    throw std::invalid_argument("invalid capture norm");
  }
  Injection result{
      {position, width, output_storage,
       std::vector<std::byte>(width * tensors::element_bytes(output_storage))},
      norm,
      scale};
  for (std::size_t i = 0; i < width; ++i) {
    const float divided = values.at(i) / norm;
    const float value = divided * scale;
    if (!std::isfinite(value)) {
      throw std::invalid_argument("nonfinite normalized activation");
    }
    if (output_storage == ElementType::float32) {
      std::memcpy(
          std::span(result.embedding.bytes).subspan(i * sizeof(float)).data(),
          &value, sizeof(float));
    } else {
      const auto stored = tensors::from_float(value);
      if (!std::isfinite(tensors::to_float(stored))) {
        throw std::invalid_argument("injection overflow");
      }
      tensors::write_le(
          stored,
          std::span<std::byte, 2>(
              std::span(result.embedding.bytes).subspan(i * 2).data(), 2));
    }
  }
  return result;
}
ExplanationRange explanation_tags(std::string_view raw) {
  constexpr std::string_view open = "<explanation>", close = "</explanation>";
  const auto first = raw.find(open), last = raw.find(close);
  if (first == raw.npos && last == raw.npos) {
    return {ExplanationFormat::missing, std::nullopt};
  }
  if (first == raw.npos || last == raw.npos || last < first + open.size() ||
      raw.find(open, first + open.size()) != raw.npos ||
      raw.find(close, last + close.size()) != raw.npos) {
    return {ExplanationFormat::malformed, std::nullopt};
  }
  auto begin = first + open.size(), end = last;
  constexpr std::string_view whitespace = " \t\n\r\f\v";
  while (begin < end && whitespace.find(raw.at(begin)) != whitespace.npos) {
    ++begin;
  }
  while (end > begin && whitespace.find(raw.at(end - 1)) != whitespace.npos) {
    --end;
  }
  return {ExplanationFormat::complete, ByteRange{begin, end}};
}
}  // namespace dolittle::nla
