/*
 * Axis X3: shared tensor addressing, independent of model and device backend.
 * Contract: Layout owns shape and byte strides of equal rank. Scalars have
 * rank zero and one element; any zero dimension denotes an empty tensor.
 * Strides are nonnegative, in bytes, and multiples of the element width.
 * required_bytes checks every extent/product/sum before arithmetic and returns
 * the addressable span from logical element zero (zero for empty tensors).
 * It does not establish non-overlap: operation preparation separately checks
 * dense/non-overlapping layout requirements. Negative strides and packed
 * encodings are outside this initial contract. No implicit dtype conversions.
 * HostView borrows both layout and immutable storage; neither owner may move,
 * mutate or be destroyed while used. validate_view checks the byte extent;
 * it does not reinterpret possibly unaligned bytes as native float pointers.
 * Invalid layout/type -> invalid_argument; overflow -> length_error.
 * contiguous_layout returns row-major strides for nonempty shapes; strides of
 * empty shapes are zero, avoiding products for addresses that cannot be used.
 * Owned metadata allocation can propagate bad_alloc.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Model dimensions and
 * notation" and "Cost model": carry explicit shape/type/strides, derive the
 * reachable byte extent, validate storage, then admit a consumer operation.
 * Cost: O(rank) validation, no tensor data copies. Optimization: resolve layout
 * checks in preparation; do not repeat dimension products per element.
 *
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.4 (#ri-typed): byte strides and stored dtype are explicit.
 * cpp-guidelines I.7 (#ri-post): checked extent does not promise dense layout.
 * cpp-guidelines ES.103 (#res-overflow): check products and sums before use.
 * cpp-perf-guidelines CDSA.32: consumer layouts are validated during setup.
 * Source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "tensors/representations/representation.h"

namespace dolittle::tensors {

struct Layout {
  ElementType element_type{};
  std::vector<std::size_t> shape;
  std::vector<std::size_t> byte_strides;
};

struct HostView {
  const Layout* layout{};
  std::span<const std::byte> bytes;
};

[[nodiscard]] std::size_t element_bytes(ElementType type);
[[nodiscard]] std::size_t required_bytes(const Layout& layout);
[[nodiscard]] Layout contiguous_layout(ElementType type,
                                       std::span<const std::size_t> shape);
void validate_view(HostView view);

}  // namespace dolittle::tensors
