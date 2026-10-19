/*
 * Axis X3: checked tensor layout arithmetic.
 * Contract: implement tensor.h without reading tensor payloads or accessing a
 * device. Invalid dtype/rank/strides/null layout -> invalid_argument; overflowing
 * products/sums -> length_error. Empty shapes and rank-zero scalars follow the
 * interface's distinct rules. Returned layouts own their shape/stride vectors.
 * Algorithm: transformer overlay, "Model computation" and "Cost model": validate
 * dtype/rank/stride units, detect empty dimensions, compute reachable extent with
 * checked arithmetic, and compare HostView capacity. contiguous_layout derives
 * strides from the innermost dimension outward without unchecked multiplication.
 * Empty tensors receive zero strides, since no element address is reachable.
 * Design and cost: O(rank) validation, no data copies; allocation only for owned
 * layout metadata. Optimization: callers retain prepared layouts outside loops.
 * Unit verification checks extent/containment/rejection contracts, not internal
 * branch order. Kernel-specific dense/non-overlap validation is a separate step.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, implementation, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): enforce the stated results and failure boundaries.
 * cpp-guidelines ES.103 (#res-overflow): reject arithmetic overflow before use.
 * cpp-perf-guidelines CDSA.32: validate explicit consumer layouts during setup.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#include "bounds.h"
#include "tensors/tensor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace dolittle::tensors {
namespace {

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::length_error("tensor extent product overflows");
    }
    return a * b;
}

std::size_t checked_sum(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::length_error("tensor extent sum overflows");
    }
    return a + b;
}

} // namespace

std::size_t element_bytes(ElementType type) {
    switch (type) {
    case ElementType::bfloat16:
        return 2;
    case ElementType::float32:
    case ElementType::uint32:
        return 4;
    }
    throw std::invalid_argument("unsupported tensor element type");
}

std::size_t required_bytes(const Layout &layout) {
    const auto width = element_bytes(layout.element_type);
    if (layout.shape.size() != layout.byte_strides.size()) {
        throw std::invalid_argument("tensor shape and stride ranks differ");
    }
    for (const auto stride : layout.byte_strides) {
        if (stride % width != 0) {
            throw std::invalid_argument("tensor stride is not an element multiple");
        }
    }
    if (std::ranges::find(layout.shape, 0) != layout.shape.end()) {
        return 0;
    }
    auto extent = width;
    for (std::size_t i = 0; i < layout.shape.size(); ++i) {
        extent = checked_sum(extent, checked_product(layout.shape.at(i) - 1, layout.byte_strides.at(i)));
    }
    return extent;
}

Layout contiguous_layout(ElementType type, std::span<const std::size_t> shape) {
    const auto width = element_bytes(type);
    Layout result{type, {shape.begin(), shape.end()}, std::vector<std::size_t>(shape.size(), 0)};
    if (std::ranges::find(shape, 0) != shape.end()) {
        return result;
    }
    auto stride = width;
    for (auto i = shape.size(); i != 0; --i) {
        result.byte_strides.at(i - 1) = stride;
        stride = checked_product(stride, dolittle::at(shape, i - 1));
    }
    return result;
}

void validate_view(HostView view) {
    if (view.layout == nullptr) {
        throw std::invalid_argument("tensor view requires a layout");
    }
    if (required_bytes(*view.layout) > view.bytes.size()) {
        throw std::invalid_argument("tensor view storage is too small");
    }
}

} // namespace dolittle::tensors
