/*
 * Shared host-side bounds access used while binding metadata and inspecting
 * borrowed tensor/observation spans. This is not a device indexing abstraction.
 * Contract: at(span, i) returns the original element by reference when i<size;
 * otherwise it throws out_of_range before touching memory. It never allocates,
 * copies elements, extends their lifetime, or changes their const
 * qualification. Container overloads delegate to the container's checked at().
 * Algorithm: compare index with span extent, then address that one element.
 * Cost: O(1), with an inline check the compiler can eliminate in bounded loops.
 * Corpus consultation, header design and implementation, 2026-10-19:
 * cpp-guidelines SL.con.3 (#rsl-bounds), I.30 (#ri-encapsulate): centralize the
 * checked access because C++20 span lacks at(); keep the raw index local.
 * cpp-perf-guidelines COPY.7: return references, never copy the borrowed data.
 * Source: guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 */
#pragma once
#include <cstddef>
#include <span>
#include <stdexcept>
namespace dolittle {
template <class T, std::size_t Extent>
constexpr T& at(std::span<T, Extent> values, std::size_t index) {
  if (index >= values.size()) {
    throw std::out_of_range("span index exceeds extent");
  }
  // The preceding check proves this sole unchecked access is in range.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
  return values[index];
}
template <class Container>
constexpr decltype(auto) at(Container& values, std::size_t index) {
  return values.at(index);
}
}  // namespace dolittle
