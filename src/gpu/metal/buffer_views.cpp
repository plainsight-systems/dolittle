/*
 * X4 pure view placement; implements buffer_views.h. Cover backing with bounded
 * page-aligned views; add an overlapping view only for a binding that crosses
 * a baseline boundary. Every returned binding index contains the entire range.
 * No physical allocation or Metal call; overlaps share backing, not bytes.
 * Setup-only O(bindings*views), O(views+bindings) metadata. Overlay:
 * metal-port.md, Memory and model residency. Implementation corpora
 * queried/read 2026-10-21: Core I.7 (#ri-post), ES.103 (#res-overflow);
 * performance GPU.9
 * (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md)
 * preserves backing ownership; GPU.6
 * (guidelines/gpu/GPU.6-batch-tiny-gpu-work.md) resolves bindings before live
 * execution. Invalid extents fail before native creation.
 */
#include "gpu/metal/buffer_views.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace dolittle::gpu::metal {
BufferViewPlan plan_buffer_views(std::size_t backing, std::size_t maximum,
                                 std::size_t page,
                                 std::span<const BufferExtent> bindings) {
  if (!page || (page & (page - 1)) || !maximum || maximum % page ||
      backing % page) {
    throw std::invalid_argument("invalid buffer view alignment");
  }
  for (const auto binding : bindings) {
    if (binding.bytes >
        std::numeric_limits<std::size_t>::max() - binding.offset) {
      throw std::length_error("buffer binding overflows");
    }
    if (!binding.bytes || binding.offset > backing ||
        binding.bytes > backing - binding.offset) {
      throw std::invalid_argument("binding exceeds backing");
    }
    const auto end = binding.offset + binding.bytes;
    const auto aligned_end = end + (page - end % page) % page;
    if (aligned_end - (binding.offset - binding.offset % page) > maximum) {
      throw std::length_error("binding cannot fit a native buffer view");
    }
  }
  BufferViewPlan result;
  for (std::size_t offset = 0; offset < backing;) {
    const auto bytes = std::min(maximum, backing - offset);
    result.views.push_back({offset, bytes});
    offset += bytes;
  }
  result.binding_views.reserve(bindings.size());
  for (const auto binding : bindings) {
    const auto view =
        std::ranges::find_if(result.views, [binding](auto candidate) {
          return binding.offset >= candidate.offset &&
                 binding.offset - candidate.offset <= candidate.bytes &&
                 binding.bytes <=
                     candidate.bytes - (binding.offset - candidate.offset);
        });
    if (view != result.views.end()) {
      result.binding_views.push_back(
          static_cast<std::size_t>(view - result.views.begin()));
    } else {
      const auto begin = binding.offset - binding.offset % page;
      const auto end = binding.offset + binding.bytes;
      const auto aligned_end = end + (page - end % page) % page;
      result.binding_views.push_back(result.views.size());
      result.views.push_back({begin, aligned_end - begin});
    }
  }
  return result;
}
}  // namespace dolittle::gpu::metal
