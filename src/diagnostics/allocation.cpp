/*
 * Axis D3: C++ allocation interception for allocation.h's checked contract.
 * All replaceable allocation/deallocation forms share the platform C heap.
 * Count on the allocating thread; allocation failure preserves standard new
 * handler behavior. No instrumentation on HTTP threads unless they enter a
 * scope. Linked only into allocation-check executables, never release builds.
 * Core I.7 (#ri-post), MEM.9
 * (guidelines/memory/MEM.9-embedded-allocate-at-init.md): exercise the real
 * generation boundary rather than inspecting source text. Both corpora
 * queried/full rules read for design/implementation, 2026-10-20.
 */
#include "diagnostics/allocation.h"

#include <algorithm>
#include <new>
// These functions implement the standard allocation boundary itself; raw
// ownership and C allocation are required here, not application ownership.
// NOLINTBEGIN(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc,cert-err58-cpp)
namespace {
void* allocate(std::size_t bytes, std::size_t alignment) {
  dolittle::diagnostics::record_allocation();
  for (;;) {
    void* pointer = nullptr;
    if (alignment <= alignof(std::max_align_t)) {
      pointer = std::malloc(std::max(bytes, std::size_t{1}));
    } else if (posix_memalign(&pointer, alignment,
                              std::max(bytes, std::size_t{1})) != 0) {
      pointer = nullptr;
    }
    if (pointer) {
      return pointer;
    }
    const auto handler = std::get_new_handler();
    if (!handler) {
      throw std::bad_alloc();
    }
    handler();
  }
}
}  // namespace
void* operator new(std::size_t n) {
  return allocate(n, alignof(std::max_align_t));
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, std::align_val_t a) {
  return allocate(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
  return ::operator new(n, a);
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  try {
    return ::operator new(n);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  try {
    return ::operator new[](n);
  } catch (...) {
    return nullptr;
  }
}
void* operator new(std::size_t n, std::align_val_t a,
                   const std::nothrow_t&) noexcept {
  try {
    return ::operator new(n, a);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t n, std::align_val_t a,
                     const std::nothrow_t&) noexcept {
  try {
    return ::operator new[](n, a);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
  std::free(p);
}
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}
void operator delete(void* p, std::align_val_t,
                     const std::nothrow_t&) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::align_val_t,
                       const std::nothrow_t&) noexcept {
  std::free(p);
}
// NOLINTEND(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc,cert-err58-cpp)
