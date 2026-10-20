/*
 * Axis D3: checked steady-state allocation contract.
 * Contract: allocation-check builds count C++ new and explicit GPU/staging
 * allocation calls on this thread. A successful guarded iteration must count
 * zero; violation aborts. Exception reporting, preparation, teardown and HTTP
 * reader threads are outside this contract. This does not intercept arbitrary
 * C malloc inside shared libraries. Release builds compile the guard away.
 * Algorithm: snapshot exception depth, enter thread-local scope, count calls,
 * leave scope, reject a nonzero count only on normal completion. No allocation
 * is performed by instrumentation. Guard includes inference observer callbacks.
 * Core I.7 (#ri-post); MEM.9
 * (guidelines/memory/MEM.9-embedded-allocate-at-init.md): enforce the
 * prepared-storage invariant in the executed loop. Both corpora queried/full
 * rules read for design and implementation, 2026-10-20.
 */
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
namespace dolittle::diagnostics {
#ifdef DOLITTLE_ALLOCATION_CHECKS
struct AllocationState {
  unsigned depth{};
  std::size_t count{};
};
inline AllocationState& allocation_state() noexcept {
  static thread_local AllocationState state;
  return state;
}
inline void record_allocation() noexcept {
  if (allocation_state().depth) {
    ++allocation_state().count;
  }
}
class AllocationScope {
 public:
  explicit AllocationScope(bool enabled, const bool* failed = nullptr) noexcept
      : enabled_(enabled),
        failed_(failed),
        exceptions_(std::uncaught_exceptions()) {
    if (enabled_ && allocation_state().depth++ == 0) {
      allocation_state().count = 0;
    }
  }
  ~AllocationScope() {
    if (enabled_ && --allocation_state().depth == 0 &&
        allocation_state().count && (!failed_ || !*failed_) &&
        std::uncaught_exceptions() == exceptions_) {
      std::fputs("generation allocation invariant failed\n", stderr);
      std::abort();
    }
  }
  AllocationScope(const AllocationScope&) = delete;
  AllocationScope& operator=(const AllocationScope&) = delete;
  AllocationScope(AllocationScope&&) = delete;
  AllocationScope& operator=(AllocationScope&&) = delete;

 private:
  bool enabled_;
  const bool* failed_;
  int exceptions_;
};
#else
inline void record_allocation() noexcept {}
class AllocationScope {
 public:
  explicit AllocationScope(bool, const bool* = nullptr) noexcept {}
};
#endif
}  // namespace dolittle::diagnostics
