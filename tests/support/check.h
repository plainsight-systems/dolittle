/*
 * Axis D3: minimal assertions for isolated in-process unit tests.
 * Contract: check throws runtime_error on false; expect_throws succeeds only
 * when the callable throws the requested exception type (or a derived type).
 * Unexpected exceptions propagate; a missing exception throws runtime_error.
 * Callables and messages are borrowed for this call, never stored. No external
 * resources, subprocesses, application startup or test registration is involved.
 * Algorithm: docs/architecture/transformer-execution.md, "Model computation"
 * and "Verification boundaries": unit tests supply observable contract facts;
 * these helpers turn a violated fact into a failing executable exit.
 * Cost: constant helper work excluding callable/message size; allocate only
 * for a failure diagnostic. Optimization: forward callables without type
 * erasure, capture copies or heap-backed test registration.
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): explicit assertion inputs
 * and exception outcomes; no hidden test state.
 * cpp-perf-guidelines COPY.7: no std::function or implicit message allocation
 * on a successful assertion. Source:
 * guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 */
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace dolittle::tests {
inline void check(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Exception, typename Callable>
void expect_throws(Callable&& callable) {
    try {
        std::forward<Callable>(callable)();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}
}  // namespace dolittle::tests
