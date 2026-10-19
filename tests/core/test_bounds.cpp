/*
 * Pure host bounds-access contracts: valid indices return the original object;
 * empty/end/oversized indices throw before access, and constness is preserved.
 * No files, GPU work, subprocesses, or dependence on the helper's implementation.
 * Algorithm responsibility: check the shared metadata/span access boundary.
 * Constant-size stack inputs exercise both span and owning-container callers.
 * Corpus consultation, header design and implementation, 2026-10-19:
 * cpp-guidelines SL.con.3 (#rsl-bounds): exercise range boundaries explicitly.
 * cpp-perf-guidelines COPY.7: verify reference identity, not element copies.
 * Source: guidelines/copy-move/COPY.7-eliminate-hidden-copies.md.
 */
#include "bounds.h"
#include "tests/support/check.h"
#include <array>
#include <limits>
#include <type_traits>

int main() try {
    std::array values{3, 7};
    const std::span<int> view(values);
    const std::span<const int> read_only(values);
    static_assert(std::is_same_v<decltype(dolittle::at(read_only, 0)), const int &>);
    dolittle::tests::check(&dolittle::at(view, 1) == &values.back(), "span access borrows original element");
    dolittle::at(view, 1) = 11;
    dolittle::tests::check(dolittle::at(read_only, 1) == 11, "borrowed mutation is visible");
    dolittle::tests::check(&dolittle::at(values, 0) == &values.front(), "container access borrows element");
    for (const auto index : {values.size(), std::numeric_limits<std::size_t>::max()}) {
        dolittle::tests::expect_throws<std::out_of_range>([&] { (void)dolittle::at(view, index); });
        dolittle::tests::expect_throws<std::out_of_range>([&] { (void)dolittle::at(values, index); });
    }
    dolittle::tests::expect_throws<std::out_of_range>([] { (void)dolittle::at(std::span<int>{}, 0); });
} catch (const std::exception &error) {
    return dolittle::tests::report_failure(error.what());
} catch (...) {
    return dolittle::tests::report_failure("unexpected non-standard exception");
}
