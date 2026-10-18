/*
 * Implementation-file header design only; no function bodies yet.
 * Axis X5: pure operation validation and prepared backend dispatch.
 * Contract: implement operations.h; separate mathematical validation from device
 * capability selection. Validate layouts, referenced IDs, constants, access roles
 * and checked shape products without touching a GPU. Read/read aliasing is legal;
 * any output/workspace overlap with another live region is rejected.
 * Algorithm: transformer overlay, "Preparing the first execution recipe":
 * validate the operation; select a compiled backend implementation with explicit
 * numerical/shape support; copy needed metadata; report operand alignment and
 * workspace. At submit validate bound regions/device identity, then enqueue the
 * selected launch with no model-name dispatch or host completion wait.
 * Design and cost: bounded operands per mathematical operation, gather rows are
 * prepared once. Optimization: do not reselect kernels, allocate device workspace
 * or rediscover tensor layouts at each submit. Backend-private launch handles and
 * their cleanup must remain alive until submitted work completes. No user callbacks.
 * Unit tests exercise validate_operation's shape/access/error contracts with
 * ordinary metadata; kernel execution is integration work.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit): explicit inputs and no hidden execution state.
 * cpp-perf-guidelines GPU.9: prepare storage once and preserve last-use lifetimes.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "kernels/operations.h"
