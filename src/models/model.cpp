/*
 * Implementation-file header design only; no function bodies yet.
 * Axis M6: pure structural validation of a model-produced forward recipe.
 * Contract: implement validate_recipe and Model lifetime definitions. Validation
 * does not execute operations, load artifacts or query a device. It follows
 * model.h's invalid-input/overflow/allocation failures, with no partial side effects.
 * Algorithm: transformer overlay, "Preparing the first execution recipe":
 * validate token shape, parameter assignments and layouts; walk ordered operations
 * using validate_operation; reject read-before-produce, multiple producers and
 * writes to external input; validate catalog/site values and selected logit shape.
 * Observation names may alias one completed tensor at adjacent block boundaries.
 * Design and cost: O(tensors+operand uses+sites) expected work with indexed state,
 * plus model metadata creation. Optimization: validate once before resource planning.
 * Tests check producer, shape, site and ownership contracts using in-memory data;
 * they do not snapshot slot numbers, operation counts or a private recipe layout.
 * Mathematical equivalence still requires independent numerical integration.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): enforce the stated results and failure boundaries.
 * cpp-perf-guidelines GPU.9: prepare storage once and preserve last-use lifetimes.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "models/model.h"
