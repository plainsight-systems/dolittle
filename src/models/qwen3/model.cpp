/*
 * Implementation-file header design only; no function bodies yet.
 * Axis M6, Qwen3 kind: immutable target-model description and forward recipe.
 * Contract: implement model.h for the declared Qwen3 subset only. Validate positive
 * dimensions, checked products, supported dtype, context bound and positional
 * constants before constructing parameter metadata. Invalid modes never fall back.
 * Algorithm: transformer overlay, "One Qwen3 transformer block": emit embedding,
 * input norm, Q/K/V projections, per-head Q/K norm, Q/K RoPE, causal GQA, output
 * projection, first residual, post-attention norm, gate/up projections,
 * SiLU-times-up, down projection and second residual. Record both block boundaries
 * without choosing a capture index. Gather requested final rows, normalize and
 * project vocabulary logits only when that selection is nonempty.
 * Design and cost: O(blocks+selected rows) metadata; widths come from Config, not
 * the selected checkpoint's constants. Optimization: reuse the second norm, keep
 * KV heads grouped, and do not project unused logit rows. Store each mathematical
 * value once in the recipe; physical aliasing is the executor's responsibility.
 * Unit checks cover advertised shapes/sites/configuration contracts. Full output
 * equivalence uses the pinned reference in an integration scenario; exact private
 * instruction ordering/counts are not unit assertions.
 * Overlay: docs/architecture/transformer-execution.md.
 * Corpus consultation, header design, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit): explicit inputs and no hidden execution state.
 * cpp-perf-guidelines GPU.9: prepare storage once and preserve last-use lifetimes.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "models/qwen3/model.h"
