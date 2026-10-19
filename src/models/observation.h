/*
 * Header design; model catalogs and capture execution are pending.
 * Axis: M6, model observation semantics. This is a shared vocabulary; each
 * model kind advertises a validated catalog with its own block count/widths.
 *
 * Contract: a point identifies a semantic boundary, not a scratch-buffer
 * address or an API hidden-state-array index. BlockIndex and TokenPosition
 * are zero based and distinct. Block output is after the whole block; for
 * Qwen3 this is after the second residual sum. Each advertised site describes
 * one unnormalized vector per consumed token, with positive width and explicit
 * storage type. The catalog contains no duplicate point. A request is only
 * valid when that exact point is advertised by the loaded model.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Configurable
 * observation points" and "One Qwen3 transformer block". Declare block input
 * and output boundaries, associate shape/type, then let the execution planner
 * resolve requested sites before the pass. Later intermediates need explicit
 * vocabulary/catalog support. Neither this contract nor its consumers know
 * an NLA's required capture block. This interface cannot mutate activations.
 *
 * Design and cost: catalog descriptions are small immutable planning inputs.
 * Optimization: identify sites during preparation, not by string matching or
 * callbacks for every token/element. This header owns no tensor memory.
 *
 * Corpus guidance queried for header design, 2026-10-17:
 * cpp-guidelines I.4 (#ri-typed): block and token indices have distinct types.
 * cpp-guidelines I.7 (#ri-post): output means the completed block residual.
 * cpp-perf-guidelines CDSA.32: resolve stable metadata into the consumer's
 * order once, outside execution. MCP source:
 * guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include <cstdint>
#include "tensors/representations/representation.h"

#include <cstddef>

namespace dolittle::models {

struct BlockIndex {
    std::size_t value;
};

struct TokenPosition {
    std::size_t value;
};

struct SequenceLength {
    std::size_t value;
};

enum class ObservationBoundary : std::uint8_t { block_input, block_output };

struct ObservationPoint {
    ObservationBoundary boundary;
    BlockIndex block;
};

struct ObservationSite {
    ObservationPoint point;
    std::size_t vector_width;
    tensors::ElementType storage_type;
};

} // namespace dolittle::models
