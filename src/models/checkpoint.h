/*
 * Bound checkpoint ownership interface.
 * Axis M3: verified parameters bound to a selected model's semantic slots.
 * Contract: a concrete BoundCheckpoint exclusively owns its Model and source
 * archive. parameter(id) returns a validated immutable HostView exactly matching
 * that model's ParameterSpec; every required parameter exists, none is randomly
 * initialized. Unknown IDs -> invalid_argument. Views/model reference remain
 * valid until owner destruction. A checkpoint is stable and noncopyable.
 * identity combines verified artifact identity, model recipe identity and
 * binding version, distinguishing identical architecture with different weights.
 * parameter_bytes counts distinct source payload bytes, not virtual address
 * space, file headers, device allocations or a promise of peak host residency.
 * Forward plans borrow this owner and upload selected parameters themselves;
 * destroying a plan does not destroy the host checkpoint. No GPU state here.
 *
 * Algorithm: docs/architecture/transformer-execution.md, "Checkpoint loading
 * and binding": model-specific binder interprets config, validates serialized
 * names/shapes/dtypes, assigns parameter slots, then exposes the completed owner.
 * Missing/unsupported input fails before a bound checkpoint is published.
 * Cost: O(parameter count) binding metadata, borrowed payloads. Optimization:
 * resolve serialized names once; execution indexes slots, not string maps.
 *
 * Corpus consultation, implementation (full rules read), 2026-10-18:
 * cpp-guidelines R.1 (#rr-raii): checkpoint owns model and backing archive.
 * cpp-guidelines I.7 (#ri-post): successful binding guarantees required weights.
 * cpp-perf-guidelines CDSA.32: immutable binding follows consumer slot order.
 * Source: guidelines/cpu-dsa/CDSA.32-transform-static-data-into-the-consumer-layout.md.
 */
#pragma once

#include "models/model.h"
#include "tensors/tensor.h"
#include <cstddef>
#include <string_view>

namespace dolittle::models {

class BoundCheckpoint {
  public:
    BoundCheckpoint(BoundCheckpoint &&) = delete;
    BoundCheckpoint &operator=(BoundCheckpoint &&) = delete;
    virtual ~BoundCheckpoint();
    [[nodiscard]] virtual const Model &model() const noexcept = 0;
    [[nodiscard]] virtual std::string_view identity() const noexcept = 0;
    [[nodiscard]] virtual tensors::HostView parameter(ParameterId id) const = 0;
    [[nodiscard]] virtual std::size_t parameter_bytes() const noexcept = 0;
    BoundCheckpoint(const BoundCheckpoint &) = delete;
    BoundCheckpoint &operator=(const BoundCheckpoint &) = delete;

  protected:
    BoundCheckpoint();
};

} // namespace dolittle::models
