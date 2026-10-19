/*
 * Implementation of the reviewed ownership and decoding contracts.
 * Axis M3: BoundCheckpoint base lifetime definitions, not a model binder.
 * Contract: stable, noncopyable, polymorphically destroyed owner.
 * Model-specific derived owners release views before their archive/model
 * backing; the base performs no I/O, GPU wait or parameter initialization and
 * never throws. Algorithm: docs/architecture/transformer-execution.md,
 * "Checkpoint loading and binding": provide the ownership boundary consumed by
 * forward preparation; concrete binding publishes it only when complete and
 * releases it after all borrowing forward plans are gone. No selected-model
 * names in this file. Design and cost: constant base teardown, zero payload
 * copies or allocations. Optimization: weights remain immutable and shared by
 * borrowed views. Corpus consultation, implementation (full rules read),
 * 2026-10-18: cpp-guidelines R.1 (#rr-raii): lifetime encloses model and
 * artifact owners. cpp-perf-guidelines GPU.1: host checkpoint teardown cannot
 * trigger hidden transfers. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#include "models/checkpoint.h"

namespace dolittle::models {
BoundCheckpoint::BoundCheckpoint() = default;
BoundCheckpoint::~BoundCheckpoint() = default;
}  // namespace dolittle::models
