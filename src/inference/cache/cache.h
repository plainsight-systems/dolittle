/*
 * Target-generation interface and implementation contract.
 * Axis X2: logical ownership and validity of persistent attention K/V storage.
 * Inputs: verified checkpoint identity, sequence identity, numerical mode,
 * capacity C and each block's KV-head count/width. Output: a cache whose
 * committed length S identifies exactly the initialized prefix [0,S).
 * The sequence executor retains the exact consumed IDs; matching length alone
 * never permits reuse for different token contents, weights or execution mode.
 *
 * Algorithm/state contract:
 * 1. Allocate admitted capacity with committed length zero. Unwritten bytes
 *    are uninitialized and must not be read.
 * 2. For an append of B positive positions, require S+B <= C and no append in
 *    flight. Stage writes to logical positions [S,S+B) in every block.
 * 3. In that call, query position S+i may read keys [0,S+i], including its
 *    new self key. Previously committed rows remain immutable.
 * 4. Only after all block writes and dependent work complete successfully,
 *    commit the new shared length S+B. No block may advertise a longer prefix.
 * 5. On submission/device failure, drain pending work and invalidate the cache;
 *    do not reuse a partly written prefix. Recreate it for a new run.
 * Only the owning executor advances validity. This header declares metadata,
 * not a universal pointer/stride view. Concrete kinds expose their own device
 * bindings; attention preparation explicitly checks layout compatibility.
 *
 * Design and cost: logical bytes are sum_blocks(2*C*Hkv*d*element_bytes).
 * The initial contiguous kind reserves this once. Paged/growing kinds belong
 * beside it and may require new kernels; they do not change logical positions.
 * Optimization: keep K/V resident and append only new positions, rather than
 * recomputing or copying the entire prefix. Cache rows cannot alias transient
 * scratch. No sliding eviction, host offload or cache quantization is implied.
 * Dimensions/products/budgets are checked before allocation; metadata is owned.
 * Invalid specification -> invalid_argument; overflow/budget -> length_error;
 * allocation/device failure propagates. Device/queue outlive concrete storage.
 * Overlay: docs/architecture/transformer-execution.md, "Prefill and decoding"
 * and "Storage lifetimes"; file-mapping.md, "Cache strategies and compatible
 * attention". Corpus consultation, header design and implementation,
 * 2026-10-18: cpp-guidelines I.7 (#ri-post), R.1 (#rr-raii): completion
 * precedes shared-prefix commit; resource lifetime and failure invalidation are
 * explicit. cpp-perf-guidelines GPU.9: separate long-lived cache capacity from
 * scratch. Source:
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tensors/representations/representation.h"

namespace dolittle::inference::cache {
struct SequenceId {
  std::uint64_t value;
};
struct LayerShape {
  std::size_t kv_heads;
  std::size_t head_width;
};
struct Specification {
  std::string checkpoint_identity;
  SequenceId sequence;
  tensors::ComputationMode mode;
  std::size_t capacity;
  std::vector<LayerShape> layers;
};
struct Footprint {
  std::size_t payload_bytes;
  std::size_t allocation_bytes;
};
struct Prefix {
  std::size_t committed_positions;
  bool usable;
};
}  // namespace dolittle::inference::cache
