/*
 * Target-generation interface and implementation contract.
 * Axis X2: preallocated contiguous K/V storage, with explicit per-block views.
 * Inputs: cache Specification, byte budget, live Device and ordered Queue.
 * Output: owned storage plus borrowed block regions. Device and queue outlive
 * storage; all users finish before regions or storage are released. Moving an
 * owner while work references its regions is forbidden.
 *
 * Algorithm:
 * 1. For each block, calculate checked bytes for K[C,Hkv,d] and V[C,Hkv,d],
 *    align their ranges, and sum allocation bytes. Reject the budget before
 *    allocation; the returned estimate includes alignment, not driver overhead.
 * 2. Reserve one arena. Logical element (p,h,f) has element offset
 *    (p*Hkv+h)*d+f within that block's K or V range.
 * 3. Copy new rotated K and unrotated V rows into [S,S+B), preserving storage
 *    bits. A contiguous chunk is one copy per K/V range per block.
 * 4. Attention uses the full-capacity strides but reads only j <= S+i for
 *    query i. Capacity and valid length are different quantities.
 * 5. After successful queue completion, the executor advances the shared
 *    prefix. On any incomplete append, invalidate and drain before destruction.
 * The view exposes capacity and shape, not authority to publish a valid prefix.
 * The executor owns the prefix state and exact token history. Unwritten
 * capacity is never included as an attention key. No reset-and-reuse for a new
 * identity.
 *
 * CUDA implementation design: queue-ordered device copies after K RoPE and V
 * projection; attention streams the persistent rows directly. A future Metal
 * backend uses the same logical layout through its own resource/copy
 * operations. Cost: two copies of B*Hkv*d elements per block, plus one reserved
 * cache arena. For the selected BF16 36-block model, capacity costs 144 KiB per
 * position (576 MiB at C=4096); captures and scratch are additional
 * allocations. Optimization: no cache allocation or full-prefix copy during
 * decoding. Baseline tradeoff: this is deliberately simple, not an optimized
 * cache lifecycle across turns. Allocate once for one prompt/response turn,
 * reuse throughout prefill/decode, then release after final capture completion
 * and pending GPU work. A later turn gets a fresh allocation and logical
 * prefix. Reserving the maximum admitted length can leave unused capacity, and
 * every turn pays allocation/release cost. There is no allocation pooling or
 * prefix reuse across turns. Measure allocation/release time, peak and unused
 * capacity, prefill/decode latency and target-to-NLA memory pressure on
 * representative runs before adding reuse, pooling or another storage strategy.
 * No optimality claim or measured benefit for those alternatives is implied by
 * this design. Errors and ownership follow cache.h and gpu/device.h. Regions
 * are read-only views for attention; only append destinations receive mutable
 * access. Overlay: docs/architecture/transformer-execution.md, "Prefill and
 * decoding" and "Storage lifetimes". Corpus consultation, header design and
 * implementation, 2026-10-18: cpp-guidelines I.7 (#ri-post), R.1 (#rr-raii):
 * views borrow owned admitted storage; logical validity is separate from
 * physical capacity. cpp-guidelines Per.6 (#rper-measure): assess
 * lifecycle/capacity tradeoffs with representative runs before claiming an
 * alternative performs better. cpp-perf-guidelines GPU.9: allocate once and
 * separate persistent/transient data. Source:
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines GPU.1: append and attend without host K/V transfers.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once

#include <memory>

#include "gpu/device.h"
#include "inference/cache/cache.h"

namespace dolittle::inference::cache {
struct ContiguousLayerView {
  gpu::ConstDeviceRegion keys;
  gpu::ConstDeviceRegion values;
  std::size_t capacity;
  LayerShape shape;
};
class Contiguous {
 public:
  ~Contiguous();
  Contiguous(Contiguous&&) noexcept;
  Contiguous& operator=(Contiguous&&) noexcept;
  Contiguous(const Contiguous&) = delete;
  Contiguous& operator=(const Contiguous&) = delete;
  [[nodiscard]] const Specification& specification() const noexcept;
  [[nodiscard]] ContiguousLayerView layer(std::size_t block) const;
  void append(gpu::Queue&, std::size_t block, std::size_t first_position,
              std::size_t count, gpu::ConstDeviceRegion rotated_keys,
              gpu::ConstDeviceRegion values);

 private:
  struct Impl;
  explicit Contiguous(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
  friend std::unique_ptr<Contiguous> allocate_contiguous(gpu::Device&,
                                                         gpu::Queue&,
                                                         const Specification&,
                                                         std::size_t);
};
[[nodiscard]] Footprint estimate_contiguous(const Specification&);
[[nodiscard]] std::unique_ptr<Contiguous> allocate_contiguous(
    gpu::Device&, gpu::Queue&, const Specification&, std::size_t max_bytes);
}  // namespace dolittle::inference::cache
