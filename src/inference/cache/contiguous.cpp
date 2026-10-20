/*
 * Contiguous cache allocation and ordered append implementation.
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
 * 3. Expose checked writable windows for rotated K and unrotated V. Their
 *    producers write active rows [S,S+B) directly; append also supports copying
 *    from scratch, with one copy per K/V range if direct placement cannot fit.
 * 4. Attention uses the full-capacity strides but reads only j <= S+i for
 *    query i. Capacity and valid length are different quantities.
 * 5. After successful queue completion, the executor advances the shared
 *    prefix. On any incomplete append, invalidate and drain before destruction.
 * The view exposes capacity and shape, not authority to publish a valid prefix.
 * The executor owns the prefix state and exact token history. Unwritten
 * capacity is never included as an attention key. No reset-and-reuse for a new
 * identity.
 *
 * Implementation design: producers or ordered append copies initialize the
 * cache rows; attention streams the persistent rows directly. A future Metal
 * backend uses the same logical layout through its own resource/copy
 * operations. Cost: append fallback copies 2*B*Hkv*d elements per block;
 * direct placement avoids those copies. Both use one reserved
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
 * are read-only views for attention; only checked write windows receive
 * mutable access. Overlay: docs/architecture/transformer-execution.md, "Prefill
 * and decoding" and "Storage lifetimes". Corpus consultation, implementation,
 * 2026-10-18: cpp-guidelines I.7 (#ri-post), R.1 (#rr-raii): views borrow owned
 * admitted storage; logical validity is separate from physical capacity.
 * cpp-guidelines Per.6 (#rper-measure): assess lifecycle/capacity tradeoffs
 * with representative runs before claiming an alternative performs better.
 * cpp-perf-guidelines GPU.9: allocate once and separate persistent/transient
 * data. Source:
 * guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 * cpp-perf-guidelines GPU.1: append and attend without host K/V transfers.
 * Source: guidelines/gpu/GPU.1-keep-data-on-device.md.
 * Direct-cache-write design and implementation, 2026-10-19:
 * Borrow a checked mutable K/V window [S,S+Bmax) on the owning queue. Bind
 * the recipe's rotated-key and value tensors to those destinations for their
 * producers, later readers and captures. Kernels write only the B active rows;
 * attention reads only [0,S+B). Successful completion still precedes commit.
 * When fewer than Bmax capacity rows remain, retain scratch and append copies
 * so prepared binding extents cannot overrun the cache. No allocation occurs
 * during consume; the cache remains contiguous, BF16/FP32 and turn-owned.
 * Optimization: direct placement removes two device copies per block per
 * consumption when the prepared window fits, without changing tensor values.
 * Existing scratch remains admitted for the bounded tail fallback.
 * cpp-guidelines I.7 (#ri-post): mutable windows grant no prefix-commit right.
 * cpp-perf-guidelines GPU.9: bind producers into persistent admitted storage;
 * preserve lifetime and disjointness from transient scratch.
 * Source: guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md.
 */
#include "inference/cache/contiguous.h"

#include <limits>
#include <stdexcept>
#include <utility>

#include "tensors/tensor.h"
namespace dolittle::inference::cache {
namespace {
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::length_error("cache size overflow");
  }
  return a * b;
}
std::size_t sum(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::length_error("cache size overflow");
  }
  return a + b;
}
}  // namespace
Footprint estimate_contiguous(const Specification& s) {
  if (s.checkpoint_identity.empty() || !s.capacity || s.layers.empty() ||
      (s.mode.activation_storage != tensors::ElementType::bfloat16 &&
       s.mode.activation_storage != tensors::ElementType::float32) ||
      s.mode.reduction_accumulation != tensors::ElementType::float32) {
    throw std::invalid_argument("invalid contiguous cache specification");
  }
  std::size_t total = 0;
  for (auto layer : s.layers) {
    if (!layer.kv_heads || !layer.head_width) {
      throw std::invalid_argument("empty cache layer");
    }
    total = sum(total,
                product(product(product(product(2, s.capacity), layer.kv_heads),
                                layer.head_width),
                        tensors::element_bytes(s.mode.activation_storage)));
  }
  return {total, total};
}
struct Contiguous::Impl {
  Specification spec;
  gpu::Queue* queue;
  gpu::DeviceBuffer storage;
  std::vector<std::size_t> offsets, bytes;
  Impl(Specification s, gpu::Queue& q, gpu::DeviceBuffer buffer)
      : spec(std::move(s)), queue(&q), storage(std::move(buffer)) {
    std::size_t offset = 0;
    for (auto layer : spec.layers) {
      auto size = product(
          product(product(spec.capacity, layer.kv_heads), layer.head_width),
          tensors::element_bytes(spec.mode.activation_storage));
      offsets.push_back(offset);
      bytes.push_back(size);
      offset = sum(offset, product(2, size));
    }
  }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  ~Impl() {
    try {
      queue->drain();
      // Destruction cannot throw; the buffer owner performs a final no-throw
      // drain. NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
    }
  }
};
Contiguous::Contiguous(std::unique_ptr<Impl> p) : impl_(std::move(p)) {}
Contiguous::~Contiguous() = default;
Contiguous::Contiguous(Contiguous&&) noexcept = default;
Contiguous& Contiguous::operator=(Contiguous&&) noexcept = default;
const Specification& Contiguous::specification() const noexcept {
  return impl_->spec;
}
ContiguousLayerView Contiguous::layer(std::size_t block) const {
  if (block >= impl_->bytes.size()) {
    throw std::invalid_argument("cache layer out of range");
  }
  const auto n = impl_->bytes.at(block), offset = impl_->offsets.at(block);
  return {{&impl_->storage, offset, n},
          {&impl_->storage, offset + n, n},
          impl_->spec.capacity,
          impl_->spec.layers.at(block)};
}
ContiguousWriteWindow Contiguous::writable_window(gpu::Queue& queue,
                                                  std::size_t block,
                                                  std::size_t first,
                                                  std::size_t count) {
  if (&queue != impl_->queue || !count || first > impl_->spec.capacity ||
      count > impl_->spec.capacity - first) {
    throw std::invalid_argument("cache append queue or range mismatch");
  }
  const auto view = layer(block);
  const auto row = impl_->bytes.at(block) / impl_->spec.capacity;
  const auto size = product(count, row), offset = product(first, row);
  return {{&impl_->storage, view.keys.offset + offset, size},
          {&impl_->storage, view.values.offset + offset, size}};
}
void Contiguous::append(gpu::Queue& queue, std::size_t block, std::size_t first,
                        std::size_t count, gpu::ConstDeviceRegion k,
                        gpu::ConstDeviceRegion v) {
  const auto window = writable_window(queue, block, first, count);
  if (k.bytes != window.keys.bytes || v.bytes != window.values.bytes) {
    throw std::invalid_argument("cache append source extent mismatch");
  }
  queue.copy(window.keys, k);
  queue.copy(window.values, v);
}

std::unique_ptr<Contiguous> allocate_contiguous(gpu::Device& device,
                                                gpu::Queue& queue,
                                                const Specification& spec,
                                                std::size_t budget) {
  auto footprint = estimate_contiguous(spec);
  if (footprint.allocation_bytes > budget) {
    throw std::length_error("cache exceeds device budget");
  }
  return std::unique_ptr<Contiguous>(
      new Contiguous(std::make_unique<Contiguous::Impl>(
          spec, queue,
          device.allocate(
              {footprint.allocation_bytes,
               tensors::element_bytes(spec.mode.activation_storage)}))));
}
}  // namespace dolittle::inference::cache
