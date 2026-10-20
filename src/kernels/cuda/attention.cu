/*
 * Axis X5: full or cached causal grouped-query attention.
 * Inputs: Q[B,Hq,d], K[S+B,Hkv,d], V[S+B,Hkv,d]. S is the cached prefix
 * (zero for a full forward pass); Hq is a positive multiple of Hkv.
 * Output: Y[B,Hq,d] in the same BF16 or FP32 storage type.
 * Algorithm for local query i and head h:
 * 1. Select KV head g=floor(h/(Hq/Hkv)); visible keys are j=0..S+i.
 *    Score s[j]=dot(Q[i,h,:],K[j,g,:])/sqrt(d) in FP32. Dot products
 *    use eight contiguous feature ranges of ceil(d/8) features, visited in
 *    ascending order with FP32 FMA. Merge at offsets 4,2,1. Round the inverse
 *    square root scale once to FP32 in a device preparation kernel. The owned
 *    scalar is initialized before graph readers; no host tensor arithmetic.
 * 2. Compute m=max(s[j]), e[j]=exp(s[j]-m), l=sum_j e[j]. Thirty-two
 *    denominator partials own j modulo 32, merged at offsets 16,8,4,2,1.
 * 3. Round p[j]=FP32(e[j]/l) BEFORE weighting V. Each output feature
 *    accumulates two contiguous key ranges, the first ceil((S+i+1)/2)
 *    keys. Each range uses ascending FP32 FMA; add the two sums in FP32
 *    and round to output storage. Dividing an unnormalized final numerator
 *    would be a different floating-point computation.
 * CUDA implementation: one 256-thread block owns one query/head. For up to
 * 8192 visible keys and d<=1024, eight-lane groups compute independent QK
 * dots in parallel. Store each score once in shared memory, reduce the
 * maximum, replace scores with probabilities, then traverse V with adjacent
 * lanes owning adjacent features. Q and probabilities are shared; output
 * accumulators live in registers. Synchronization occurs between phases,
 * never once per key. Shared bytes are 4*(d+N+256), at most 37 KiB, within
 * the 48 KiB per-block baseline of supported compute capabilities >=8.6.
 * Larger shapes retain the bounded-memory recomputation path: three score
 * passes with 4*(3*d+259) shared bytes, as checked during preparation.
 * Cost: O(B*(S+B)*Hq*d), one K read and one V read on the retained-score path;
 * no global score matrix, allocation, extra dispatch or host synchronization.
 * Optimization: reuse QK scores on chip and parallelize independent keys;
 * fixed dot/denominator/value reduction orders retain cached/full equality.
 * Charlotte's attention kernels and Vigil's MLX fast SDPA dispatch motivate
 * dedicated attention execution rather than serialized per-key reductions.
 * Optimization corpus consultation, header design and implementation,
 * 2026-10-19: cpp-guidelines I.7 (#ri-post): preserve FP32 probability
 * rounding. cpp-perf-guidelines GPU.5: retain reusable scores and bound shared
 * storage. Source:
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md.
 * cpp-perf-guidelines GDSA.2: retain exact partial ownership and sum trees.
 * Source: guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 * Cached K/V appends precede attention on the same queue. Query indices
 * remain local, KV indices absolute. Never read unwritten cache capacity.
 * Value-reduction design and implementation, 2026-10-20: for d<=128, assign
 * half the block to each existing contiguous key range. Adjacent lanes still
 * read adjacent V features; both halves accumulate concurrently into FP32
 * registers. Reuse the expired 256-float reduction buffer for their sums, then
 * add the two corresponding features and narrow once. Wider heads use the
 * existing loop. No new scratch, split count or arithmetic reassociation is
 * introduced. cpp-guidelines I.7 (#ri-post): preserve the two original PV
 * partials. cpp-perf-guidelines GPU.5: reuse expired shared scratch at a phase
 * barrier. Source:
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md.
 *
 * Optimization consultation, design and implementation, 2026-10-20:
 * Aligned BF16 heads of width 128 retain each lane's 16 query features in
 * registers and load each 16-feature key range with two 128-bit loads.
 * Unpack the original BF16 bits to FP32 and execute the same 16 ordered FMAs.
 * No query/key arithmetic, partial ownership, merge, softmax or PV boundary
 * changes. Other shapes or unaligned views retain scalar loads. The explicit
 * PTX loads prevent scalarization; no additional shared/global storage is used.
 * Core I.7 (#ri-post) preserves stored results; Per.11 (#rper-comp) fixes the
 * 16-feature loop. Performance GPU.2 and GDSA.2 preserve contiguous loads and
 * the fixed reduction order. Source:
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 *
 * Overlay: docs/architecture/transformer-execution.md, "One Qwen3 transformer
 * block" and "Attention and positional semantics".
 * Contract: preparation validates shapes, extents and device limits. Inputs
 * and the separate output buffer are borrowed until queue completion; output
 * cannot overlap inputs. Launches allocate no buffers, read nothing to the
 * host and perform no host wait. The caller checks immediate launch status;
 * asynchronous errors surface at the explicit queue wait.
 * Same-build/device repeatability and row/chunk invariance are required.
 * Cross-GPU bit equality is unverified. No TF32 or FP64 tensor accumulation.
 * Corpus consultation, header design and implementation, 2026-10-19:
 * cpp-guidelines I.7 (#ri-post): specify intermediate and final rounding.
 * cpp-perf-guidelines GDSA.2: fix partial ownership and combination order
 * independently of batch size, scheduling and cache chunks.
 * Source: guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 * cpp-perf-guidelines GDSA.16: bound on-chip state and account for rereads.
 * Source: guidelines/gpu-dsa/GDSA.16-tile-through-on-chip-memory.md.
 * Static-analysis design/implementation, 2026-10-19: kernel arguments use
 * named fields to distinguish operand and dimension roles. Core I.30
 * (#ri-encapsulate) confines native pointer/array access to the guarded kernel;
 * GDSA.2 preserves the exact partial ownership and arithmetic order. Source:
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md.
 * Device-loop extension (design/implementation, 2026-10-20): a prepared
 * prefill/decode body borrows an immutable-for-this-iteration device position.
 * The graph advances it only after every reader completes. RoPE/attention use
 * that absolute position; positioned output bindings address cache rows
 * S..S+B-1. Arithmetic and reduction order are unchanged. Graph storage is
 * admitted for the maximum position; the device loop enforces S+B <= capacity.
 * No per-token host argument update, copy of position, or host launch is
 * required. Core I.7/CP.3: explicit bounds and ordered borrowing. Performance
 * GPU.1/GPU.6: resident control and stable graph arguments
 * (guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md). Both corpora queried and rules
 * read during header design and implementation.
 */
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "kernels/cuda/common.cuh"

namespace dolittle::kernels::cuda {
namespace {
inline constexpr std::size_t packed_width = 128;
inline constexpr unsigned packed_features = 16;
inline constexpr unsigned bf16_shift = 16;
inline constexpr unsigned bf16_high_mask = 0xffff0000U;
inline constexpr unsigned second_word = 5;
inline constexpr unsigned third_word = 6;
inline constexpr unsigned fourth_word = 7;
inline constexpr unsigned dot_partials = 8;
inline constexpr unsigned softmax_partials = 32;
inline constexpr unsigned full_warp_mask = 0xffffffffU;

template <class T>
struct AttentionArguments {
  const T* query;
  const T* key;
  const T* value;
  T* output;
  std::size_t tasks;
  std::size_t query_heads;
  std::size_t kv_heads;
  std::size_t width;
  std::size_t first_position;
  const float* scale;
  const std::size_t* device_position;
};

// Device-only indexing: preparation proves allocation extents; the loops
// below guard tails and index shared/register arrays within their declared
// sizes. NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-pro-bounds-constant-array-index,
// cppcoreguidelines-pro-bounds-array-to-pointer-decay)
template <class T>
__global__ void attention(AttentionArguments<T> args) {
  const auto [query, key, value, output, tasks, query_heads, kv_heads, width,
              host_position, scale, device_position] = args;
  const auto first_position =
      device_position ? *device_position : host_position;
  // CUDA dynamic shared memory is allocated per block, not process-global
  // state. NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  extern __shared__ float shared[];
  auto* q = shared;
  auto* numerator = q + width;
  auto* second_sum = numerator + width;
  auto* reduction = second_sum + width;
  auto* update = reduction + detail::threads;
  for (std::size_t task = blockIdx.x; task < tasks;) {
    const auto position = first_position + task / query_heads;
    const auto head = task % query_heads;
    const auto kv_head = head / (query_heads / kv_heads);
    for (std::size_t feature = threadIdx.x; feature < width;
         feature += blockDim.x) {
      q[feature] = detail::widen(query[task * width + feature]);
      numerator[feature] = 0.0F;
      second_sum[feature] = 0.0F;
    }
    __syncthreads();
    float maximum = 0.0F;
    float denominator_partial = 0.0F;
    for (unsigned pass = 0; pass < 3; ++pass) {
      for (std::size_t key_position = 0; key_position <= position;
           ++key_position) {
        const auto kv_offset = (key_position * kv_heads + kv_head) * width;
        float partial = 0.0F;
        if (threadIdx.x < dot_partials) {
          const auto span = width / dot_partials + (width % dot_partials != 0);
          const auto begin = threadIdx.x * span;
          const auto end =
              begin < width ? begin + min(span, width - begin) : width;
          for (std::size_t feature = begin; feature < end; ++feature) {
            partial = __fmaf_rn(
                q[feature], detail::widen(key[kv_offset + feature]), partial);
          }
        }
        const auto score =
            __fmul_rn(detail::block_sum(partial, reduction), *scale);
        if (pass == 0) {
          maximum = key_position == 0 ? score : fmaxf(maximum, score);
        } else if (pass == 1) {
          if (threadIdx.x == key_position % softmax_partials) {
            denominator_partial =
                __fadd_rn(denominator_partial, expf(score - maximum));
          }
        } else {
          if (threadIdx.x == 0) {
            update[0] = expf(score - maximum) / update[2];
          }
          __syncthreads();
          for (std::size_t feature = threadIdx.x; feature < width;
               feature += blockDim.x) {
            auto* sum =
                key_position < (position / 2 + 1) ? numerator : second_sum;
            sum[feature] =
                __fmaf_rn(update[0], detail::widen(value[kv_offset + feature]),
                          sum[feature]);
          }
        }
        __syncthreads();
      }
      if (pass == 1) {
        const auto denominator =
            detail::block_sum(denominator_partial, reduction);
        if (threadIdx.x == 0) {
          update[2] = denominator;
        }
        __syncthreads();
      }
    }
    for (std::size_t feature = threadIdx.x; feature < width;
         feature += blockDim.x) {
      output[task * width + feature] =
          detail::narrow<T>(__fadd_rn(numerator[feature], second_sum[feature]));
    }
    __syncthreads();
    if (tasks - task <= gridDim.x) {
      break;
    }
    task += gridDim.x;
  }
}

template <class T, bool PackedKeys = false>
__global__ void attention_retained(AttentionArguments<T> args,
                                   std::size_t key_capacity) {
  // Per-block dynamic shared storage, not process-global mutable state.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  extern __shared__ float shared[];
  auto* q = shared;
  auto* scores = q + args.width;
  auto* reduction = scores + key_capacity;
  const auto lane = threadIdx.x % dot_partials;
  const auto group = threadIdx.x / dot_partials;
  constexpr unsigned groups = detail::threads / dot_partials;
  const auto span =
      args.width / dot_partials + (args.width % dot_partials != 0);
  for (std::size_t task = blockIdx.x; task < args.tasks; task += gridDim.x) {
    const auto first_position =
        args.device_position ? *args.device_position : args.first_position;
    const auto count = first_position + task / args.query_heads + 1;
    const auto kv_head =
        (task % args.query_heads) / (args.query_heads / args.kv_heads);
    for (std::size_t feature = threadIdx.x; feature < args.width;
         feature += blockDim.x) {
      q[feature] = detail::widen(args.query[task * args.width + feature]);
    }
    __syncthreads();
    float query_features[packed_features];
    if constexpr (PackedKeys) {
#pragma unroll
      for (unsigned item = 0; item < packed_features; ++item) {
        query_features[item] = q[lane * packed_features + item];
      }
    }
    // Whole warps execute all shuffles, including groups beyond the key tail.
    for (std::size_t base = 0; base < count; base += groups) {
      const auto key_position = base + group;
      float partial = 0.0F;
      if (key_position < count) {
        const auto kv_offset =
            (key_position * args.kv_heads + kv_head) * args.width;
        const auto begin = lane * span;
        const auto end = min(begin + span, args.width);
        if constexpr (PackedKeys) {
          unsigned words[packed_features / 2];
          // The dispatch admits 16-byte alignment; each range is 32 bytes.
          // Explicit vector loads prevent scalarization into 16 strided loads.
          const auto* source = args.key + kv_offset + begin;
          asm("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
              : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3])
              : "l"(source));
          asm("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
              : "=r"(words[4]), "=r"(words[second_word]),
                "=r"(words[third_word]), "=r"(words[fourth_word])
              : "l"(source + packed_features / 2));
#pragma unroll
          for (unsigned item = 0; item < packed_features; ++item) {
            const auto bits = item % 2 == 0 ? words[item / 2] << bf16_shift
                                            : words[item / 2] & bf16_high_mask;
            partial =
                __fmaf_rn(query_features[item], __uint_as_float(bits), partial);
          }
        } else {
          for (std::size_t feature = begin; feature < end; ++feature) {
            partial = __fmaf_rn(q[feature],
                                detail::widen(args.key[kv_offset + feature]),
                                partial);
          }
        }
      }
#pragma unroll
      for (unsigned offset = dot_partials / 2; offset != 0; offset /= 2) {
        partial = __fadd_rn(partial, __shfl_down_sync(full_warp_mask, partial,
                                                      offset, dot_partials));
      }
      if (lane == 0 && key_position < count) {
        scores[key_position] = __fmul_rn(partial, *args.scale);
      }
    }
    __syncthreads();
    float maximum = scores[0];
    for (std::size_t j = threadIdx.x; j < count; j += blockDim.x) {
      maximum = fmaxf(maximum, scores[j]);
    }
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned offset = detail::threads / 2; offset != 0; offset /= 2) {
      if (threadIdx.x < offset) {
        reduction[threadIdx.x] =
            fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
      }
      __syncthreads();
    }
    maximum = reduction[0];
    // All lanes consume the maximum before the reduction buffer is reused.
    __syncthreads();
    float denominator_partial = 0.0F;
    if (threadIdx.x < softmax_partials) {
      for (std::size_t j = threadIdx.x; j < count; j += softmax_partials) {
        const auto exponential = expf(scores[j] - maximum);
        scores[j] = exponential;
        denominator_partial = __fadd_rn(denominator_partial, exponential);
      }
    }
    const auto denominator = detail::block_sum(denominator_partial, reduction);
    for (std::size_t j = threadIdx.x; j < count; j += blockDim.x) {
      scores[j] = scores[j] / denominator;
    }
    __syncthreads();
    const auto split = (count - 1) / 2 + 1;
    constexpr unsigned half_block = detail::threads / 2;
    if (args.width <= half_block) {
      const auto feature = threadIdx.x % half_block;
      const auto half = threadIdx.x / half_block;
      const auto begin = half == 0 ? 0 : split;
      const auto end = half == 0 ? split : count;
      float sum = 0.0F;
      if (feature < args.width) {
        for (std::size_t j = begin; j < end; ++j) {
          const auto offset =
              (j * args.kv_heads + kv_head) * args.width + feature;
          sum = __fmaf_rn(scores[j], detail::widen(args.value[offset]), sum);
        }
      }
      reduction[threadIdx.x] = sum;
      __syncthreads();
      if (threadIdx.x < args.width) {
        args.output[task * args.width + threadIdx.x] =
            detail::narrow<T>(__fadd_rn(reduction[threadIdx.x],
                                        reduction[threadIdx.x + half_block]));
      }
    } else {
      for (std::size_t feature = threadIdx.x; feature < args.width;
           feature += blockDim.x) {
        float first = 0.0F;
        float second = 0.0F;
        for (std::size_t j = 0; j < split; ++j) {
          const auto offset =
              (j * args.kv_heads + kv_head) * args.width + feature;
          first =
              __fmaf_rn(scores[j], detail::widen(args.value[offset]), first);
        }
        for (std::size_t j = split; j < count; ++j) {
          const auto offset =
              (j * args.kv_heads + kv_head) * args.width + feature;
          second =
              __fmaf_rn(scores[j], detail::widen(args.value[offset]), second);
        }
        args.output[task * args.width + feature] =
            detail::narrow<T>(__fadd_rn(first, second));
      }
    }
    __syncthreads();
  }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-pro-bounds-constant-array-index,
// cppcoreguidelines-pro-bounds-array-to-pointer-decay)

template <class T>
void submit(const void* query, const void* key, const void* value, void* output,
            std::size_t positions, std::size_t query_heads,
            std::size_t kv_heads, std::size_t width, cudaStream_t stream,
            std::size_t first_position, const std::size_t* device_position,
            std::size_t capacity, const float* scale) {
  const auto tasks = positions * query_heads;
  const AttentionArguments<T> args{.query = static_cast<const T*>(query),
                                   .key = static_cast<const T*>(key),
                                   .value = static_cast<const T*>(value),
                                   .output = static_cast<T*>(output),
                                   .tasks = tasks,
                                   .query_heads = query_heads,
                                   .kv_heads = kv_heads,
                                   .width = width,
                                   .first_position = first_position,
                                   .scale = scale,
                                   .device_position = device_position};
  const auto key_capacity =
      device_position ? capacity : first_position + positions;
  if (key_capacity <= attention_retained_keys &&
      width <= attention_retained_width) {
    const auto shared_bytes =
        (width + key_capacity + detail::threads) * sizeof(float);
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      // Alignment test only, no object-type conversion.
      // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
      const bool aligned =
          reinterpret_cast<std::uintptr_t>(key) % alignof(uint4) == 0;
      // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
      if (width == packed_width && aligned) {
        attention_retained<T, true>
            <<<detail::grid_blocks(tasks), detail::threads, shared_bytes,
               stream>>>(args, key_capacity);
        return;
      }
    }
    attention_retained<T>
        <<<detail::grid_blocks(tasks), detail::threads, shared_bytes, stream>>>(
            args, key_capacity);
  } else {
    const auto shared_bytes = (3 * width + detail::threads + 3) * sizeof(float);
    attention<<<detail::grid_blocks(tasks), detail::threads, shared_bytes,
                stream>>>(args);
  }
}
__global__ void prepare_scale(float* output, std::size_t width) {
  *output =
      __double2float_rn(__ddiv_rn(1.0, __dsqrt_rn(static_cast<double>(width))));
}
}  // namespace

void launch_attention_scale(float* output, std::size_t width,
                            cudaStream_t stream) {
  prepare_scale<<<1, 1, 0, stream>>>(output, width);
}
void launch_attention(tensors::ElementType type, const void* query,
                      const void* key, const void* value, void* output,
                      std::size_t positions, std::size_t query_heads,
                      std::size_t kv_heads, std::size_t width,
                      cudaStream_t stream, std::size_t first_position,
                      const std::size_t* device_position, std::size_t capacity,
                      const float* scale) {
  if (detail::is_bfloat16(type)) {
    submit<__nv_bfloat16>(query, key, value, output, positions, query_heads,
                          kv_heads, width, stream, first_position,
                          device_position, capacity, scale);
  } else {
    submit<float>(query, key, value, output, positions, query_heads, kv_heads,
                  width, stream, first_position, device_position, capacity,
                  scale);
  }
}
}  // namespace dolittle::kernels::cuda
