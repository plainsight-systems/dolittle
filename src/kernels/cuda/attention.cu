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
 *    square root scale once to FP32 on the host.
 * 2. Visit visible keys to find m=max(s[j]). Visit them again to compute
 *    e[j]=exp(s[j]-m) and denominator l=sum_j e[j]. Thirty-two partials
 *    own j modulo 32, merged with the descending half-reduction tree.
 * 3. Visit the keys a third time. Recompute each score and round
 *    p[j]=FP32(e[j]/l) to FP32 BEFORE weighting V. Accumulate
 *    Y[...,f]=sum_j p[j]*V[j,g,f] using two contiguous key ranges, the
 *    first of ceil((S+i+1)/2) keys. Each range uses ascending FP32 FMA;
 *    add its two sums in FP32, then round the completed vector to storage.
 * The probability rounding is explicit; carrying an unnormalized numerator
 * and dividing it at the end is a different floating-point computation.
 * CUDA implementation: 256 threads own one query/head. Q, the output sums
 * and a reduction buffer live in shared memory. Three passes recompute QK
 * dots rather than retaining an N-by-N score matrix. K is read three times,
 * V once; barriers protect shared reductions and scalar broadcasts.
 * Cost: O(B*(S+B)*Hq*d) work, with three score passes, and (3*d+259)
 * shared floats per block. This spends extra arithmetic to preserve the
 * probability rounding boundary without sequence-sized scratch.
 * Optimization: stage Q and sums on chip; directly index shared KV heads.
 * Cached K/V appends precede attention on the same queue. Query indices
 * remain local, KV indices absolute. Never read unwritten cache capacity.
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
 */
#include "kernels/cuda/common.cuh"

#include <cmath>

namespace dolittle::kernels::cuda {
namespace {
inline constexpr unsigned dot_partials = 8;
inline constexpr unsigned softmax_partials = 32;

template <class T> struct AttentionArguments {
    const T *query;
    const T *key;
    const T *value;
    T *output;
    std::size_t tasks;
    std::size_t query_heads;
    std::size_t kv_heads;
    std::size_t width;
    std::size_t first_position;
    float scale;
};

// Device-only indexing: preparation proves allocation extents; the loops
// below guard tails and index shared/register arrays within their declared sizes.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-constant-array-index, cppcoreguidelines-pro-bounds-array-to-pointer-decay)
template <class T> __global__ void attention(AttentionArguments<T> args) {
    const auto [query, key, value, output, tasks, query_heads, kv_heads, width, first_position, scale] = args;
    // CUDA dynamic shared memory is allocated per block, not process-global state.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    extern __shared__ float shared[];
    auto *q = shared;
    auto *numerator = q + width;
    auto *second_sum = numerator + width;
    auto *reduction = second_sum + width;
    auto *update = reduction + detail::threads;
    for (std::size_t task = blockIdx.x; task < tasks;) {
        const auto position = first_position + task / query_heads;
        const auto head = task % query_heads;
        const auto kv_head = head / (query_heads / kv_heads);
        for (std::size_t feature = threadIdx.x; feature < width; feature += blockDim.x) {
            q[feature] = detail::widen(query[task * width + feature]);
            numerator[feature] = 0.0F;
            second_sum[feature] = 0.0F;
        }
        __syncthreads();
        float maximum = 0.0F;
        float denominator_partial = 0.0F;
        for (unsigned pass = 0; pass < 3; ++pass) {
            for (std::size_t key_position = 0; key_position <= position; ++key_position) {
                const auto kv_offset = (key_position * kv_heads + kv_head) * width;
                float partial = 0.0F;
                if (threadIdx.x < dot_partials) {
                    const auto span = width / dot_partials + (width % dot_partials != 0);
                    const auto begin = threadIdx.x * span;
                    const auto end = begin < width ? begin + min(span, width - begin) : width;
                    for (std::size_t feature = begin; feature < end; ++feature) {
                        partial = __fmaf_rn(q[feature], detail::widen(key[kv_offset + feature]), partial);
                    }
                }
                const auto score = __fmul_rn(detail::block_sum(partial, reduction), scale);
                if (pass == 0) {
                    maximum = key_position == 0 ? score : fmaxf(maximum, score);
                } else if (pass == 1) {
                    if (threadIdx.x == key_position % softmax_partials) {
                        denominator_partial = __fadd_rn(denominator_partial, expf(score - maximum));
                    }
                } else {
                    if (threadIdx.x == 0) {
                        update[0] = expf(score - maximum) / update[2];
                    }
                    __syncthreads();
                    for (std::size_t feature = threadIdx.x; feature < width; feature += blockDim.x) {
                        auto *sum = key_position < (position / 2 + 1) ? numerator : second_sum;
                        sum[feature] =
                            __fmaf_rn(update[0], detail::widen(value[kv_offset + feature]), sum[feature]);
                    }
                }
                __syncthreads();
            }
            if (pass == 1) {
                const auto denominator = detail::block_sum(denominator_partial, reduction);
                if (threadIdx.x == 0) {
                    update[2] = denominator;
                }
                __syncthreads();
            }
        }
        for (std::size_t feature = threadIdx.x; feature < width; feature += blockDim.x) {
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

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic, cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-pro-bounds-constant-array-index, cppcoreguidelines-pro-bounds-array-to-pointer-decay)

template <class T>
void submit(const void *query, const void *key, const void *value, void *output, std::size_t positions,
            std::size_t query_heads, std::size_t kv_heads, std::size_t width, cudaStream_t stream,
            std::size_t first_position) {
    const auto tasks = positions * query_heads;
    const auto shared_bytes = (3 * width + detail::threads + 3) * sizeof(float);
    attention<<<detail::grid_blocks(tasks), detail::threads, shared_bytes, stream>>>(
        AttentionArguments<T>{.query = static_cast<const T *>(query),
                              .key = static_cast<const T *>(key),
                              .value = static_cast<const T *>(value),
                              .output = static_cast<T *>(output),
                              .tasks = tasks,
                              .query_heads = query_heads,
                              .kv_heads = kv_heads,
                              .width = width,
                              .first_position = first_position,
                              .scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(width)))});
}
} // namespace

void launch_attention(tensors::ElementType type, const void *query, const void *key, const void *value,
                      void *output, std::size_t positions, std::size_t query_heads, std::size_t kv_heads,
                      std::size_t width, cudaStream_t stream, std::size_t first_position) {
    if (detail::is_bfloat16(type)) {
        submit<__nv_bfloat16>(query, key, value, output, positions, query_heads, kv_heads, width, stream,
                              first_position);
    } else {
        submit<float>(query, key, value, output, positions, query_heads, kv_heads, width, stream,
                      first_position);
    }
}
} // namespace dolittle::kernels::cuda
