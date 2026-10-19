/*
 * Axis X5: full-sequence causal grouped-query attention.
 * Inputs: Q[N,Hq,d], K[N,Hkv,d] and V[N,Hkv,d], where N is the sequence
 * length, d is the head width, and Hq is a positive multiple of Hkv.
 * Output: Y[N,Hq,d] in the same BF16 or FP32 storage type as the inputs.
 *
 * Algorithm: for each query position p and query head h:
 * 1. Select KV head g = floor(h / (Hq/Hkv)). Queries in the same group
 *    share this head; K and V do not need duplicated head arrays.
 * 2. For each key position j in [0,p], compute the FP32 score
 *    s_j = dot(Q[p,h,:], K[j,g,:]) / sqrt(d).
 * 3. Compute softmax over those scores and use its probabilities to weight
 *    the corresponding V vectors. The result is
 *    Y[p,h,:] = sum_j(exp(s_j-m) * V[j,g,:]) / sum_j(exp(s_j-m)),
 *    where m is the maximum score for this query.
 * 4. Round the resulting vector to storage. No scores, probabilities or
 *    intermediate weighted sums are rounded to BF16.
 * Only j <= p participates: a query cannot see future positions. Inputs
 * are unpadded, so every query has at least key position zero.
 *
 * Streaming implementation of step 3: retain a maximum m, scalar denominator
 * l and vector numerator a instead of storing all scores. For the first key,
 * set m = s_0, l = 1 and a = V[0,g,:]. For each later key j:
 *   next_m = max(m, s_j)
 *   alpha = exp(m - next_m); beta = exp(s_j - next_m)
 *   l = alpha*l + beta
 *   a = alpha*a + beta*V[j,g,:]
 *   m = next_m
 * The rescaling keeps earlier contributions expressed relative to the new
 * maximum. After the last key, store a/l. All carried state is FP32.
 *
 * CUDA implementation: a 256-thread block handles one query position/head
 * at a time. It retains Q and a in shared memory, streams one K/V row at a
 * time, and reduces the dot product across feature threads. Thread zero
 * updates the scalar state; barriers make each update visible before the
 * next key. Blocks advance to more queries when needed.
 * Cost: O(N*N*Hq*d) work and (2*d+259) shared floats per block.
 * Optimization: keep query and running state on chip, read contiguous K/V
 * features, and avoid an N-by-N score/probability buffer or expanded KV heads.
 * Reduction order can differ from other attention backends; equivalence is
 * numerical within validated bounds, not a bitwise guarantee.
 *
 * Overlay: docs/architecture/transformer-execution.md,
 * "Attention and positional semantics".
 *
 * Contract: preparation validates shapes, byte/launch extents and device limits.
 * Inputs and the separate output buffer are borrowed until queue completion;
 * output must not overlap an input. Launches allocate no buffers and perform
 * no host readback or host wait. The caller checks immediate launch status;
 * asynchronous device errors surface at the explicit queue wait. See
 * kernels/operations.h for the public operation and failure contracts.
 *
 * Corpus consultation, comment design, 2026-10-18:
 * cpp-guidelines I.7 (#ri-post): state the output and rounding guarantees.
 * cpp-perf-guidelines GPU.2: contiguous features map to adjacent threads.
 * Source: guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 * cpp-perf-guidelines GDSA.16: combine each incoming key with small on-chip
 * state rather than materializing the score matrix.
 * Source: guidelines/gpu-dsa/GDSA.16-tile-through-on-chip-memory.md.
 */
#include "kernels/cuda/common.cuh"

namespace dolittle::kernels::cuda {
namespace {
template <class T>
__global__ void attention(const T* query, const T* key, const T* value, T* output,
                          std::size_t tasks, std::size_t query_heads,
                          std::size_t kv_heads, std::size_t width) {
    extern __shared__ float shared[];
    auto* q = shared;
    auto* numerator = q + width;
    auto* reduction = numerator + width;
    auto* update = reduction + detail::threads;
    const auto scale = 1.0F / sqrtf(static_cast<float>(width));
    for (std::size_t task = blockIdx.x; task < tasks;) {
        const auto position = task / query_heads;
        const auto head = task % query_heads;
        const auto kv_head = head / (query_heads / kv_heads);
        for (std::size_t feature = threadIdx.x; feature < width; feature += blockDim.x) {
            q[feature] = detail::widen(query[task * width + feature]);
            numerator[feature] = 0.0F;
        }
        if (threadIdx.x == 0) update[2] = 0.0F;
        __syncthreads();
        float maximum = 0.0F;  // Only lane zero owns the running maximum.
        for (std::size_t key_position = 0; key_position <= position; ++key_position) {
            const auto kv_offset = (key_position * kv_heads + kv_head) * width;
            float partial = 0.0F;
            for (std::size_t feature = threadIdx.x; feature < width; feature += blockDim.x) {
                partial = fmaf(q[feature], detail::widen(key[kv_offset + feature]), partial);
            }
            const auto score = __fmul_rn(detail::block_sum(partial, reduction), scale);
            if (threadIdx.x == 0) {
                if (key_position == 0) {
                    maximum = score;
                    update[0] = 0.0F;
                    update[1] = 1.0F;
                } else {
                    const auto next_maximum = fmaxf(maximum, score);
                    update[0] = expf(maximum - next_maximum);
                    update[1] = expf(score - next_maximum);
                    maximum = next_maximum;
                }
                update[2] = fmaf(update[0], update[2], update[1]);
            }
            __syncthreads();
            for (std::size_t feature = threadIdx.x; feature < width; feature += blockDim.x) {
                numerator[feature] = fmaf(update[1], detail::widen(value[kv_offset + feature]),
                                          __fmul_rn(update[0], numerator[feature]));
            }
            __syncthreads();
        }
        for (std::size_t feature = threadIdx.x; feature < width; feature += blockDim.x) {
            output[task * width + feature] = detail::narrow<T>(numerator[feature] / update[2]);
        }
        __syncthreads();
        if (tasks - task <= gridDim.x) break;
        task += gridDim.x;
    }
}

template <class T>
void submit(const void* query, const void* key, const void* value, void* output,
            std::size_t positions, std::size_t query_heads, std::size_t kv_heads,
            std::size_t width, cudaStream_t stream) {
    const auto tasks = positions * query_heads;
    const auto shared_bytes = (2 * width + detail::threads + 3) * sizeof(float);
    attention<<<detail::grid_blocks(tasks), detail::threads, shared_bytes, stream>>>(
        static_cast<const T*>(query), static_cast<const T*>(key), static_cast<const T*>(value),
        static_cast<T*>(output), tasks, query_heads, kv_heads, width);
}
}  // namespace

void launch_attention(tensors::ElementType type, const void* query, const void* key,
                      const void* value, void* output, std::size_t positions,
                      std::size_t query_heads, std::size_t kv_heads,
                      std::size_t width, cudaStream_t stream) {
    if (detail::is_bfloat16(type)) {
        submit<__nv_bfloat16>(query, key, value, output, positions, query_heads, kv_heads, width, stream);
    } else {
        submit<float>(query, key, value, output, positions, query_heads, kv_heads, width, stream);
    }
}
}  // namespace dolittle::kernels::cuda
