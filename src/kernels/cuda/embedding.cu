/*
 * Axis X5: token embedding lookup.
 * Role: turn token IDs into the residual vectors entering the first block.
 *
 * Inputs: tokens[N] contains validated uint32 vocabulary indices; W[V,D] is
 * an embedding table with V vocabulary entries and D features per entry.
 * Output: X[N,D], with the same storage type as W.
 *
 * Algorithm: for each token position p:
 * 1. Read vocabulary index t = tokens[p].
 * 2. Copy all D features of W[t,:] into X[p,:].
 * The copy preserves every storage bit. It performs no floating-point
 * arithmetic, normalization or text decoding.
 *
 * CUDA implementation: a 256-thread block processes one token position at a
 * time. Thread zero loads the token index into shared memory; after a barrier,
 * threads copy adjacent features. A second barrier protects the shared index
 * before the block advances to another token position.
 * Cost: O(N*D) copied elements and one shared uint32 per block.
 * Optimization: read the token index once per position and copy contiguous
 * features directly on device. Selected activation-row copies (GatherRows)
 * are implemented by prepared dispatch in operations.cpp, not this kernel.
 *
 * Overlay: docs/architecture/transformer-execution.md, "Full model flow".
 * Token indices must be below V; the executor validates them before upload.
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
 */
#include "kernels/cuda/common.cuh"

namespace dolittle::kernels::cuda {
namespace {
template <class Bits>
__global__ void embedding(const std::uint32_t* tokens, const Bits* weight,
                          Bits* output, std::size_t rows, std::size_t width) {
    __shared__ std::uint32_t token;
    for (std::size_t row = blockIdx.x; row < rows;) {
        if (threadIdx.x == 0) token = tokens[row];
        __syncthreads();
        const auto source = static_cast<std::size_t>(token) * width;
        for (std::size_t feature = threadIdx.x; feature < width; feature += blockDim.x) {
            output[row * width + feature] = weight[source + feature];
        }
        __syncthreads();
        if (rows - row <= gridDim.x) break;
        row += gridDim.x;
    }
}

template <class Bits>
void submit(const std::uint32_t* tokens, const void* weight, void* output,
            std::size_t rows, std::size_t width, cudaStream_t stream) {
    embedding<<<detail::grid_blocks(rows), detail::threads, 0, stream>>>(
        tokens, static_cast<const Bits*>(weight), static_cast<Bits*>(output), rows, width);
}
}  // namespace

void launch_embedding(tensors::ElementType type, const std::uint32_t* tokens,
                      const void* weight, void* output, std::size_t rows,
                      std::size_t width, cudaStream_t stream) {
    if (detail::is_bfloat16(type)) {
        submit<std::uint16_t>(tokens, weight, output, rows, width, stream);
    } else {
        submit<std::uint32_t>(tokens, weight, output, rows, width, stream);
    }
}
}  // namespace dolittle::kernels::cuda
