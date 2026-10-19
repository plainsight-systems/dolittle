/*
 * Axis X5, private CUDA launch boundary for prepared dense operations.
 * Contract: preparation supplies BF16/FP32 type, positive dimensions with checked
 * tensor byte extents, aligned device pointers on one device and a live ordered
 * stream. Output does not overlap inputs. Tokens are validated uint32 indices;
 * RoPE width is even, positions do not overflow; Hq is divisible by Hkv.
 * Pointers borrow storage until stream completion. Calls enqueue only: no tensor
 * allocation, transfer, wait or ownership change. Unsupported type throws
 * invalid_argument; caller checks immediate CUDA launch status and waits for
 * asynchronous faults. No TF32 or precision fallback. GatherRows stays in the
 * prepared dispatch as checked contiguous-range device-to-device copies.
 * Algorithm responsibility: these declarations connect a prepared operation
 * to its CUDA implementation. Each launch selects the BF16 or FP32 kernel,
 * computes its grid from the validated dimensions, and enqueues it on the
 * supplied stream. Return means enqueued, not completed. The corresponding
 * .cu file defines that operation's input-to-output computation.
 * Overlay: docs/architecture/transformer-execution.md,
 * "One Qwen3 transformer block".
 * Cost: 256 threads/block, 1D grid capped at 65535 with grid-stride work; projection
 * uses 16x16 threads and 528 floats shared, RMSNorm 256 floats, attention
 * (2*width+259) floats dynamic shared, embedding one uint32. Other kernels use
 * no shared memory.
 * Preparation checks these device limits. No global workspace is required.
 * Optimization: keep values on device and amortize shape/capability validation.
 * Corpus consultation, header design and implementation, 2026-10-18:
 * cpp-guidelines I.1 (#ri-explicit), I.7 (#ri-post): explicit type/dimensions,
 * borrowing/completion preconditions and numerical-operation postconditions.
 * cpp-perf-guidelines GPU.1: no intermediate host round trips. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.2: contiguous features map to adjacent lanes. Source:
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 */
#pragma once

#include "tensors/representations/representation.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace dolittle::kernels::cuda {
void launch_projection(tensors::ElementType type, const void* input,
                       const void* weight, void* output, std::size_t rows,
                       std::size_t input_width, std::size_t output_width,
                       cudaStream_t stream);
void launch_rms_norm(tensors::ElementType type, const void* input,
                     const void* weight, void* output, std::size_t rows,
                     std::size_t width, float epsilon, cudaStream_t stream);
void launch_rotary(tensors::ElementType type, const void* input, void* output,
                   std::size_t positions, std::size_t heads, std::size_t width,
                   std::size_t first_position, double theta, cudaStream_t stream);
void launch_attention(tensors::ElementType type, const void* query, const void* key,
                      const void* value, void* output, std::size_t positions,
                      std::size_t query_heads, std::size_t kv_heads,
                      std::size_t width, cudaStream_t stream);
void launch_embedding(tensors::ElementType type, const std::uint32_t* tokens,
                      const void* weight, void* output, std::size_t rows,
                      std::size_t width, cudaStream_t stream);
void launch_add(tensors::ElementType type, const void* left, const void* right,
                void* output, std::size_t count, cudaStream_t stream);
void launch_silu_multiply(tensors::ElementType type, const void* gate, const void* up,
                          void* output, std::size_t count, cudaStream_t stream);
}  // namespace dolittle::kernels::cuda
