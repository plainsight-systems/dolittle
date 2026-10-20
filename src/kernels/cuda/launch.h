/*
 * Axis X5, private CUDA launch boundary for prepared dense operations.
 * Contract: preparation supplies BF16/FP32 type, positive dimensions with
 * checked tensor byte extents, aligned device pointers on one device and a live
 * ordered stream. Output does not overlap inputs. Tokens are validated uint32
 * indices; RoPE width is even, positions do not overflow; Hq is divisible by
 * Hkv. Attention first_position S makes local query i attend through key S+i.
 * Its K/V pointers contain every initialized row [0,S+B); query/output store
 * only the B active rows. S=0 supplies the existing full-forward behavior.
 * Pointers borrow storage until stream completion. Calls enqueue only: no
 * tensor allocation, transfer, wait or ownership change. Unsupported type
 * throws invalid_argument; caller checks immediate CUDA launch status and waits
 * for asynchronous faults. No TF32 or precision fallback. GatherRows stays in
 * the prepared dispatch as checked contiguous-range device-to-device copies.
 * Algorithm responsibility: these declarations connect a prepared operation
 * to its CUDA implementation. Each launch selects the BF16 or FP32 kernel,
 * computes its grid from the validated dimensions, and enqueues it on the
 * supplied stream. Return means enqueued, not completed. The corresponding
 * .cu file defines that operation's input-to-output computation.
 * Overlay: docs/architecture/transformer-execution.md,
 * "One Qwen3 transformer block".
 * Cost: grids cap at 65535 blocks with grid-stride work. Decode projection
 * uses 128 threads and registers; multi-row projection uses 16x16 threads
 * and up to 528 shared floats. RMSNorm stages (width+32) shared floats for
 * 128<width<=rms_staged_width; other widths use warp registers. Attention
 * retains up to attention_retained_keys scores with
 * width<=attention_retained_width; its shared bound is declared below for
 * preparation and launch to share. Larger attention shapes use 3*width+259
 * shared floats. Embedding shares one uint32; other elementwise kernels need no
 * shared storage. Attention borrows one initialized FP32 device scale scalar.
 * No other global operation workspace is required. Optimization: keep values on
 * device and amortize shape/capability validation. Corpus consultation, header
 * design and implementation, 2026-10-18: cpp-guidelines I.1 (#ri-explicit), I.7
 * (#ri-post): explicit type/dimensions, borrowing/completion preconditions and
 * numerical-operation postconditions. cpp-perf-guidelines GPU.1: no
 * intermediate host round trips. Source:
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 * cpp-perf-guidelines GPU.2: contiguous features map to adjacent lanes. Source:
 * guidelines/gpu/GPU.2-shape-data-for-coalesced-lane-access.md.
 * RMS staging consultation, design and implementation, 2026-10-20:
 * cpp-guidelines I.7 and cpp-perf-guidelines GPU.5: share the width bound
 * between preparation and dispatch so the staged input is always admitted.
 * Optimization consultation, header design and implementation, 2026-10-19:
 * cpp-guidelines I.7 (#ri-post): expose one bound for admission and launch.
 * cpp-perf-guidelines GPU.5: cap retained scores to bound shared-memory cost.
 * Source:
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md.
 * Device-loop extension (design/implementation, 2026-10-20): a prepared
 * prefill/decode body borrows an immutable-for-this-iteration device position.
 * The graph advances it only after every reader completes. RoPE/attention use
 * that absolute position; positioned outputs address cache rows S..S+B-1.
 * Arithmetic and reduction order are unchanged. Graph storage is admitted for
 * the maximum position; the device loop enforces S+B <= capacity. No per-token
 * host argument update, copy of position, or host launch is required.
 * Core I.7/CP.3: explicit bounds and ordered borrowing. Performance
 * GPU.1/GPU.6: resident control and stable graph arguments
 * (guidelines/gpu/GPU.1-keep-data-on-device.md;
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md). Both corpora queried and rules
 * read during header design and implementation.
 */
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "tensors/representations/representation.h"

namespace dolittle::kernels::cuda {
inline constexpr std::size_t rms_staged_partials = 32;
inline constexpr std::size_t rms_warp_width = 128;
inline constexpr std::size_t rms_staged_width = 8192;
inline constexpr std::size_t attention_retained_keys = 8192;
inline constexpr std::size_t attention_retained_width = 1024;
inline constexpr std::size_t attention_retained_shared_bytes =
    (attention_retained_keys + attention_retained_width + 256) * sizeof(float);

void launch_projection(tensors::ElementType type, const void* input,
                       const void* weight, void* output, std::size_t rows,
                       std::size_t input_width, std::size_t output_width,
                       cudaStream_t stream,
                       const std::size_t* output_position = nullptr);
void launch_rms_norm(tensors::ElementType type, const void* input,
                     const void* weight, void* output, std::size_t rows,
                     std::size_t width, float epsilon, cudaStream_t stream);
void launch_rotary(tensors::ElementType type, const void* input, void* output,
                   std::size_t positions, std::size_t heads, std::size_t width,
                   std::size_t first_position, double theta,
                   cudaStream_t stream,
                   const std::size_t* device_position = nullptr,
                   bool positioned_output = false);
void launch_attention_scale(float* output, std::size_t width,
                            cudaStream_t stream);
void launch_attention(tensors::ElementType type, const void* query,
                      const void* key, const void* value, void* output,
                      std::size_t positions, std::size_t query_heads,
                      std::size_t kv_heads, std::size_t width,
                      cudaStream_t stream, std::size_t first_position,
                      const std::size_t* device_position, std::size_t capacity,
                      const float* scale);
void launch_embedding(tensors::ElementType type, const std::uint32_t* tokens,
                      const void* weight, void* output, std::size_t rows,
                      std::size_t width, cudaStream_t stream);
void launch_add(tensors::ElementType type, const void* left, const void* right,
                void* output, std::size_t count, cudaStream_t stream);
void launch_silu_multiply(tensors::ElementType type, const void* gate,
                          const void* up, void* output, std::size_t count,
                          cudaStream_t stream);
}  // namespace dolittle::kernels::cuda
