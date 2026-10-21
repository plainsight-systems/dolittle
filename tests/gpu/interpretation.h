/*
 * Steering verification takes a sibling frozen fixture. It returns owned
 * numerical metrics after AR/direction/replay checks, releasing each model
 * before switching roles. The device outlives all temporary owners.
 * Steering corpus consultation, header design and implementation 2026-10-20:
 * Core I.1 (#ri-explicit), typed contracts; performance MEM.9, bounded setup
 * before execution (guidelines/memory/MEM.9-embedded-allocate-at-init.md).
 * Both local corpora queried and full rules read.
 *
 * Verification axis D3: AV continuation of the existing tiny/8B experiment
 * scenarios. Input is a frozen target fixture; the sibling AV fixture supplies
 * independently frozen injection/logit/capture expectations. Returns owned
 * metrics. No sixth scenario is registered. The implementation owns temporary
 * checkpoints and releases each plan before switching model roles; caller's
 * device outlives it. Algorithm: docs/architecture/logical-overview.md,
 * sections 2 through 4:
 * 1. Read pinned fixtures and verify model artifacts. Normalize the independent
 *    target row; compare FP32 values and exact stored replacement bytes.
 * 2. Generate through Run, release target execution and retain the selected
 * row. Compare it with the unchanged target limits, then snapshot its owned
 * bytes.
 * 3. Feed identical reference injection bytes to AV full and cached execution.
 *    Compare selected activations and logits against both fixed error bounds
 * for every row. Require exact final cached/full logits on the same consumed
 * IDs.
 * 4. Interpret the retained engine row twice. Require expected greedy IDs and
 *    fresh sequence state with retained AV weights, then check pre-request
 *    cancellation, rejected calls and capture lifetime after model close.
 * Cost: bounded fixture passes; comparison work and reference copies stay
 * outside inference timing. Provider callbacks borrow immutable requests that
 * outlive Run. Optimization: compare completed host bytes without additional
 * diagnostic waits. Corpus implementation consultation, 2026-10-19:
 * cpp-guidelines R.1 (#rr-raii), I.7 (#ri-post) enforce ownership and numerical
 * postconditions. Performance GPU.1 separates fixture/readback work from
 * compute; source guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>

#include "gpu/device.h"
namespace dolittle::tests {
[[nodiscard]] nlohmann::json check_steering(
    gpu::Device&, const std::filesystem::path& target_fixture);
[[nodiscard]] nlohmann::json check_interpretation(
    gpu::Device&, const std::filesystem::path& target_fixture);
}  // namespace dolittle::tests
