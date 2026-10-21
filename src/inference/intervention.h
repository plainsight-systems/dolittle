/*
 * Axis R6/X1: one-position residual steering input.
 * Contract: direction leases exactly D FP32 unit-vector elements on the same
 * device as the sequence. Strength is finite and nonnegative; position belongs
 * to the fixed replay prefix. The model must advertise the selected boundary.
 * Source captures are immutable; only the branch working residual is writable.
 * The plan retains the lease through completion, including cancellation.
 * Algorithm (steering.md): after the selected block produces h[p,D], add
 * alpha*L2(h[p])*direction[i] to each widened element and narrow once to the
 * activation storage type. Other rows pass through unchanged. Alpha zero is
 * an exact bypass. Device status reports invalid values or repeated mutation.
 * Optimization: O(1) shared ownership; no vector copy during admission.
 * Header design: Core I.1 (#ri-explicit), R.1 (#rr-raii); performance GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md) and GPU.9
 * (guidelines/gpu/GPU.9-suballocate-and-alias-transient-gpu-memory.md).
 * Both corpora queried/read, 2026-10-20: explicit mutation and persistent
 * lease.
 */
#pragma once

#include "inference/observations.h"

namespace dolittle::inference {
struct ResidualIntervention {
  models::ObservationPoint site{};
  models::TokenPosition position{};
  DeviceCapture direction;
  std::size_t width{};
  float strength{};
};
}  // namespace dolittle::inference
