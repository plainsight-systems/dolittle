/*
 * Axis X5: device normalization for a declared NLA injection law.
 * Contract: borrow a complete FP32/BF16 source row and its Device/Queue through
 * completion; own separate normalized output plus norm/status. Source is never
 * changed. submit may be captured in a graph and allocates or reads back
 * nothing. Status zero permits embedding replacement; nonzero forbids
 * subsequent inference. Algorithm: execution-boundary.md H2. Widen exactly;
 * accumulate h[i]^2 in eight FP32 lanes i%8, ascending i, with separate
 * multiply/add. Combine lanes 0..7 left-to-right; sqrt in FP32; require finite
 * nonzero norm. For each element divide by norm, multiply by scale separately
 * and cast once to output storage. Reject nonfinite source/intermediate/output
 * on-device. Norm is provenance only. Design/cost: O(D) GPU work, D*b output
 * and constant reduction/status storage. Optimization: no host FP32 temporary,
 * raw-vector readback or injection upload. Header design 2026-10-20: Core
 * I.1/R.1 make status and asynchronous lifetime explicit; Performance
 * GPU.1/GPU.9 keep intermediate values in owned device storage outside
 * transient transformer scratch. Both corpora queried/read. Source
 * guidelines/gpu/GPU.1-keep-data-on-device.md.
 */
#pragma once
#include <memory>

#include "gpu/device.h"
#include "tensors/representations/representation.h"
namespace dolittle::inference {
class DeviceInjection {
 public:
  DeviceInjection(gpu::Device&, gpu::Queue&, gpu::ConstDeviceRegion,
                  tensors::ElementType source_type,
                  tensors::ElementType output_type, std::size_t width,
                  float scale);
  ~DeviceInjection();
  DeviceInjection(const DeviceInjection&) = delete;
  DeviceInjection& operator=(const DeviceInjection&) = delete;
  DeviceInjection(DeviceInjection&&) = delete;
  DeviceInjection& operator=(DeviceInjection&&) = delete;
  [[nodiscard]] static std::size_t required_bytes(std::size_t,
                                                  tensors::ElementType);
  void submit();
  [[nodiscard]] gpu::ConstDeviceRegion output() const;
  [[nodiscard]] gpu::ConstDeviceRegion status() const;
  [[nodiscard]] gpu::ConstDeviceRegion norm() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dolittle::inference
