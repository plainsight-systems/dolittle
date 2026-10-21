/*
 * D3 deterministic device-only sequence fixture. Advance a token recurrence
 * and EOS word entirely on GPU; the production scheduler must consume the last
 * selection even at EOS/cancel. No transformer/oracle replacement. Core I.7
 * (#ri-post), performance GPU.1 (guidelines/gpu/GPU.1-keep-data-on-device.md),
 * read for implementation: delayed collection must not affect progress.
 */
#include <metal_stdlib>
using namespace metal;
kernel void fixture_select(device const uint* tokens [[buffer(0)]],
                           device uint* selection [[buffer(1)]],
                           device uint* terminal [[buffer(2)]],
                           constant uint& eos [[buffer(3)]]) {
  selection[0] = tokens[0] + 1;
  selection[1] = 0;
  terminal[0] = selection[0] == eos;
}

// Independent IEEE/libm vectors are supplied by the host test. This shader is
// the production arithmetic under test, not a second implementation of it.
#include "../../src/kernels/metal/elementary.metal"
#include "../../src/kernels/metal/numeric.metal"
kernel void fixture_binary64(device const ulong* input [[buffer(0)]],
                             device ulong* output [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
  using namespace binary64;
  const auto a = F64::raw(input[3 * i]), b = F64::raw(input[3 * i + 1]);
  ulong result = 0;
  switch (input[3 * i + 2]) {
    case 0:
      result = (a + b).bits;
      break;
    case 1:
      result = (a - b).bits;
      break;
    case 2:
      result = (a * b).bits;
      break;
    case 3:
      result = (a / b).bits;
      break;
    case 4:
      result = binary64::sqrt(a).bits;
      break;
    case 5:
      result = as_type<uint>(to_float(a));
      break;
    case 6:
      result = F64(as_type<float>(uint(input[3 * i]))).bits;
      break;
    case 7:
      result = binary64::exp(a).bits;
      break;
    case 8:
      result = binary64::pow(a, b).bits;
      break;
    case 9:
      result = as_type<uint>(exp_nonpositive(to_float(a)));
      break;
    case 10:
      result = as_type<uint>(precise::rsqrt(to_float(a)));
      break;
  }
  output[i] = result;
}
