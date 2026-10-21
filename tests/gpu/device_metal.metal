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
