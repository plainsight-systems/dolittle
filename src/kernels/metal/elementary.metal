/*
 * Axis X3/X5: FP32 exponential used by Metal softmax and SiLU.
 * Contract: exp_nonpositive accepts x<=0 (or NaN), returns exp(x) in FP32;
 * no tensor precision promotion, allocation, or CPU computation. Accuracy
 * depends on Metal's exp2 primitive; independent device vectors check the
 * supported device, including subnormal outputs. The model oracle remains
 * a separate acceptance check.
 *
 * Algorithm (transformer-execution.md, One Qwen3 transformer block): write
 * x*log2(e)=k+r. Split log2(e) into high/low FP32 parts; two explicit FMAs
 * retain the range-reduction residual before evaluating exp2(r). Clamp k to
 * [-126,0] so 2^k is normal, then scale the result by that exact power of two.
 * For subnormal output, shift/round its significand as integer bits rather
 * than depend on native subnormal multiplication. Zero maps to one, negative
 * infinity to zero, and NaNs propagate. Every tensor operation stays FP32.
 *
 * Optimization: reducing the native exp2 argument avoids the rounding error
 * of a single x*log2(e) conversion. Constant arithmetic, two FMAs, one native
 * transcendental, no workspace. NVIDIA libdevice's expf uses the same broad
 * split-range-reduction mechanism; its hardware approximation is not assumed
 * bit-identical to Apple's. Reference: CUDA libdevice User's Guide, __nv_expf;
 * Metal Shading Language specification, section 8.1 (math accuracy).
 *
 * Implementation consultation: Core I.7 (#ri-post), numerical postconditions;
 * performance GDSA.2, preserve explicit arithmetic/rounding boundaries
 * (guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md).
 */
#ifndef DOLITTLE_METAL_ELEMENTARY
#define DOLITTLE_METAL_ELEMENTARY
#include <metal_stdlib>
using namespace metal;
inline float exp_nonpositive(float x) {
  if (isnan(x)) return x;
  if (x < -1.0397208405e2f) return 0;
  const float log2_high = 1.4426950216293334961f;
  const float log2_low = 1.9259629911266174689e-8f;
  const float k = clamp(floor(x * log2_high), -126.0f, 0.0f);
  const float fraction = fma(x, log2_low, fma(x, log2_high, -k));
  const float y = precise::exp2(fraction);
  if (k > -126.0f || y >= 1.0f)
    return y * as_type<float>(uint(127 + int(k)) << 23);
  const uint bits = as_type<uint>(y);
  const uint exponent = (bits >> 23) & 255;
  const uint significand = (bits & 0x7fffff) | 0x800000;
  const uint shift = 127 - exponent;
  if (shift > 24) return 0;
  const uint retained = significand >> shift;
  const uint remainder = significand & ((1u << shift) - 1);
  const uint halfway = 1u << (shift - 1);
  return as_type<float>(
      retained +
      uint(remainder > halfway || (remainder == halfway && (retained & 1))));
}
#endif
