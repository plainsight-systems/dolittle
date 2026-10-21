/*
 * Implementation: GPU binary64 primitives and pinned OpenLibm range reduction.
 *
 * Axis X3/X5: software binary64 arithmetic needed by Metal sampling and RoPE
 * frequency initialization. This is a scalar arithmetic implementation, not
 * a new model tensor representation. Tensor storage remains BF16/FP32.
 * Contract: represent IEEE binary64 as integer bits; basic +,-,*,/, sqrt and
 * conversions use round-to-nearest/ties-to-even with signed zero, subnormal,
 * infinity and NaN handling. Operands/results never leave GPU execution.
 * No FP32 approximation is substituted for a binary64 operation.
 *
 * Algorithm: decode sign/exponent/significand; normalize subnormals; align
 * significands with sticky bits for add/subtract; use full-width products for
 * multiplication, quotient/remainder for division, and integer square-root
 * remainder for sqrt (including the pow(x,0.5) branch). Normalize and round
 * once using guard/round/sticky bits; encode the binary64 result. Compare by
 * sign/exponent/significand with explicit unordered NaNs. Conversion to FP32
 * performs its own final nearest/even rounding. Integer 53-bit RNG fractions
 * are exactly representable. No mutable global rounding mode or exception flag.
 *
 * Exp: reduce x=k*ln(2)+r using split constants, evaluate a bounded rational
 * approximation over the reduced interval using binary64 operations, then
 * scale by 2^k with explicit subnormal/overflow handling. Pow for RoPE accepts
 * positive finite theta and finite exponent: compute split log2(theta),
 * multiply by exponent with a compensated remainder, then reconstruct exp2.
 * Use the pinned OpenLibm algorithms below and retain their notices. Target <1
 * ulp exp error and validate the final FP32 frequencies against frozen
 * fixtures; do not claim correctly rounded transcendental functions without
 * evidence. RoPE computes 1/pow(theta,exponent) in the original order, not
 * pow(theta,-x).
 *
 * References inspected: Berkeley SoftFloat 3e's IEEE operation contracts;
 * OpenLibm e_exp.c/e_pow.c range-reduction algorithms; Metal-Float64 and
 * metal-softfloat demonstrate integer-backed Metal arithmetic. These are
 * references, not added runtime dependencies. Preserve notices if code or
 * coefficients are adapted. No inference framework is introduced.
 * https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat.html
 * https://github.com/JuliaMath/openlibm/blob/5fe399749f9276eaa0b8403e507470da05cbbb3f/src/e_exp.c
 * https://github.com/JuliaMath/openlibm/blob/5fe399749f9276eaa0b8403e507470da05cbbb3f/src/e_pow.c
 * https://github.com/philipturner/metal-float64
 * https://github.com/guyfischman/metal-softfloat
 *
 * Cost: bounded integer work per scalar operation, no memory allocation.
 * Optimization: keep this layer out of matrix/RMS/attention arithmetic; run
 * frequency setup once and parallelize independent sampler exponentials.
 * Validation: independent binary64 boundary/random vectors, signed zero,
 * subnormals, halfway rounding, overflow/underflow and nonfinite values on
 * GPU; exp/pow compare with high-precision offline vectors. Existing numerical
 * fixtures and limits stay fixed. A shared C++ arithmetic model, if used for
 * unit coverage, is a unit under test and cannot serve as its own oracle.
 *
 * Header-design consultation, 2026-10-21: both local corpora searched and
 * cited rules read. Core I.7 (#ri-post) fixes numerical/completion results;
 * SL.con.3 (#rsl-bounds) requires checked extents before native indexing;
 * R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.1: software arithmetic remains on the GPU.
 * guidelines/gpu/GPU.1-keep-data-on-device.md
 * GDSA.2: precision and rounding are independent of scheduling.
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md
 */
#ifndef DOLITTLE_METAL_NUMERIC
#define DOLITTLE_METAL_NUMERIC
#include <metal_stdlib>
using namespace metal;
namespace binary64 {
constant ulong sign_mask = 0x8000000000000000ul;
constant ulong fraction_mask = 0x000ffffffffffffful;
constant ulong infinity = 0x7ff0000000000000ul;
constant ulong quiet_nan = 0x7ff8000000000000ul;
inline ulong jam(ulong x, uint n) {
  return n == 0   ? x
         : n < 64 ? (x >> n) | ulong((x << (64 - n)) != 0)
                  : ulong(x != 0);
}
struct Parts {
  bool sign;
  int exponent;
  ulong significand;
};
inline Parts unpack(ulong x) {
  const uint e = uint((x >> 52) & 2047);
  ulong m = x & fraction_mask;
  int exponent = int(e) - 1023;
  if (e)
    m |= 1ul << 52;
  else {
    exponent = -1022;
    if (m) {
      const uint n = clz(m) - 11;
      m <<= n;
      exponent -= int(n);
    }
  }
  return {bool(x >> 63), exponent, m};
}
// Extended significand has its leading bit at 55 and three rounding bits.
inline ulong pack(bool sign, int e, ulong s) {
  const ulong sign_bits = sign ? sign_mask : 0;
  if (!s) return sign_bits;
  const int shift = 8 - int(clz(s));
  if (shift > 0) {
    s = jam(s, uint(shift));
    e += shift;
  } else if (shift < 0) {
    s <<= uint(-shift);
    e += shift;
  }
  if (e < -1022) {
    s = jam(s, uint(-1022 - e));
    e = -1022;
  }
  const ulong low = s & 7;
  ulong m = (s >> 3) + ulong(low > 4 || (low == 4 && ((s >> 3) & 1)));
  if (m >= (1ul << 53)) {
    m >>= 1;
    ++e;
  }
  if (e > 1023) return sign_bits | infinity;
  const ulong exponent = m >= (1ul << 52) ? ulong(e + 1023) : 0;
  return sign_bits | (exponent << 52) | (m & fraction_mask);
}
inline bool nan(ulong x) { return (x & ~sign_mask) > infinity; }
inline ulong from_uint(ulong x) {
  if (!x) return 0;
  const int e = 63 - int(clz(x));
  return pack(false, e, e <= 55 ? x << uint(55 - e) : jam(x, uint(e - 55)));
}
inline ulong from_float(float x) {
  const uint bits = as_type<uint>(x);
  const ulong sign = ulong(bits >> 31) << 63;
  const uint e = (bits >> 23) & 255;
  const uint f = bits & 0x7fffff;
  if (e == 255) return sign | (f ? quiet_nan : infinity);
  if (!e) {
    if (!f) return sign;
    const uint shift = clz(f) - 8;
    return sign | (ulong(897 - shift) << 52) |
           ((ulong(f) << (29 + shift)) & fraction_mask);
  }
  return sign | (ulong(e + 896) << 52) | (ulong(f) << 29);
}
struct F64 {
  ulong bits;
  F64() thread : bits(0) {}
  F64(int n)
  thread : bits(from_uint(n < 0 ? ulong(-long(n)) : ulong(n)) |
                (n < 0 ? sign_mask : 0)) {}
  explicit F64(float n) thread : bits(from_float(n)) {}
  static F64 raw(ulong n) {
    F64 r;
    r.bits = n;
    return r;
  }
  explicit operator int() const thread {
    auto p = unpack(bits);
    if (p.exponent < 0) return 0;
    const ulong n = p.exponent < 52 ? p.significand >> uint(52 - p.exponent)
                                    : p.significand << uint(p.exponent - 52);
    return p.sign ? -int(n) : int(n);
  }
};
inline F64 operator-(F64 a) { return F64::raw(a.bits ^ sign_mask); }
inline F64 operator+(F64 a, F64 b) {
  ulong ax = a.bits & ~sign_mask, bx = b.bits & ~sign_mask;
  if (ax >= infinity || bx >= infinity) {
    if (nan(a.bits) || nan(b.bits) ||
        (ax == infinity && bx == infinity && ((a.bits ^ b.bits) & sign_mask)))
      return F64::raw(quiet_nan);
    return ax == infinity ? a : b;
  }
  if (ax < bx) {
    auto swap = a;
    a = b;
    b = swap;
  }
  const auto x = unpack(a.bits), y = unpack(b.bits);
  ulong s = x.significand << 3;
  const ulong t =
      jam(y.significand << 3, uint(max(0, x.exponent - y.exponent)));
  if (x.sign == y.sign) return F64::raw(pack(x.sign, x.exponent, s + t));
  s -= t;
  return F64::raw(pack(s ? x.sign : false, x.exponent, s));
}
inline F64 operator-(F64 a, F64 b) { return a + (-b); }
inline F64 operator*(F64 a, F64 b) {
  const ulong ax = a.bits & ~sign_mask, bx = b.bits & ~sign_mask;
  const bool sign = bool((a.bits ^ b.bits) >> 63);
  if (nan(a.bits) || nan(b.bits) || (ax == infinity && !bx) ||
      (bx == infinity && !ax))
    return F64::raw(quiet_nan);
  if (ax == infinity || bx == infinity)
    return F64::raw((sign ? sign_mask : 0) | infinity);
  if (!ax || !bx) return F64::raw(sign ? sign_mask : 0);
  const auto x = unpack(a.bits), y = unpack(b.bits);
  const ulong lo = x.significand * y.significand;
  const ulong hi = mulhi(x.significand, y.significand);
  const bool carry = bool(hi & (1ul << 41));
  const uint shift = carry ? 50 : 49;
  const ulong s =
      (hi << (64 - shift)) | (lo >> shift) | ulong((lo << (64 - shift)) != 0);
  return F64::raw(pack(sign, x.exponent + y.exponent + int(carry), s));
}
inline F64 operator/(F64 a, F64 b) {
  const ulong ax = a.bits & ~sign_mask, bx = b.bits & ~sign_mask;
  const bool sign = bool((a.bits ^ b.bits) >> 63);
  if (nan(a.bits) || nan(b.bits) || (!ax && !bx) ||
      (ax == infinity && bx == infinity))
    return F64::raw(quiet_nan);
  if (ax == infinity || !bx) return F64::raw((sign ? sign_mask : 0) | infinity);
  if (!ax || bx == infinity) return F64::raw(sign ? sign_mask : 0);
  const auto x = unpack(a.bits), y = unpack(b.bits);
  ulong r = x.significand, q = 0;
  int e = x.exponent - y.exponent;
  if (r < y.significand) {
    r <<= 1;
    --e;
  }
  for (uint i = 0; i < 56; ++i) {
    q <<= 1;
    if (r >= y.significand) {
      r -= y.significand;
      q |= 1;
    }
    r <<= 1;
  }
  return F64::raw(pack(sign, e, q | ulong(r != 0)));
}
inline bool operator==(F64 a, F64 b) {
  return !nan(a.bits) && !nan(b.bits) &&
         (a.bits == b.bits || (((a.bits | b.bits) & ~sign_mask) == 0));
}
inline bool operator<(F64 a, F64 b) {
  if (nan(a.bits) || nan(b.bits) || a == b) return false;
  if ((a.bits ^ b.bits) & sign_mask) return bool(a.bits >> 63);
  return a.bits & sign_mask ? a.bits > b.bits : a.bits < b.bits;
}
inline bool operator>(F64 a, F64 b) { return b < a; }
inline bool operator<=(F64 a, F64 b) { return a == b || a < b; }
inline bool operator>=(F64 a, F64 b) { return a == b || a > b; }
inline thread F64& operator+=(thread F64& a, F64 b) {
  a = a + b;
  return a;
}
inline thread F64& operator-=(thread F64& a, F64 b) {
  a = a - b;
  return a;
}
inline thread F64& operator*=(thread F64& a, F64 b) {
  a = a * b;
  return a;
}
inline F64 fabs(F64 a) { return F64::raw(a.bits & ~sign_mask); }
inline F64 scalbn(F64 a, int n) {
  if ((a.bits & ~sign_mask) >= infinity || !(a.bits & ~sign_mask)) return a;
  auto p = unpack(a.bits);
  return F64::raw(pack(p.sign, p.exponent + n, p.significand << 3));
}
inline F64 sqrt(F64 a) {
  const ulong magnitude = a.bits & ~sign_mask;
  if (!magnitude || a.bits == infinity) return a;
  if ((a.bits >> 63) || nan(a.bits)) return F64::raw(quiet_nan);
  auto p = unpack(a.bits);
  const uint shift = (p.exponent & 1) ? 59 : 58;
  const ulong lo = p.significand << shift, hi = p.significand >> (64 - shift);
  ulong root = 0, remainder = 0;
  for (int i = 55; i >= 0; --i) {
    const uint bit = uint(2 * i);
    const ulong pair = (bit >= 64 ? hi >> (bit - 64) : lo >> bit) & 3;
    remainder = (remainder << 2) | pair;
    const ulong trial = (root << 2) | 1;
    root <<= 1;
    if (remainder >= trial) {
      remainder -= trial;
      ++root;
    }
  }
  return F64::raw(pack(false, p.exponent >> 1, root | ulong(remainder != 0)));
}
inline float to_float(F64 a) {
  const uint sign = uint(a.bits >> 32) & 0x80000000u;
  if (nan(a.bits)) return as_type<float>(sign | 0x7fc00000u);
  if ((a.bits & ~sign_mask) == infinity)
    return as_type<float>(sign | 0x7f800000u);
  auto p = unpack(a.bits);
  ulong s = jam(p.significand, 26);
  if (p.exponent < -126) {
    s = jam(s, uint(-126 - p.exponent));
    p.exponent = -126;
  }
  const ulong r = s & 7;
  ulong m = (s >> 3) + ulong(r > 4 || (r == 4 && ((s >> 3) & 1)));
  if (m >= (1ul << 24)) {
    m >>= 1;
    ++p.exponent;
  }
  if (p.exponent > 127) return as_type<float>(sign | 0x7f800000u);
  return as_type<float>(sign |
                        (m >= (1ul << 23) ? uint(p.exponent + 127) << 23 : 0) |
                        (uint(m) & 0x7fffffu));
}

/* Adapted from OpenLibm e_exp.c and e_pow.c, revision
 * 5fe399749f9276eaa0b8403e507470da05cbbb3f.
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 * Permission to use, copy, modify, and distribute this software is freely
 * granted, provided that this notice is preserved.
 * Arithmetic is replaced by the integer-backed F64 operations above.
 */
#define GET_HIGH_WORD(h, x) h = decltype(h)(uint((x).bits >> 32))
#define GET_LOW_WORD(l, x) l = decltype(l)(uint((x).bits))
#define SET_HIGH_WORD(x, h) \
  (x).bits = ((x).bits & 0xfffffffful) | (ulong(uint(h)) << 32)
#define SET_LOW_WORD(x, l) \
  (x).bits = ((x).bits & 0xffffffff00000000ul) | uint(l)
#define INSERT_WORDS(x, h, l) (x).bits = (ulong(uint(h)) << 32) | uint(l)
#define EXTRACT_WORDS(h, l, x) \
  {                            \
    GET_HIGH_WORD(h, x);       \
    GET_LOW_WORD(l, x);        \
  }
#define STRICT_ASSIGN(type, x, v) x = v
namespace exp_impl {
inline F64 evaluate(F64 x) {
  const F64 one = F64::raw(0x3ff0000000000000ul),
            halF[2] =
                {
                    F64::raw(0x3fe0000000000000ul),
                    -F64::raw(0x3fe0000000000000ul),
                },
            huge = F64::raw(0x7e37e43c8800759cul),
            o_threshold = F64::raw(0x40862e42fefa39eful),
            u_threshold = -F64::raw(0x40874910d52d3051ul),
            ln2HI[2] =
                {
                    F64::raw(0x3fe62e42fee00000ul),
                    -F64::raw(0x3fe62e42fee00000ul),
                },
            ln2LO[2] =
                {
                    F64::raw(0x3dea39ef35793c76ul),
                    -F64::raw(0x3dea39ef35793c76ul),
                },
            invln2 = F64::raw(0x3ff71547652b82feul),
            P1 = F64::raw(0x3fc555555555553eul),
            P2 = -F64::raw(0x3f66c16c16bebd93ul),
            P3 = F64::raw(0x3f11566aaf25de2cul),
            P4 = -F64::raw(0x3ebbbd41c5d26bf1ul),
            P5 = F64::raw(0x3e66376972bea4d0ul);

  const F64 twom1000 = F64::raw(0x0170000000000000ul);

  F64 y, hi = F64::raw(0x0000000000000000ul),
         lo = F64::raw(0x0000000000000000ul), c, t, twopk;
  int k = 0, xsb;
  uint hx;

  GET_HIGH_WORD(hx, x);
  xsb = (hx >> 31) & 1;
  hx &= 0x7fffffff;

  if (hx >= 0x40862E42) {
    if (hx >= 0x7ff00000) {
      uint lx;
      GET_LOW_WORD(lx, x);
      if (((hx & 0xfffff) | lx) != 0)
        return x + x;
      else
        return (xsb == 0) ? x : F64::raw(0x0000000000000000ul);
    }
    if (x > o_threshold) return huge * huge;
    if (x < u_threshold) return twom1000 * twom1000;
  }

  if (x == F64::raw(0x3ff0000000000000ul))
    return F64::raw(0x4005bf0a8b145769ul);

  if (hx > 0x3fd62e42) {
    if (hx < 0x3FF0A2B2) {
      hi = x - ln2HI[xsb];
      lo = ln2LO[xsb];
      k = 1 - xsb - xsb;
    } else {
      k = (int)(invln2 * x + halF[xsb]);
      t = k;
      hi = x - t * ln2HI[0];
      lo = t * ln2LO[0];
    }
    STRICT_ASSIGN(F64, x, hi - lo);
  } else if (hx < 0x3e300000) {
    if (huge + x > one) return one + x;
  } else
    k = 0;

  t = x * x;
  if (k >= -1021)
    INSERT_WORDS(twopk, 0x3ff00000 + (uint(k) << 20), 0);
  else
    INSERT_WORDS(twopk, 0x3ff00000 + (uint(k + 1000) << 20), 0);
  c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
  if (k == 0)
    return one - ((x * c) / (c - F64::raw(0x4000000000000000ul)) - x);
  else
    y = one - ((lo - (x * c) / (F64::raw(0x4000000000000000ul) - c)) - hi);
  if (k >= -1021) {
    if (k == 1024)
      return y * F64::raw(0x4000000000000000ul) *
             F64::raw(0x7fe0000000000000ul);
    return y * twopk;
  } else {
    return y * twopk * twom1000;
  }
}

}  // namespace exp_impl
namespace pow_impl {
inline F64 evaluate(F64 x, F64 y) {
  const F64 bp[] =
      {
          F64::raw(0x3ff0000000000000ul),
          F64::raw(0x3ff8000000000000ul),
      },
            dp_h[] =
                {
                    F64::raw(0x0000000000000000ul),
                    F64::raw(0x3fe2b80340000000ul),
                },
            dp_l[] =
                {
                    F64::raw(0x0000000000000000ul),
                    F64::raw(0x3e4cfdeb43cfd006ul),
                },
            zero = F64::raw(0x0000000000000000ul),
            one = F64::raw(0x3ff0000000000000ul),
            two = F64::raw(0x4000000000000000ul),
            two53 = F64::raw(0x4340000000000000ul),
            huge = F64::raw(0x7e37e43c8800759cul),
            tiny = F64::raw(0x01a56e1fc2f8f359ul),

            L1 = F64::raw(0x3fe3333333333303ul),
            L2 = F64::raw(0x3fdb6db6db6fabfful),
            L3 = F64::raw(0x3fd55555518f264dul),
            L4 = F64::raw(0x3fd17460a91d4101ul),
            L5 = F64::raw(0x3fcd864a93c9db65ul),
            L6 = F64::raw(0x3fca7e284a454eeful),
            P1 = F64::raw(0x3fc555555555553eul),
            P2 = -F64::raw(0x3f66c16c16bebd93ul),
            P3 = F64::raw(0x3f11566aaf25de2cul),
            P4 = -F64::raw(0x3ebbbd41c5d26bf1ul),
            P5 = F64::raw(0x3e66376972bea4d0ul),
            lg2 = F64::raw(0x3fe62e42fefa39eful),
            lg2_h = F64::raw(0x3fe62e4300000000ul),
            lg2_l = -F64::raw(0x3e205c610ca86c39ul),
            ovt = F64::raw(0x3c971547652b82feul),
            cp = F64::raw(0x3feec709dc3a03fdul),
            cp_h = F64::raw(0x3feec709e0000000ul),
            cp_l = -F64::raw(0x3e3e2fe0145b01f5ul),
            ivln2 = F64::raw(0x3ff71547652b82feul),
            ivln2_h = F64::raw(0x3ff7154760000000ul),
            ivln2_l = F64::raw(0x3e54ae0bf85ddf44ul);

  F64 z, ax, z_h, z_l, p_h, p_l;
  F64 y1, t1, t2, r, s, t, u, v, w;
  int i, j, k, yisint, n;
  int hx, hy, ix, iy;
  uint lx, ly;

  EXTRACT_WORDS(hx, lx, x);
  EXTRACT_WORDS(hy, ly, y);
  ix = hx & 0x7fffffff;
  iy = hy & 0x7fffffff;

  if ((iy | ly) == 0) return one;

  if (hx == 0x3ff00000 && lx == 0) return one;

  if (ix > 0x7ff00000 || ((ix == 0x7ff00000) && (lx != 0)) || iy > 0x7ff00000 ||
      ((iy == 0x7ff00000) && (ly != 0)))
    return (x + F64::raw(0x0000000000000000ul)) +
           (y + F64::raw(0x0000000000000000ul));

  yisint = 0;
  if (hx < 0) {
    if (iy >= 0x43400000)
      yisint = 2;
    else if (iy >= 0x3ff00000) {
      k = (iy >> 20) - 0x3ff;
      if (k > 20) {
        j = ly >> (52 - k);
        if (uint(j << (52 - k)) == ly) yisint = 2 - (j & 1);
      } else if (ly == 0) {
        j = iy >> (20 - k);
        if ((j << (20 - k)) == iy) yisint = 2 - (j & 1);
      }
    }
  }

  if (ly == 0) {
    if (iy == 0x7ff00000) {
      if (((ix - 0x3ff00000) | lx) == 0)
        return one;
      else if (ix >= 0x3ff00000)
        return (hy >= 0) ? y : zero;
      else
        return (hy < 0) ? -y : zero;
    }
    if (iy == 0x3ff00000) {
      if (hy < 0)
        return one / x;
      else
        return x;
    }
    if (hy == 0x40000000) return x * x;
    if (hy == 0x40080000) return x * x * x;
    if (hy == 0x40100000) {
      u = x * x;
      return u * u;
    }
    if (hy == 0x3fe00000) {
      if (hx >= 0) return sqrt(x);
    }
  }

  ax = fabs(x);

  if (lx == 0) {
    if (ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000) {
      z = ax;
      if (hy < 0) z = one / z;
      if (hx < 0) {
        if (((ix - 0x3ff00000) | yisint) == 0) {
          z = (z - z) / (z - z);
        } else if (yisint == 1)
          z = -z;
      }
      return z;
    }
  }

  n = ((uint)hx >> 31) - 1;

  if ((n | yisint) == 0) return (x - x) / (x - x);

  s = one;
  if ((n | (yisint - 1)) == 0) s = -one;

  if (iy > 0x41e00000) {
    if (iy > 0x43f00000) {
      if (ix <= 0x3fefffff) return (hy < 0) ? huge * huge : tiny * tiny;
      if (ix >= 0x3ff00000) return (hy > 0) ? huge * huge : tiny * tiny;
    }

    if (ix < 0x3fefffff) return (hy < 0) ? s * huge * huge : s * tiny * tiny;
    if (ix > 0x3ff00000) return (hy > 0) ? s * huge * huge : s * tiny * tiny;

    t = ax - one;
    w = (t * t) * (F64::raw(0x3fe0000000000000ul) -
                   t * (F64::raw(0x3fd5555555555555ul) -
                        t * F64::raw(0x3fd0000000000000ul)));
    u = ivln2_h * t;
    v = t * ivln2_l - w * ivln2;
    t1 = u + v;
    SET_LOW_WORD(t1, 0);
    t2 = v - (t1 - u);
  } else {
    F64 ss, s2, s_h, s_l, t_h, t_l;
    n = 0;

    if (ix < 0x00100000) {
      ax *= two53;
      n -= 53;
      GET_HIGH_WORD(ix, ax);
    }
    n += ((ix) >> 20) - 0x3ff;
    j = ix & 0x000fffff;

    ix = j | 0x3ff00000;
    if (j <= 0x3988E)
      k = 0;
    else if (j < 0xBB67A)
      k = 1;
    else {
      k = 0;
      n += 1;
      ix -= 0x00100000;
    }
    SET_HIGH_WORD(ax, ix);

    u = ax - bp[k];
    v = one / (ax + bp[k]);
    ss = u * v;
    s_h = ss;
    SET_LOW_WORD(s_h, 0);

    t_h = zero;
    SET_HIGH_WORD(t_h, ((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18));
    t_l = ax - (t_h - bp[k]);
    s_l = v * ((u - s_h * t_h) - s_h * t_l);

    s2 = ss * ss;
    r = s2 * s2 *
        (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
    r += s_l * (s_h + ss);
    s2 = s_h * s_h;
    t_h = F64::raw(0x4008000000000000ul) + s2 + r;
    SET_LOW_WORD(t_h, 0);
    t_l = r - ((t_h - F64::raw(0x4008000000000000ul)) - s2);

    u = s_h * t_h;
    v = s_l * t_h + t_l * ss;

    p_h = u + v;
    SET_LOW_WORD(p_h, 0);
    p_l = v - (p_h - u);
    z_h = cp_h * p_h;
    z_l = cp_l * p_h + p_l * cp + dp_l[k];

    t = (F64)n;
    t1 = (((z_h + z_l) + dp_h[k]) + t);
    SET_LOW_WORD(t1, 0);
    t2 = z_l - (((t1 - t) - dp_h[k]) - z_h);
  }

  y1 = y;
  SET_LOW_WORD(y1, 0);
  p_l = (y - y1) * t1 + y * t2;
  p_h = y1 * t1;
  z = p_l + p_h;
  EXTRACT_WORDS(j, i, z);
  if (j >= 0x40900000) {
    if (((j - 0x40900000) | i) != 0)
      return s * huge * huge;
    else {
      if (p_l + ovt > z - p_h) return s * huge * huge;
    }
  } else if ((j & 0x7fffffff) >= 0x4090cc00) {
    if (((j - 0xc090cc00) | i) != 0)
      return s * tiny * tiny;
    else {
      if (p_l <= z - p_h) return s * tiny * tiny;
    }
  }

  i = j & 0x7fffffff;
  k = (i >> 20) - 0x3ff;
  n = 0;
  if (i > 0x3fe00000) {
    n = j + (0x00100000 >> (k + 1));
    k = ((n & 0x7fffffff) >> 20) - 0x3ff;
    t = zero;
    SET_HIGH_WORD(t, n & ~(0x000fffff >> k));
    n = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
    if (j < 0) n = -n;
    p_h -= t;
  }
  t = p_l + p_h;
  SET_LOW_WORD(t, 0);
  u = t * lg2_h;
  v = (p_l - (t - p_h)) * lg2 + t * lg2_l;
  z = u + v;
  w = v - (z - u);
  t = z * z;
  t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
  r = (z * t1) / (t1 - two) - (w + z * w);
  z = one - (r - z);
  GET_HIGH_WORD(j, z);
  j += (uint(n) << 20);
  if ((j >> 20) <= 0)
    z = scalbn(z, n);
  else
    SET_HIGH_WORD(z, j);
  return s * z;
}

}  // namespace pow_impl
inline F64 exp(F64 x) { return exp_impl::evaluate(x); }
inline F64 pow(F64 x, F64 y) { return pow_impl::evaluate(x, y); }
#undef GET_HIGH_WORD
#undef GET_LOW_WORD
#undef SET_HIGH_WORD
#undef SET_LOW_WORD
#undef INSERT_WORDS
#undef EXTRACT_WORDS
#undef STRICT_ASSIGN
}  // namespace binary64
#endif
