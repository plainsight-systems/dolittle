/*
 * Metal implementation of the shared target-generation operations.
 *
 * Axis R1/X5: GPU categorical selection with binary64 weights/CDF, distinct
 * from BF16/FP32 transformer arithmetic. Implements sampling/device.h.
 * Input one vocabulary row; output token/status and terminal/RNG device words.
 * NaN/+inf/all-masked input reports failure. -inf has zero mass. Signed zeros
 * compare equal. Stable equal-logit order is ascending token ID.
 *
 * Algorithm: transformer-execution.md, Device selection.
 * 1. Validate logits and encode descending FP32 sort keys. Four stable byte
 *    radix passes histogram, prefix and scatter {key,id}. Integer counts and
 *    original index ranks determine destinations, never atomic arrival order.
 *    Metal SIMD ballot/shuffle or per-group digit masks replace CUDA match_any.
 * 2. Retain top-k (zero means all). For each finite retained logit, compute
 *    exp_binary64((binary64(logit)-binary64(maximum))/temperature). -inf -> 0.
 * 3. Scan each fixed 256-entry tile left-to-right in binary64, starting at
 *    zero. Scan tile totals in tile-index order to produce exclusive offsets.
 *    Each prefix is one binary64 addition of its offset and local prefix.
 * 4. Choose the first inclusive prefix/total >= top_p. Keep that token too.
 *    Divide prefixes by retained mass; choose first normalized CDF > u.
 *    Integer-min reductions choose the first crossing. No parallel floating
 *    scan tree may replace the prescribed serial-within-tile grouping.
 * 5. One GPU lane owns 312 uint64 MT19937-64 words and the index. Twist only
 *    when index reaches 312; temper the next word; u=(word>>11)*2^-53 exactly
 *    in binary64. Greedy selects highest logit/lowest ID without a draw.
 * 6. Latch selection status/EOS. Later inactive selections preserve the last
 *    result and RNG; DeviceSequence still consumes a valid pending final ID.
 *
 * Metal has no native double. numeric.metal implements binary64 storage and
 * operations on integer words; settings arrive as exact bit patterns. This
 * preserves the declared precision and 53-bit uniform mapping without CPU
 * arithmetic or a float32 CDF. Exp may differ from CUDA libdevice at the last
 * bit, as allowed by the existing transcendental contract; no promise of
 * identical sampled prose across backends. Test cutoff/draw boundaries and
 * frozen supplied draws independently from full-model greedy comparisons.
 *
 * Cost O(V) per radix/weight/scan pass; no tensor readback. Optimization:
 * parallel exp across vocabulary, block-local integer histogram aggregation,
 * deterministic scatter, and a greedy fast path. The binary64 scan is confined
 * to the sampler and must be profiled separately from transformer kernels.
 *
 * Header-design consultation, 2026-10-21: both local corpora searched and
 * cited rules read. Core I.7 (#ri-post) fixes numerical/completion results;
 * SL.con.3 (#rsl-bounds) requires checked extents before native indexing;
 * R.1 (#rr-raii) keeps prepared owners alive through completion.
 * GPU.1: GPU RNG and CDF remove CPU selection feedback.
 * guidelines/gpu/GPU.1-keep-data-on-device.md
 * GPU.5: block-local histograms and prefix reuse bound memory traffic.
 * guidelines/gpu/GPU.5-use-shared-memory-when-reuse-or-reordering-pays.md
 * GDSA.2: fixed scans and stable ties define repeatability.
 * guidelines/gpu-dsa/GDSA.2-declare-each-reductions-determinism.md
 */
#include "common.metal"
#include "numeric.metal"
using binary64::F64;
struct Candidate {
  uint key, id;
};
struct SampleResult {
  uint token;
  atomic_uint status;
};
struct ScanState {
  ulong total;
  atomic_uint cutoff, selected;
};
struct RandomState {
  ulong words[312];
  uint index, terminal;
  ulong draw;
  uint previous_token, previous_status;
};
struct Sampler {
  device Candidate* first;
  device Candidate* second;
  device ulong* probabilities;
  device uint* counts;
  device ScanState* scan;
  device SampleResult* result;
  device RandomState* random;
  device uchar* stops;
  uint n, keep, blocks;
};
inline Sampler sampler_state(constant OperationArgs& a) {
  Sampler s;
  s.n = uint(a.dimensions[0]);
  s.keep = uint(a.dimensions[1]);
  s.blocks = (s.n + 255) / 256;
  s.first = (device Candidate*)a.address[1];
  s.second = s.first + s.n;
  s.probabilities = (device ulong*)(s.second + s.n);
  s.counts = (device uint*)(s.probabilities + s.n);
  s.scan = (device ScanState*)(s.counts + ulong(s.blocks) * 256);
  s.result = (device SampleResult*)(s.scan + 1);
  s.random = (device RandomState*)(s.result + 1);
  s.stops = (device uchar*)(s.random + 1);
  return s;
}
inline uint status(device SampleResult* r) {
  return atomic_load_explicit(&r->status, memory_order_relaxed);
}
inline void fail(device SampleResult* r, uint value) {
  atomic_store_explicit(&r->status, value, memory_order_relaxed);
}
kernel void sampler_seed(constant OperationArgs& a [[buffer(0)]]) {
  auto s = sampler_state(a);
  auto state = s.random;
  state->words[0] = a.capacity;
  for (uint i = 1; i < 312; ++i) {
    ulong p = state->words[i - 1];
    state->words[i] = 6364136223846793005ul * (p ^ (p >> 62)) + i;
  }
  state->index = 312;
  state->terminal = 0;
  state->draw = 0;
  state->previous_token = 0;
  state->previous_status = 0;
}
kernel void sampler_clear_stops(constant OperationArgs& a [[buffer(0)]],
                                uint i [[thread_position_in_grid]]) {
  auto s = sampler_state(a);
  if (i < s.n) s.stops[i] = 0;
}
kernel void sampler_set_stop(constant OperationArgs& a [[buffer(0)]]) {
  sampler_state(a).stops[a.first] = 1;
}
kernel void sampler_begin(constant OperationArgs& a [[buffer(0)]]) {
  auto s = sampler_state(a);
  s.result->token = 0;
  fail(s.result, 0);
  if (!(a.flags & 1) || (a.flags & 2) || s.random->terminal) return;
  auto state = s.random;
  if (state->index == 312) {
    for (uint i = 0; i < 312; ++i) {
      ulong x = (state->words[i] & 0xffffffff80000000ul) |
                (state->words[(i + 1) % 312] & 0x7ffffffful);
      state->words[i] = state->words[(i + 156) % 312] ^ (x >> 1) ^
                        ((x & 1) ? 0xb5026f5aa96619e9ul : 0);
    }
    state->index = 0;
  }
  ulong x = state->words[state->index++];
  x ^= (x >> 29) & 0x5555555555555555ul;
  x ^= (x << 17) & 0x71d67fffeda60000ul;
  x ^= (x << 37) & 0xfff7eee000000000ul;
  x ^= x >> 43;
  state->draw =
      binary64::scalbn(F64::raw(binary64::from_uint(x >> 11)), -53).bits;
}
kernel void sampler_commit(constant OperationArgs& a [[buffer(0)]]) {
  auto s = sampler_state(a);
  auto r = s.random;
  if (r->terminal) {
    s.result->token = r->previous_token;
    fail(s.result, r->previous_status);
    return;
  }
  r->previous_token = s.result->token;
  r->previous_status = status(s.result);
  r->terminal = r->previous_status || s.stops[s.result->token];
}
kernel void sampler_keys(constant OperationArgs& a [[buffer(0)]],
                         uint i [[thread_position_in_grid]]) {
  auto s = sampler_state(a);
  if (i >= s.n) return;
  float v = load(a.address[0], i, a.type);
  if (isnan(v) || v == INFINITY) fail(s.result, 1);
  if (v == 0) v = 0.0f;
  uint b = as_type<uint>(v);
  s.first[i] = {~((b & 0x80000000u) ? ~b : b ^ 0x80000000u), i};
}
// Nine ballots include the invalid-tail digit 256. Ranking never uses atomic
// arrival order: equal digits are ordered by original lane, then SIMD group.
inline uint peers_for(uint digit) {
  uint peers = 0xffffffffu;
  for (uint bit = 0; bit < 9; ++bit) {
    const bool set = (digit & (1u << bit)) != 0;
    const uint vote = uint(ulong(simd_ballot(set)));
    peers &= set ? vote : ~vote;
  }
  return peers;
}
kernel void sampler_histogram(constant OperationArgs& a [[buffer(0)]],
                              threadgroup uint* bins [[threadgroup(0)]],
                              uint t [[thread_index_in_threadgroup]],
                              uint b [[threadgroup_position_in_grid]]) {
  auto s = sampler_state(a);
  const uint i = b * 256 + t, lane = t % 32, warp = t / 32;
  auto input = a.dimensions[2] % 16 == 0 ? s.first : s.second;
  for (uint w = 0; w < 8; ++w) bins[w * 256 + t] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint digit = i < s.n ? (input[i].key >> a.dimensions[2]) & 255 : 256;
  const uint peers = peers_for(digit);
  if (digit < 256 && lane == ctz(peers))
    bins[warp * 256 + digit] = popcount(peers);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint total = 0;
  for (uint w = 0; w < 8; ++w) total += bins[w * 256 + t];
  s.counts[ulong(b) * 256 + t] = total;
}
kernel void sampler_radix_prefix(constant OperationArgs& a [[buffer(0)]],
                                 threadgroup uint* totals [[threadgroup(0)]],
                                 uint digit [[thread_index_in_threadgroup]]) {
  auto s = sampler_state(a);
  uint total = 0;
  for (uint b = 0; b < s.blocks; ++b) {
    const ulong index = ulong(b) * 256 + digit;
    uint v = s.counts[index];
    s.counts[index] = total;
    total += v;
  }
  totals[digit] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base = 0;
  for (uint d = 0; d < digit; ++d) base += totals[d];
  for (uint b = 0; b < s.blocks; ++b) s.counts[ulong(b) * 256 + digit] += base;
}
kernel void sampler_scatter(constant OperationArgs& a [[buffer(0)]],
                            threadgroup uint* bins [[threadgroup(0)]],
                            uint t [[thread_index_in_threadgroup]],
                            uint b [[threadgroup_position_in_grid]]) {
  auto s = sampler_state(a);
  const uint i = b * 256 + t, lane = t % 32, warp = t / 32;
  auto input = a.dimensions[2] % 16 == 0 ? s.first : s.second;
  auto output = a.dimensions[2] % 16 == 0 ? s.second : s.first;
  for (uint w = 0; w < 8; ++w) bins[w * 256 + t] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint digit = i < s.n ? (input[i].key >> a.dimensions[2]) & 255 : 256;
  const uint peers = peers_for(digit);
  if (digit < 256 && lane == ctz(peers))
    bins[warp * 256 + digit] = popcount(peers);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i < s.n) {
    uint rank = popcount(peers & ((1u << lane) - 1));
    for (uint w = 0; w < warp; ++w) rank += bins[w * 256 + digit];
    output[s.counts[ulong(b) * 256 + digit] + rank] = input[i];
  }
}
kernel void sampler_weights(constant OperationArgs& a [[buffer(0)]],
                            threadgroup ulong* tile [[threadgroup(0)]],
                            uint t [[thread_index_in_threadgroup]],
                            uint b [[threadgroup_position_in_grid]]) {
  auto s = sampler_state(a);
  const uint i = b * 256 + t;
  F64 weight(0);
  if (i < s.keep) {
    float v = load(a.address[0], s.first[i].id, a.type);
    if (isfinite(v))
      weight = binary64::exp(
          (F64(v) - F64(load(a.address[0], s.first[0].id, a.type))) /
          F64::raw(a.address[6]));
  }
  tile[t] = weight.bits;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!t) {
    F64 sum(0);
    for (uint j = 0; j < 256; ++j) {
      sum += F64::raw(tile[j]);
      tile[j] = sum.bits;
    }
    ((device ulong*)s.counts)[b] = sum.bits;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i < s.keep) s.probabilities[i] = tile[t];
}
kernel void sampler_scan_blocks(constant OperationArgs& a [[buffer(0)]]) {
  auto s = sampler_state(a);
  auto totals = (device ulong*)s.counts;
  F64 total(0);
  for (uint b = 0; b < (s.keep + 255) / 256; ++b) {
    F64 value = F64::raw(totals[b]);
    totals[b] = total.bits;
    total += value;
  }
  s.scan->total = total.bits;
  atomic_store_explicit(&s.scan->cutoff, s.keep - 1, memory_order_relaxed);
  atomic_store_explicit(&s.scan->selected, s.keep - 1, memory_order_relaxed);
  if (!(total > F64(0))) fail(s.result, 2);
}
inline uint minimum(uint value, threadgroup uint* minima, uint t) {
  for (uint offset = 16; offset; offset /= 2)
    value = min(value, simd_shuffle_down(value, offset));
  if (!(t % 32)) minima[t / 32] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 32) {
    value = t < 8 ? minima[t] : 0xffffffffu;
    for (uint offset = 16; offset; offset /= 2)
      value = min(value, simd_shuffle_down(value, offset));
  }
  return value;
}
kernel void sampler_cutoff(constant OperationArgs& a [[buffer(0)]],
                           threadgroup uint* minima [[threadgroup(0)]],
                           uint t [[thread_index_in_threadgroup]],
                           uint b [[threadgroup_position_in_grid]]) {
  auto s = sampler_state(a);
  const uint i = b * 256 + t;
  uint candidate = 0xffffffffu;
  if (i < s.keep &&
      (F64::raw(((device ulong*)s.counts)[b]) + F64::raw(s.probabilities[i])) /
              F64::raw(s.scan->total) >=
          F64::raw(a.address[7]))
    candidate = i;
  const uint first = minimum(candidate, minima, t);
  if (!t)
    atomic_fetch_min_explicit(&s.scan->cutoff, first, memory_order_relaxed);
}
kernel void sampler_select(constant OperationArgs& a [[buffer(0)]],
                           threadgroup uint* minima [[threadgroup(0)]],
                           uint t [[thread_index_in_threadgroup]],
                           uint b [[threadgroup_position_in_grid]]) {
  auto s = sampler_state(a);
  const uint i = b * 256 + t;
  uint candidate = 0xffffffffu;
  const uint last = atomic_load_explicit(&s.scan->cutoff, memory_order_relaxed);
  auto offsets = (device ulong*)s.counts;
  const F64 retained =
      F64::raw(offsets[last / 256]) + F64::raw(s.probabilities[last]);
  const F64 draw =
      a.flags & 1
          ? F64::raw(s.random->draw)
          : binary64::scalbn(F64::raw(binary64::from_uint(a.first >> 11)), -53);
  if (i <= last &&
      draw < (F64::raw(offsets[b]) + F64::raw(s.probabilities[i])) / retained)
    candidate = i;
  const uint first = minimum(candidate, minima, t);
  if (!t)
    atomic_fetch_min_explicit(&s.scan->selected, first, memory_order_relaxed);
}
kernel void sampler_finish(constant OperationArgs& a [[buffer(0)]]) {
  auto s = sampler_state(a);
  if (!status(s.result))
    s.result->token =
        s
            .first[min(
                atomic_load_explicit(&s.scan->selected, memory_order_relaxed),
                atomic_load_explicit(&s.scan->cutoff, memory_order_relaxed))]
            .id;
}
kernel void sampler_greedy(constant OperationArgs& a [[buffer(0)]],
                           threadgroup uint* memory [[threadgroup(0)]],
                           uint t [[thread_index_in_threadgroup]],
                           uint b [[threadgroup_position_in_grid]]) {
  auto s = sampler_state(a);
  const uint i = b * 256 + t;
  const float v = i < s.n ? load(a.address[0], i, a.type) : -INFINITY;
  if (isnan(v) || v == INFINITY) fail(s.result, 1);
  auto values = (threadgroup float*)memory;
  auto ids = memory + 256;
  values[t] = v;
  ids[t] = i;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = 128; offset; offset /= 2) {
    if (t < offset &&
        (values[t + offset] > values[t] ||
         (values[t + offset] == values[t] && ids[t + offset] < ids[t]))) {
      values[t] = values[t + offset];
      ids[t] = ids[t + offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!t) s.first[b] = {as_type<uint>(values[0]), ids[0]};
}
kernel void sampler_greedy_finish(constant OperationArgs& a [[buffer(0)]]) {
  auto s = sampler_state(a);
  if (status(s.result)) return;
  float value = -INFINITY;
  uint id = 0xffffffffu;
  for (uint i = 0; i < s.blocks; ++i) {
    float v = as_type<float>(s.first[i].key);
    if (v > value || (v == value && s.first[i].id < id)) {
      value = v;
      id = s.first[i].id;
    }
  }
  if (!isfinite(value))
    fail(s.result, 2);
  else
    s.result->token = id;
}
