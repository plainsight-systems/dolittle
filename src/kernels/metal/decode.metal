/*
 * X1/X4 GPU sequence control (metal-port.md, Metal scheduling). One control
 * thread owns position, selected count, pending-token promise and terminal
 * reason. It gates reusable ICBs through a device-written execution range.
 * Body kernels gather/capture rows in parallel using the same position; no
 * host-readable token controls continuation. Completion copies a bounded event
 * journal into a fresh publication window. Cancellation is a device-coherent
 * atomic store on a separate queue, with no spin or payload publication.
 * FP32 norm metadata passes through unchanged; no transformer arithmetic here.
 * Core I.7 (#ri-post), performance GPU.1/GPU.8/GDSA.4 queried/read for
 * implementation: guidelines/gpu/GPU.1-keep-data-on-device.md,
 * guidelines/gpu/GPU.8-make-barriers-describe-real-hazards.md,
 * guidelines/gpu-dsa/GDSA.4-no-cross-workgroup-waits-without-forward-progress.md.
 */
#include <metal_stdlib>
using namespace metal;
struct Event {
  uchar kind;
  uchar padding[3];
  uint status;
  ulong position, count;
  uint token;
  float norm;
};
struct State {
  ulong position, generated;
  uint events, error, pending, active;
  uint range_begin, range_count, injection_reported, done;
};
struct Bindings {
  device State* state;
  device const uint* selection;
  device const uint* terminal;
  device uint* tokens;
  device const uint* prompt;
  coherent(device) device atomic_uint* cancel;
  device Event* events;
  device const uint* injection_status;
  device const float* injection_norm;
  device const uint* intervention_status;
  device const uint* intervention_count;
  ulong prompt_count, generated_limit, chunk, forward_only;
};
struct Step {
  ulong stage, phase, rows, commands;
};
inline bool cancelled(const device Bindings& b) {
  return atomic_load_explicit(b.cancel, memory_order_relaxed) != 0;
}
inline void publish(const device Bindings& b, uchar kind, uint status,
                    ulong pos, ulong count, uint token = 0, float norm = 0) {
  b.events[b.state->events++] = {kind,  {0, 0, 0}, status, pos,
                                 count, token,     norm};
}
kernel void sequence_control(const device Bindings& b [[buffer(0)]],
                             constant Step& step [[buffer(1)]]) {
  device State& s = *b.state;
  if (step.phase == 0) {
    if (!s.injection_reported) {
      s.injection_reported = 1;
      if (b.injection_status) {
        s.error = *b.injection_status;
        publish(b, 3, s.error, 0, 0, 0, *b.injection_norm);
      }
    }
    if (b.intervention_status) s.error |= *b.intervention_status;
    bool run = false;
    if (!s.error) {
      switch (step.stage) {
        case 0:
          run = !cancelled(b) && s.position + step.rows <= b.prompt_count;
          break;
        case 1:
          run = !cancelled(b) && s.position == b.prompt_count - step.rows;
          break;
        case 2:
          run = !cancelled(b) && s.position == b.prompt_count &&
                (b.generated_limit || b.forward_only);
          break;
        case 3:
          run = s.pending;
          break;
        case 4:
          run = s.active && !cancelled(b) && !*b.terminal &&
                s.generated < b.generated_limit;
          break;
      }
    }
    s.active = run;
    s.range_begin = 0;
    s.range_count = run ? uint(step.commands) : 0;
  } else if (step.phase == 1) {
    if (!s.active) return;
    if (step.stage == 0 || step.stage == 1 || step.stage == 3) {
      publish(b, 0, 0, s.position, step.rows);
      s.position += step.rows;
      if (step.stage == 3) s.pending = 0;
      if (b.intervention_status) s.error |= *b.intervention_status;
    } else if (!b.forward_only) {
      s.error |= b.selection[1];
      if (!s.error) {
        ++s.generated;
        s.pending = 1;
        publish(b, 1, 0, s.position, 1, b.selection[0]);
      }
    }
  } else if (step.phase == 2) {
    if (b.intervention_status) {
      s.error |= *b.intervention_status;
      if (s.position >= b.prompt_count && *b.intervention_count != 1)
        s.error = 1;
    }
    uint reason = s.error
                      ? 3
                      : ((s.generated && *b.terminal)
                             ? 1
                             : ((s.position == b.prompt_count + s.generated &&
                                 s.generated == b.generated_limit)
                                    ? 0
                                    : 2));
    publish(b, 2, reason, s.position, s.generated, s.error);
    s.done = 1;
  }
}
kernel void sequence_load(const device Bindings& b [[buffer(0)]],
                          constant Step& step [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < step.rows)
    b.tokens[i] =
        step.stage == 3 ? b.selection[0] : b.prompt[b.state->position + i];
}
kernel void sequence_publish(const device Bindings& b [[buffer(0)]],
                             device ulong* output [[buffer(1)]]) {
  output[0] = b.state->events;
  device Event* events = reinterpret_cast<device Event*>(output + 2);
  for (uint i = 0; i < b.state->events; ++i) events[i] = b.events[i];
  b.state->events = 0;
}
kernel void sequence_cancel(coherent(device) device atomic_uint* flag
                            [[buffer(0)]]) {
  atomic_store_explicit(flag, 1, memory_order_relaxed);
}
struct Copy {
  ulong bytes, rows, marker;
};
kernel void sequence_capture(const device Bindings& b [[buffer(0)]],
                             device const uchar* source [[buffer(1)]],
                             device uchar* target [[buffer(2)]],
                             constant Copy& copy [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < copy.bytes * copy.rows)
    target[b.state->position * copy.bytes + i] = source[i];
}
kernel void sequence_replace(const device Bindings& b [[buffer(0)]],
                             device const uchar* source [[buffer(1)]],
                             device uchar* target [[buffer(2)]],
                             constant Copy& copy [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  if (copy.marker >= b.state->position &&
      copy.marker - b.state->position < copy.rows && i < copy.bytes)
    target[(copy.marker - b.state->position) * copy.bytes + i] = source[i];
}
// Prepared whole-region copies (for example the AR head) remain inside the
// body.
kernel void sequence_copy(device const uchar* source [[buffer(0)]],
                          device uchar* destination [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
  destination[i] = source[i];
}
