/*
 * AR/steering control extension, X1/X4: forward_only performs all fixed input
 * chunks followed by one vector head, publishing no sampled IDs. Null sampler
 * regions are valid only in that mode. Mutation status is latched at chunk and
 * selection boundaries; terminal state checks one application after full
 * replay. All decisions read device words. No extra host transfers or hot
 * allocations. Overlay: steering.md, Device ownership and model phases. Core
 * I.7 (#ri-post) and performance GPU.1
 * (guidelines/gpu/GPU.1-keep-data-on-device.md), queried and read for
 * implementation 2026-10-20, enforce explicit autonomous control. CUDA
 * execution of inference/decoding/device.h; axis X1/X4. Contract/algorithm:
 * execution-boundary.md sections 3-4. One root graph owns full-chunk WHILE,
 * remainder IF, first-selection IF and decode WHILE bodies. Device position
 * controls ID gathers, KV placement and capture destinations. Decode commits
 * consumption before its conditional head/sample tail; every selected ID is
 * consumed even when cancellation races with its publication. Capture bytes
 * copy D2D into independent owners. Only immutable 32-byte events cross to
 * mapped host memory. Release/acquire publishes them without CPU ack. Graph
 * edges order all readers before state/scratch reuse; no grid spin waits.
 * Failure ends partial capture and drains before destruction. All regions and
 * journal bounds are admitted before launch; collection never allocates.
 * Optimization: graph size is O(recipe), host output O(P+G), device control
 * O(1). Design and implementation consultation 2026-10-20: Core I.1/R.1 make
 * region, lifetime and readiness explicit; Performance GPU.1/GPU.6/GPU.7/GPU.9
 * keep intermediates resident, batch dispatch and separate observation
 * resources. Both local corpora queried/read in both phases. Sources:
 * guidelines/gpu/ GPU.1-keep-data-on-device.md; NVIDIA CUDA Programming
 * Guide 4.2 and 5.7.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cuda/atomic>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>

#include "bounds.h"
#include "gpu/cuda/backend.h"
#include "inference/decoding/device.h"

namespace dolittle::inference {
namespace {
using Access = gpu::detail::BackendAccess;
using Stage = DeviceSequence::Stage;
using Kind = DeviceSequence::Kind;
using Reason = DeviceSequence::Reason;
using Event = DeviceSequence::Event;
using Atomic = cuda::atomic_ref<unsigned, cuda::thread_scope_system>;
constexpr std::size_t publication_alignment = 64;
constexpr unsigned copy_threads = 256;
constexpr std::size_t condition_count = 5;
constexpr auto poll_interval = std::chrono::microseconds(100);
struct State {
  std::size_t position, generated;
  unsigned events, error;
};
struct Publication {
  alignas(publication_alignment) unsigned completed;
  alignas(publication_alignment) unsigned cancel;
};
struct Control {
  State* state{};
  const unsigned *selection{}, *terminal{}, *prompt_ids{}, *injection_status{};
  const float* injection_norm{};
  const unsigned *intervention_status{}, *intervention_count{};
  bool forward_only = false;
  unsigned* token_input{};
  Publication* publication{};
  Event* events{};
  std::size_t prompt{}, generated{}, chunk{};
  // Trivially copied kernel arguments require a CUDA-accessible native array.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  cudaGraphConditionalHandle condition[condition_count]{};
};
// Native CUDA pointer indexing is bounded by admitted prompt/event/capture
// extents. Kernels execute in graph order; only Publication uses system
// atomics.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-type-reinterpret-cast)
__device__ bool cancelled(Control c) {
  return Atomic(c.publication->cancel).load(cuda::memory_order_acquire) != 0;
}
__device__ void publish(Control c, Event e) {
  c.events[c.state->events++] = e;
  Atomic(c.publication->completed)
      .store(c.state->events, cuda::memory_order_release);
}
__global__ void initialize(State* state) { *state = {}; }
__global__ void decide_stage(Control c, Stage stage) {
  auto& s = *c.state;
  if (stage == Stage::prefill && c.injection_status) {
    s.error = *c.injection_status;
    publish(c, {Kind::injection, s.error, 0, 0, 0, *c.injection_norm});
  }
  if (c.intervention_status) {
    s.error |= *c.intervention_status;
  }
  bool run = false;
  if (!s.error) {
    switch (stage) {
      case Stage::prefill:
        run = !cancelled(c) && s.position + c.chunk <= c.prompt;
        break;
      case Stage::tail:
        run = !cancelled(c) && c.prompt % c.chunk != 0 &&
              s.position == c.prompt - c.prompt % c.chunk;
        break;
      case Stage::first_selection:
        run = !cancelled(c) && s.position == c.prompt &&
              (c.generated != 0 || c.forward_only);
        break;
      case Stage::decode:
        // A selection promises consumption, including EOS and cancellation.
        run = s.generated != 0;
        break;
    }
  }
  cudaGraphSetConditional(c.condition[static_cast<unsigned>(stage)], run);
}
__global__ void load_tokens(Control c, std::size_t rows, bool decode) {
  for (std::size_t i = threadIdx.x; i < rows; i += blockDim.x) {
    c.token_input[i] =
        decode ? c.selection[0] : c.prompt_ids[c.state->position + i];
  }
}
__global__ void finish_prompt(Control c, std::size_t rows, bool repeat) {
  auto& s = *c.state;
  publish(c, {Kind::consumed, 0, s.position, rows, 0, 0});
  s.position += rows;
  if (c.intervention_status) {
    s.error |= *c.intervention_status;
  }
  if (repeat) {
    cudaGraphSetConditional(
        c.condition[0],
        !s.error && !cancelled(c) && s.position + c.chunk <= c.prompt);
  }
}
__global__ void finish_selection(Control c) {
  auto& s = *c.state;
  s.error = c.selection[1];
  if (!s.error) {
    ++s.generated;
    publish(c, {Kind::selected, 0, s.position, 1, c.selection[0], 0});
  }
}
__global__ void finish_consumption(Control c) {
  auto& s = *c.state;
  publish(c, {Kind::consumed, 0, s.position, 1, 0, 0});
  ++s.position;
  if (c.intervention_status) {
    s.error |= *c.intervention_status;
  }
  const bool more =
      !s.error && !*c.terminal && s.generated < c.generated && !cancelled(c);
  cudaGraphSetConditional(c.condition[4], more);
  // The loop continues only if the selection branch below succeeds.
  cudaGraphSetConditional(c.condition[3], 0);
}
__global__ void continue_decode(Control c) {
  cudaGraphSetConditional(c.condition[3], !c.state->error);
}
__global__ void finish_sequence(Control c) {
  auto& s = *c.state;
  if (c.intervention_status) {
    s.error |= *c.intervention_status;
    if (s.position >= c.prompt && *c.intervention_count != 1) {
      s.error = 1;
    }
  }
  const auto reason = s.error ? Reason::failure
                              : (s.generated && *c.terminal
                                     ? Reason::eos
                                     : (s.position == c.prompt + s.generated &&
                                                s.generated == c.generated
                                            ? Reason::limit
                                            : Reason::cancelled));
  publish(c, {Kind::terminal, static_cast<unsigned>(reason), s.position,
              s.generated, s.error, 0});
}
struct Copy {
  const std::byte* source;
  std::byte* destination;
  std::size_t row_bytes, rows, marker;
};
__global__ void copy_capture(Control c, Copy copy) {
  const auto size = copy.rows * copy.row_bytes;
  auto* target = copy.destination + c.state->position * copy.row_bytes;
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  if ((reinterpret_cast<std::uintptr_t>(copy.source) |
       reinterpret_cast<std::uintptr_t>(target) | size) %
          sizeof(unsigned) ==
      0) {
    const auto* from = reinterpret_cast<const unsigned*>(copy.source);
    auto* to = reinterpret_cast<unsigned*>(target);
    for (auto i = first; i < size / sizeof(unsigned); i += stride) {
      to[i] = from[i];
    }
  } else {
    for (auto i = first; i < size; i += stride) {
      target[i] = copy.source[i];
    }
  }
}
__global__ void replace_embedding(Control c, Copy copy) {
  const auto position = c.state->position;
  if (copy.marker < position || copy.marker - position >= copy.rows) {
    return;
  }
  auto* target = copy.destination + (copy.marker - position) * copy.row_bytes;
  for (std::size_t i = threadIdx.x; i < copy.row_bytes; i += blockDim.x) {
    target[i] = copy.source[i];
  }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-type-reinterpret-cast)
std::size_t event_capacity(DeviceSequence::Shape s) {
  const auto max = std::numeric_limits<std::size_t>::max();
  if (!s.prompt || !s.chunk || s.prompt > max - 2 ||
      s.generated > (max - s.prompt - 2) / 2 ||
      s.chunk > s.prompt + s.generated) {
    throw std::length_error("sequence capacity");
  }
  const auto n = s.prompt + 2 * s.generated + 2;
  if (n > std::numeric_limits<unsigned>::max()) {
    throw std::length_error("event count");
  }
  return n;
}
}  // namespace
struct DeviceSequence::Impl {
  std::reference_wrapper<gpu::Device> device;
  std::reference_wrapper<gpu::Queue> queue;
  cudaStream_t stream;
  gpu::DeviceBuffer state;
  gpu::HostBuffer output;
  Publication* host{};
  Control control{};
  cudaGraph_t graph{}, body{};
  cudaGraphNode_t root_tail{}, stage_node{}, selection_node{};
  cudaGraphExec_t executable{};
  Stage stage{};
  unsigned next_stage = 0;
  bool capturing = false, begun = false, in_stage = false, selection = false,
       launched = false;
  Impl(gpu::Device& d, gpu::Queue& q, Shape shape)
      : device(d),
        queue(q),
        stream(Access::stream(q, d)),
        state(d.allocate({sizeof(State), alignof(State)})),
        output(d.allocate_staging(
            {DeviceSequence::host_bytes(shape), alignof(Publication)})) {
    // Native structure boundary; allocation and offsets have verified
    // alignment. NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    host = reinterpret_cast<Publication*>(output.bytes().data());
    std::construct_at(host);
    control.prompt = shape.prompt;
    control.generated = shape.generated;
    control.chunk = shape.chunk;
    control.forward_only = shape.forward_only;
  }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;
  void check(cudaError_t e) { Access::check(device, e); }
  void stop() noexcept {
    Atomic(host->cancel).store(1, cuda::memory_order_release);
  }
  ~Impl() {
    stop();
    if (capturing) {
      cudaGraph_t ignored{};
      (void)cudaStreamEndCapture(stream, &ignored);
    }
    Access::drain_noexcept(device);
    if (executable) {
      (void)cudaGraphExecDestroy(executable);
    }
    if (graph) {
      (void)cudaGraphDestroy(graph);
    }
  }
  const std::byte* pointer(gpu::ConstDeviceRegion r) {
    if (!r.owner || r.offset > r.owner->size_bytes() ||
        r.bytes > r.owner->size_bytes() - r.offset) {
      throw std::invalid_argument("invalid sequence region");
    }
    const auto* base =
        static_cast<const std::byte*>(Access::pointer(*r.owner, device));
    return std::span(base, r.owner->size_bytes())
        .subspan(r.offset, r.bytes)
        .data();
  }
  std::byte* pointer(gpu::DeviceRegion r) {
    (void)pointer(gpu::ConstDeviceRegion{r.owner, r.offset, r.bytes});
    auto* base = static_cast<std::byte*>(Access::pointer(*r.owner, device));
    return std::span(base, r.owner->size_bytes())
        .subspan(r.offset, r.bytes)
        .data();
  }
  void capture_into(cudaGraph_t target, cudaGraphNode_t predecessor = {}) {
    check(cudaStreamBeginCaptureToGraph(
        stream, target, predecessor ? &predecessor : nullptr, nullptr,
        predecessor ? 1 : 0, cudaStreamCaptureModeThreadLocal));
    capturing = true;
  }
  cudaGraphNode_t end_capture() {
    cudaStreamCaptureStatus status{};
    const cudaGraphNode_t* dependencies{};
    std::size_t count{};
    check(cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr,
                                   &dependencies, nullptr, &count));
    if (count != 1) {
      throw std::logic_error("sequence body needs one ordered tail");
    }
    const auto last = *dependencies;
    cudaGraph_t captured{};
    auto result = cudaStreamEndCapture(stream, &captured);
    capturing = false;
    check(result);
    return last;
  }
  cudaGraph_t conditional(cudaGraph_t parent, cudaGraphNode_t predecessor,
                          unsigned index, bool loop, cudaGraphNode_t& node) {
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    // CUDA tagged union selected by the explicit type above.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    params.conditional.handle =
        dolittle::at(std::span(control.condition), index);
    params.conditional.type =
        loop ? cudaGraphCondTypeWhile : cudaGraphCondTypeIf;
    params.conditional.size = 1;
    check(cudaGraphAddNode(&node, parent, &predecessor, nullptr, 1, &params));
    return *params.conditional.phGraph_out;
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
  }
};
std::size_t DeviceSequence::device_bytes() { return sizeof(State); }
std::size_t DeviceSequence::host_bytes(Shape s) {
  const auto n = event_capacity(s);
  if (n > (std::numeric_limits<std::size_t>::max() - sizeof(Publication)) /
              sizeof(Event)) {
    throw std::length_error("sequence journal bytes");
  }
  return sizeof(Publication) + n * sizeof(Event);
}
DeviceSequence::DeviceSequence(gpu::Device& d, gpu::Queue& q, Shape s)
    : impl_(std::make_unique<Impl>(d, q, s)) {}
DeviceSequence::~DeviceSequence() = default;
gpu::ConstDeviceRegion DeviceSequence::position() const {
  return {&impl_->state, offsetof(State, position), sizeof(std::size_t)};
}
void DeviceSequence::begin(gpu::ConstDeviceRegion selection,
                           gpu::ConstDeviceRegion terminal,
                           gpu::DeviceRegion tokens,
                           gpu::ConstDeviceRegion prompt,
                           gpu::ConstDeviceRegion injection_status,
                           gpu::ConstDeviceRegion injection_norm,
                           gpu::ConstDeviceRegion intervention_status,
                           gpu::ConstDeviceRegion intervention_count) {
  auto& p = *impl_;
  if (p.begun ||
      (!p.control.forward_only && (selection.bytes != 2 * sizeof(unsigned) ||
                                   terminal.bytes != sizeof(unsigned))) ||
      (p.control.forward_only &&
       (p.control.generated || selection.owner || terminal.owner)) ||
      tokens.bytes < p.control.chunk * sizeof(unsigned) ||
      prompt.bytes != p.control.prompt * sizeof(unsigned) ||
      selection.offset % alignof(unsigned) ||
      terminal.offset % alignof(unsigned) ||
      tokens.offset % alignof(unsigned) || prompt.offset % alignof(unsigned)) {
    throw std::logic_error("invalid sequence begin");
  }
  void* mapped{};
  p.check(cudaHostGetDevicePointer(&mapped, p.output.bytes().data(), 0));
  // Checked native typed boundaries; the graph borrows these buffers through
  // drain. NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  p.control.state = static_cast<State*>(Access::pointer(p.state, p.device));
  p.control.publication = static_cast<Publication*>(mapped);
  p.control.events = reinterpret_cast<Event*>(
      std::span(static_cast<std::byte*>(mapped), p.output.bytes().size())
          .subspan(sizeof(Publication))
          .data());
  if (!p.control.forward_only) {
    p.control.selection =
        reinterpret_cast<const unsigned*>(p.pointer(selection));
    p.control.terminal = reinterpret_cast<const unsigned*>(p.pointer(terminal));
  }
  if (intervention_status.owner || intervention_count.owner) {
    if (intervention_status.bytes != sizeof(unsigned) ||
        intervention_count.bytes != sizeof(unsigned) ||
        intervention_status.offset % alignof(unsigned) ||
        intervention_count.offset % alignof(unsigned)) {
      throw std::invalid_argument("intervention control bounds");
    }
    p.control.intervention_status =
        reinterpret_cast<const unsigned*>(p.pointer(intervention_status));
    p.control.intervention_count =
        reinterpret_cast<const unsigned*>(p.pointer(intervention_count));
  }
  p.control.prompt_ids = reinterpret_cast<const unsigned*>(p.pointer(prompt));
  p.control.token_input = reinterpret_cast<unsigned*>(p.pointer(tokens));
  if (injection_status.owner) {
    if (injection_status.bytes != sizeof(unsigned) ||
        injection_norm.bytes != sizeof(float) ||
        injection_status.offset % alignof(unsigned) ||
        injection_norm.offset % alignof(float)) {
      throw std::invalid_argument("invalid injection controls");
    }
    p.control.injection_status =
        reinterpret_cast<const unsigned*>(p.pointer(injection_status));
    p.control.injection_norm =
        reinterpret_cast<const float*>(p.pointer(injection_norm));
  }
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  p.check(cudaGraphCreate(&p.graph, 0));
  for (auto& handle : p.control.condition) {
    p.check(cudaGraphConditionalHandleCreate(&handle, p.graph, 0,
                                             cudaGraphCondAssignDefault));
  }
  p.capture_into(p.graph);
  initialize<<<1, 1, 0, p.stream>>>(p.control.state);
  Access::check_launch(p.device);
  p.begun = true;
}
void DeviceSequence::begin_stage(Stage stage) {
  auto& p = *impl_;
  if (!p.begun || p.in_stage || static_cast<unsigned>(stage) != p.next_stage) {
    throw std::logic_error("invalid sequence stage order");
  }
  if (!p.capturing) {
    p.capture_into(p.graph, p.root_tail);
  }
  decide_stage<<<1, 1, 0, p.stream>>>(p.control, stage);
  Access::check_launch(p.device);
  const auto predecessor = p.end_capture();
  p.stage = stage;
  p.in_stage = true;
  p.body = p.conditional(p.graph, predecessor, static_cast<unsigned>(stage),
                         stage == Stage::prefill || stage == Stage::decode,
                         p.stage_node);
  p.capture_into(p.body);
  if (stage != Stage::first_selection) {
    const auto rows =
        stage == Stage::prefill
            ? p.control.chunk
            : (stage == Stage::tail ? p.control.prompt % p.control.chunk : 1);
    // Empty remainder body still needs a node to make capture endpoints
    // explicit.
    load_tokens<<<1, copy_threads, 0, p.stream>>>(p.control, rows,
                                                  stage == Stage::decode);
    Access::check_launch(p.device);
  }
}
void DeviceSequence::begin_selection() {
  auto& p = *impl_;
  if (!p.in_stage || p.stage != Stage::decode || p.selection) {
    throw std::logic_error("invalid decode selection branch");
  }
  finish_consumption<<<1, 1, 0, p.stream>>>(p.control);
  Access::check_launch(p.device);
  const auto predecessor = p.end_capture();
  const auto body =
      p.conditional(p.body, predecessor, 4, false, p.selection_node);
  p.capture_into(body);
  p.selection = true;
}
void DeviceSequence::end_stage() {
  auto& p = *impl_;
  if (!p.in_stage) {
    throw std::logic_error("no sequence stage");
  }
  if (p.stage == Stage::prefill || p.stage == Stage::tail) {
    const auto rows = p.stage == Stage::prefill
                          ? p.control.chunk
                          : p.control.prompt % p.control.chunk;
    finish_prompt<<<1, 1, 0, p.stream>>>(p.control, rows,
                                         p.stage == Stage::prefill);
  } else {
    if (p.stage == Stage::decode && !p.selection) {
      throw std::logic_error("missing selection tail");
    }
    if (!p.control.forward_only) {
      finish_selection<<<1, 1, 0, p.stream>>>(p.control);
    }
    if (p.stage == Stage::decode) {
      continue_decode<<<1, 1, 0, p.stream>>>(p.control);
    }
  }
  Access::check_launch(p.device);
  (void)p.end_capture();
  p.root_tail = p.stage_node;
  p.in_stage = false;
  ++p.next_stage;
}
void DeviceSequence::capture(gpu::ConstDeviceRegion source,
                             gpu::DeviceRegion destination,
                             std::size_t row_bytes, std::size_t rows) {
  auto& p = *impl_;
  const auto capacity = p.control.prompt + p.control.generated;
  if (p.stage == Stage::first_selection || p.selection) {
    throw std::logic_error("capture requires a consumption body");
  }
  if (!p.in_stage || !row_bytes || !rows || rows > p.control.chunk ||
      row_bytes > std::numeric_limits<std::size_t>::max() / capacity ||
      source.bytes != rows * row_bytes ||
      destination.bytes != capacity * row_bytes) {
    throw std::invalid_argument("invalid sequence capture bounds");
  }
  const Copy copy{p.pointer(source), p.pointer(destination), row_bytes, rows,
                  0};
  const auto blocks = std::min<std::size_t>(
      65535, 1 + (source.bytes - 1) / (copy_threads * sizeof(unsigned)));
  copy_capture<<<static_cast<unsigned>(blocks), copy_threads, 0, p.stream>>>(
      p.control, copy);
  Access::check_launch(p.device);
}
void DeviceSequence::replace(gpu::ConstDeviceRegion row,
                             gpu::DeviceRegion embeddings, std::size_t marker,
                             std::size_t rows) {
  auto& p = *impl_;
  if (!p.in_stage || marker >= p.control.prompt || !rows || !row.bytes ||
      row.bytes > embeddings.bytes / rows) {
    throw std::invalid_argument("embedding override bounds");
  }
  replace_embedding<<<1, copy_threads, 0, p.stream>>>(
      p.control,
      Copy{p.pointer(row), p.pointer(embeddings), row.bytes, rows, marker});
  Access::check_launch(p.device);
}
void DeviceSequence::complete() {
  auto& p = *impl_;
  if (p.next_stage != 4 || p.in_stage || p.executable) {
    throw std::logic_error("incomplete sequence");
  }
  p.capture_into(p.graph, p.root_tail);
  finish_sequence<<<1, 1, 0, p.stream>>>(p.control);
  Access::check_launch(p.device);
  (void)p.end_capture();
  p.check(cudaGraphInstantiate(&p.executable, p.graph, 0));
  p.check(cudaGraphUpload(p.executable, p.stream));
}
void DeviceSequence::start() {
  auto& p = *impl_;
  if (!p.executable || p.launched) {
    throw std::logic_error("sequence start");
  }
  p.check(cudaGraphLaunch(p.executable, p.stream));
  p.launched = true;
}
void DeviceSequence::request_stop() noexcept { impl_->stop(); }
void DeviceSequence::drain() { impl_->queue.get().drain(); }
DeviceSequence::Event DeviceSequence::collect(std::size_t index,
                                              const std::stop_token& stop) {
  auto& p = *impl_;
  if (!p.launched ||
      index >= event_capacity(
                   {p.control.prompt, p.control.generated, p.control.chunk})) {
    throw std::logic_error("event outside sequence");
  }
  while (Atomic(p.host->completed).load(cuda::memory_order_acquire) <= index) {
    if (stop.stop_requested()) {
      p.stop();
    }
    const auto status = cudaStreamQuery(p.stream);
    if (status == cudaSuccess) {
      if (Atomic(p.host->completed).load(cuda::memory_order_acquire) <= index) {
        throw std::runtime_error("sequence ended without event");
      }
      break;
    }
    if (status != cudaErrorNotReady) {
      p.check(status);
    }
    std::this_thread::sleep_for(poll_interval);
  }
  Event event{};
  std::memcpy(&event,
              p.output.bytes()
                  .subspan(sizeof(Publication) + index * sizeof(Event))
                  .data(),
              sizeof(Event));
  return event;
}
}  // namespace dolittle::inference
