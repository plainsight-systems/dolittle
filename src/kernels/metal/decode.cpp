/*
 * X1/X4 Metal DeviceSequence implementation; contract/algorithm: decode.h and
 * metal-port.md, Metal scheduling. Prepare five reusable ICB bodies (prompt,
 * tail, first head, decode, subsequent head); GPU control chooses full/empty
 * execution ranges. Encode all bounded windows before start, each ending in
 * its own immutable publication buffer. Collection waits for completed windows
 * only. Position, stopping and pending-token consumption remain GPU-owned.
 * Cancellation has a separate prepared queue/dispatch and device atomic word.
 * Hook copies address retained rows using device position. O(recipe+P/B+G)
 * preparation/storage, no live tensor allocation or token-driven submission.
 * Optimization: immutable control arguments/tables are shared by every matching
 * step. Hook/control constants use one bounded staging arena, not a native
 * allocation per token. Publication windows retain separate backing resources.
 * Core R.1/I.7/ES.103 (#rr-raii/#ri-post/#res-overflow), performance GPU.1,
 * GPU.6/GPU.7/GPU.8/GDSA.4 queried/read for implementation; sources
 * guidelines/gpu/GPU.1-keep-data-on-device.md,
 * guidelines/gpu/GPU.6-batch-tiny-gpu-work.md,
 * guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md,
 * guidelines/gpu/GPU.8-make-barriers-describe-real-hazards.md,
 * guidelines/gpu-dsa/GDSA.4-no-cross-workgroup-waits-without-forward-progress.md.
 */
#include "kernels/metal/decode.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <optional>
#include <thread>
#include <vector>

#include "gpu/metal/backend.h"
#include "gpu/metal/commands.h"
#include "gpu/metal/support.h"
#include "inference/decoding/device.h"
#include "kernels/metal/sequence_plan.h"
namespace dolittle::inference {
namespace {
using Access = gpu::detail::BackendAccess;
using gpu::metal::BufferBinding;
using gpu::metal::owned;
constexpr std::size_t state_bytes = 48, range_offset = 32,
                      publication_header = 16;
constexpr std::size_t iterations_per_window = 8, max_steps = 65536;
constexpr std::size_t command_budget = 512ULL * 1024 * 1024;
constexpr std::size_t subsequent_head = 4;
constexpr std::size_t copy_threads = 256;
constexpr std::size_t parameter_storage_bytes = std::size_t{256} * 1024;
constexpr std::size_t control_table_count = 11;
struct Step {
  std::uint64_t stage, phase, rows, commands;
};
struct Copy {
  std::uint64_t bytes, rows, marker;
};
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::length_error("Metal sequence size overflow");
  }
  return a * b;
}
BufferBinding binding(MTL::Buffer& buffer) {
  return {&buffer, 0, buffer.length(), buffer.gpuAddress()};
}
void barrier(MTL4::ComputeCommandEncoder& encoder) {
  encoder.barrierAfterEncoderStages(MTL::StageDispatch | MTL::StageBlit,
                                    MTL::StageDispatch | MTL::StageBlit,
                                    MTL4::VisibilityOptionDevice);
}
}  // namespace
struct DeviceSequence::Impl {
  std::reference_wrapper<gpu::Device> device;
  std::reference_wrapper<gpu::Queue> queue;
  Shape shape;
  kernels::metal::SequencePlan plan;
  gpu::DeviceBuffer state, cancel, events;
  gpu::Queue cancellation_queue;
  std::unique_ptr<gpu::metal::Submission> timeline, cancellation;
  std::array<gpu::metal::CommandBody, subsequent_head + 1> bodies;
  std::array<bool, subsequent_head + 1> recorded{};
  std::optional<std::size_t> recording;
  bool begun{}, complete{}, started{}, stop_sent{};
  gpu::HostBuffer parameters;
  std::size_t parameter_cursor{};
  std::array<gpu::metal::NativePtr<MTL4::ArgumentTable>, control_table_count>
      control_tables;
  std::vector<gpu::metal::NativePtr<MTL::Buffer>> outputs;
  std::vector<gpu::metal::NativePtr<MTL4::ArgumentTable>> tables;
  std::vector<gpu::ConstDeviceRegion> borrowed;
  BufferBinding bindings{};
  gpu::metal::NativePtr<MTL::ComputePipelineState> control, load, publish,
      capture, replace, cancel_kernel;
  std::size_t collected_window{}, collected_base{}, next_event{};
  Impl(gpu::Device& d, gpu::Queue& q, Shape s)
      : device(d),
        queue(q),
        shape(s),
        plan(kernels::metal::plan_sequence(s,
                                           {iterations_per_window, max_steps})),
        state(d.allocate({state_bytes, alignof(std::uint64_t)})),
        cancel(d.allocate({sizeof(std::uint32_t), alignof(std::uint32_t)})),
        events(
            d.allocate({product(max_events(), sizeof(Event)), alignof(Event)})),
        cancellation_queue(d.create_queue()),
        parameters(d.allocate_staging(
            {parameter_storage_bytes, alignof(std::uint64_t)})),
        control(gpu::metal::pipeline(d, DOLITTLE_METAL_LIBRARY,
                                     "sequence_control",
                                     gpu::metal::PipelineUse::direct)),
        load(gpu::metal::pipeline(d, DOLITTLE_METAL_LIBRARY, "sequence_load")),
        publish(gpu::metal::pipeline(d, DOLITTLE_METAL_LIBRARY,
                                     "sequence_publish",
                                     gpu::metal::PipelineUse::direct)),
        capture(gpu::metal::pipeline(d, DOLITTLE_METAL_LIBRARY,
                                     "sequence_capture")),
        replace(gpu::metal::pipeline(d, DOLITTLE_METAL_LIBRARY,
                                     "sequence_replace")),
        cancel_kernel(gpu::metal::pipeline(d, DOLITTLE_METAL_LIBRARY,
                                           "sequence_cancel",
                                           gpu::metal::PipelineUse::direct)) {
    auto zero = d.allocate_staging(
        {state_bytes + sizeof(std::uint32_t), alignof(std::uint64_t)});
    std::ranges::fill(zero.bytes(), std::byte{});
    q.upload({&state, 0, state_bytes}, {&zero, 0, state_bytes});
    q.upload({&cancel, 0, sizeof(std::uint32_t)},
             {&zero, state_bytes, sizeof(std::uint32_t)});
    q.drain();  // Initialization completes before either inference or cancel
                // queue.
  }
  std::size_t max_events() const {
    std::size_t count = 0;
    for (const auto window : plan.windows) {
      count = std::max(count, window.event_capacity);
    }
    return count;
  }
  template <class T>
  BufferBinding parameter(const T& value) {
    const auto offset =
        parameter_cursor +
        (alignof(T) - parameter_cursor % alignof(T)) % alignof(T);
    if (offset > parameters.bytes().size() ||
        sizeof(T) > parameters.bytes().size() - offset) {
      throw std::length_error("Metal prepared parameter arena exhausted");
    }
    std::memcpy(parameters.bytes().subspan(offset, sizeof(T)).data(), &value,
                sizeof(T));
    parameter_cursor = offset + sizeof(T);
    return Access::binding(gpu::HostRegion{&parameters, offset, sizeof(T)},
                           device);
  }
  std::uint64_t address(gpu::ConstDeviceRegion region, std::size_t minimum,
                        bool optional = false) {
    if (!region.owner && optional) {
      require(region.offset == 0 && region.bytes == 0, "invalid absent region");
      return 0;
    }
    require(region.bytes >= minimum, "short sequence operand");
    const auto view = Access::binding(region, device);
    require(view.address % alignof(std::uint32_t) == 0,
            "unaligned sequence operand");
    borrowed.push_back(region);
    return view.address;
  }
  auto arguments(std::span<const BufferBinding> inputs) {
    auto descriptor = owned(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(inputs.size());
    NS::Error* error = nullptr;
    auto table =
        gpu::metal::transfer(Access::native_device(device).newArgumentTable(
            descriptor.get(), &error));
    if (!table) {
      throw std::runtime_error("Metal arguments: " +
                               gpu::metal::describe(error));
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      table->setAddress(inputs[i].address, i);
    }
    tables.push_back(table);
    return table;
  }
  void dispatch(MTL4::ComputeCommandEncoder& encoder,
                MTL::ComputePipelineState& pipeline,
                std::span<const BufferBinding> inputs) {
    barrier(encoder);
    encoder.setComputePipelineState(&pipeline);
    encoder.setArgumentTable(arguments(inputs).get());
    encoder.dispatchThreads({1, 1, 1}, {1, 1, 1});
  }
  void command(MTL4::ComputeCommandEncoder& encoder, std::size_t stage,
               std::size_t phase, std::size_t rows) {
    const auto index = phase == 2 ? control_table_count - 1 : 2 * stage + phase;
    auto& table = control_tables.at(index);
    if (!table) {
      const auto constants =
          parameter(Step{stage, phase, rows,
                         stage < bodies.size() ? bodies.at(stage).count() : 0});
      const std::array inputs{bindings, constants};
      table = arguments(inputs);
    }
    barrier(encoder);
    encoder.setComputePipelineState(control.get());
    encoder.setArgumentTable(table.get());
    encoder.dispatchThreads({1, 1, 1}, {1, 1, 1});
  }
  void body(MTL4::ComputeCommandEncoder& encoder, std::size_t stage,
            std::size_t rows) {
    command(encoder, stage, 0, rows);
    barrier(encoder);
    auto& body = bodies.at(stage);
    encoder.executeCommandsInBuffer(
        body.native(),
        Access::binding(gpu::ConstDeviceRegion{&state, range_offset,
                                               2 * sizeof(std::uint32_t)},
                        device)
            .address);
    command(encoder, stage, 1, rows);
  }
  void append_copy(gpu::ConstDeviceRegion source, gpu::DeviceRegion destination,
                   Copy copy, bool replacement) {
    if (!recording) {
      throw std::invalid_argument("hook outside sequence recording");
    }
    const auto stage = recording.value();
    require(stage == static_cast<std::size_t>(Stage::prefill) ||
                stage == static_cast<std::size_t>(Stage::tail) ||
                stage == static_cast<std::size_t>(Stage::decode),
            "hook outside forward body");
    const auto active_rows =
        stage == static_cast<std::size_t>(Stage::prefill)
            ? shape.chunk
            : (stage == static_cast<std::size_t>(Stage::tail)
                   ? shape.prompt % shape.chunk
                   : 1);
    require(
        copy.rows == active_rows &&
            (!replacement || stage != static_cast<std::size_t>(Stage::decode)),
        "hook rows differ from body");
    require(copy.bytes && copy.rows, "empty hook copy");
    const auto count = product(copy.bytes, replacement ? 1 : copy.rows);
    require(count <= std::numeric_limits<std::uint32_t>::max(),
            "hook dispatch too large");
    require(source.bytes >= count, "short hook source");
    const auto needed = product(
        copy.bytes, replacement ? copy.rows : shape.prompt + shape.generated);
    require(destination.bytes >= needed, "short hook destination");
    if (replacement) {
      require(copy.marker < shape.prompt, "replacement outside prompt");
    }
    borrowed.push_back(source);
    borrowed.push_back(
        {destination.owner, destination.offset, destination.bytes});
    const auto parameter_buffer = parameter(copy);
    const std::array inputs{bindings, Access::binding(source, device),
                            Access::binding(destination, device),
                            parameter_buffer};
    bodies.at(stage).append(replacement ? *replace.get() : *capture.get(),
                            inputs, {count, 1, 1},
                            {std::min<std::size_t>(count, copy_threads), 1, 1});
  }
};
DeviceSequence::DeviceSequence(gpu::Device& device, gpu::Queue& queue,
                               Shape shape) {
  const auto autorelease = gpu::metal::pool();
  impl_ = std::make_unique<Impl>(device, queue, shape);
}
DeviceSequence::~DeviceSequence() {
  if (!impl_) return;
  try {
    if (impl_->recording) Access::record(impl_->queue, impl_->device, nullptr);
    if (impl_->started) request_stop();
    drain();
  } catch (...) {
    Access::drain_noexcept(impl_->device);
  }
}
std::size_t DeviceSequence::device_bytes() {
  return state_bytes + sizeof(std::uint32_t);
}
std::size_t DeviceSequence::host_bytes(Shape shape) {
  const auto plan =
      kernels::metal::plan_sequence(shape, {iterations_per_window, max_steps});
  return product(plan.event_capacity, sizeof(Event)) +
         product(plan.windows.size(), publication_header);
}
gpu::ConstDeviceRegion DeviceSequence::position() const {
  return {&impl_->state, 0, sizeof(std::uint64_t)};
}
void DeviceSequence::begin(gpu::ConstDeviceRegion selection,
                           gpu::ConstDeviceRegion terminal,
                           gpu::DeviceRegion tokens,
                           gpu::ConstDeviceRegion prompt,
                           gpu::ConstDeviceRegion injection_status,
                           gpu::ConstDeviceRegion injection_norm,
                           gpu::ConstDeviceRegion intervention_status,
                           gpu::ConstDeviceRegion intervention_count) {
  auto& s = *impl_;
  require(!s.begun, "sequence already begun");
  require((injection_status.owner != nullptr) ==
                  (injection_norm.owner != nullptr) &&
              (intervention_status.owner != nullptr) ==
                  (intervention_count.owner != nullptr),
          "incomplete control status pair");
  const std::array<std::uint64_t, 15> data{
      s.address({&s.state, 0, state_bytes}, state_bytes),
      s.address(selection, 8, s.shape.forward_only || !s.shape.generated),
      s.address(terminal, 4, s.shape.forward_only || !s.shape.generated),
      s.address({tokens.owner, tokens.offset, tokens.bytes},
                product(std::min(s.shape.chunk, s.shape.prompt),
                        sizeof(std::uint32_t))),
      s.address(prompt, product(s.shape.prompt, sizeof(std::uint32_t))),
      s.address({&s.cancel, 0, 4}, 4),
      s.address({&s.events, 0, s.events.size_bytes()}, s.events.size_bytes()),
      s.address(injection_status, 4, true),
      s.address(injection_norm, 4, true),
      s.address(intervention_status, 4, true),
      s.address(intervention_count, 4, true),
      s.shape.prompt,
      s.shape.generated,
      s.shape.chunk,
      s.shape.forward_only};
  s.bindings = s.parameter(data);
  s.begun = true;
}
void DeviceSequence::begin_stage(Stage stage) {
  auto& s = *impl_;
  const auto index = static_cast<std::size_t>(stage);
  require(s.begun && !s.complete && !s.recording && index < 4 &&
              !s.recorded.at(index),
          "invalid sequence stage");
  Access::record(s.queue, s.device, &s.bodies.at(index));
  s.recording = index;
  s.recorded.at(index) = true;
  if (stage != Stage::first_selection) {
    const auto rows =
        stage == Stage::decode
            ? 1
            : (stage == Stage::tail ? s.shape.prompt % s.shape.chunk
                                    : s.shape.chunk);
    if (!rows) {
      return;
    }
    const auto parameters = s.parameter(Step{index, 0, rows, 0});
    const std::array inputs{s.bindings, parameters};
    s.bodies.at(index).append(
        *s.load.get(), inputs, {rows, 1, 1},
        {std::min<std::size_t>(rows, copy_threads), 1, 1});
  }
}
void DeviceSequence::begin_selection() {
  auto& s = *impl_;
  require(s.recording == static_cast<std::size_t>(Stage::decode),
          "selection outside decode");
  Access::record(s.queue, s.device, nullptr);
  s.recording = subsequent_head;
  s.recorded.at(subsequent_head) = true;
  Access::record(s.queue, s.device, &s.bodies.at(subsequent_head));
}
void DeviceSequence::end_stage() {
  require(impl_->recording.has_value(), "no active stage");
  Access::record(impl_->queue, impl_->device, nullptr);
  impl_->recording.reset();
}
void DeviceSequence::capture(gpu::ConstDeviceRegion source,
                             gpu::DeviceRegion destination,
                             std::size_t row_bytes, std::size_t rows) {
  impl_->append_copy(source, destination, {row_bytes, rows, 0}, false);
}
void DeviceSequence::replace(gpu::ConstDeviceRegion row,
                             gpu::DeviceRegion embeddings,
                             std::size_t prompt_position, std::size_t rows) {
  impl_->append_copy(row, embeddings, {row.bytes, rows, prompt_position}, true);
}
void DeviceSequence::complete() {
  auto& s = *impl_;
  const auto autorelease = gpu::metal::pool();
  require(s.begun && !s.recording && !s.complete,
          "invalid sequence completion");
  for (std::size_t i = 0; i < s.bodies.size(); ++i) {
    if (s.recorded.at(i)) s.bodies.at(i).seal(s.device);
  }
  for (const auto step : s.plan.steps) {
    if (step == kernels::metal::SequenceStep::finalize) continue;
    const auto index = static_cast<std::size_t>(step);
    require(s.recorded.at(index) && s.bodies.at(index).count(),
            "missing sequence body");
  }
  if (s.shape.generated) {
    require(s.recorded.at(subsequent_head), "missing decode selection body");
  }
  s.timeline = std::make_unique<gpu::metal::Submission>(
      s.device, s.queue,
      gpu::metal::TimelineLimits{s.plan.windows.size(), command_budget});
  for (std::size_t w = 0; w < s.plan.windows.size(); ++w) {
    const auto window = s.plan.windows.at(w);
    auto output = owned(Access::native_device(s.device).newBuffer(
        publication_header + product(window.event_capacity, sizeof(Event)),
        MTL::ResourceStorageModeShared));
    s.outputs.push_back(output);
    s.timeline->retain_native(*output.get());
    auto& commands = s.timeline->begin_window(w);
    auto* encoder = commands.computeCommandEncoder();
    if (!encoder) throw std::runtime_error("Metal sequence encoder failed");
    for (std::size_t i = window.first_step;
         i < window.first_step + window.step_count; ++i) {
      const auto step = s.plan.steps.at(i);
      if (step == kernels::metal::SequenceStep::finalize) {
        s.command(*encoder, 0, 2, 0);
        continue;
      }
      const auto index = static_cast<std::size_t>(step);
      const auto rows = step == kernels::metal::SequenceStep::prefill
                            ? s.shape.chunk
                            : (step == kernels::metal::SequenceStep::tail
                                   ? s.shape.prompt % s.shape.chunk
                                   : 1);
      s.body(*encoder, index, rows);
      if (step == kernels::metal::SequenceStep::decode) {
        s.body(*encoder, subsequent_head, 1);
      }
    }
    const std::array inputs{s.bindings, binding(*output.get())};
    s.dispatch(*encoder, *s.publish.get(), inputs);
    encoder->endEncoding();
    s.timeline->end_window(w);
  }
  for (const auto& body : s.bodies) {
    if (body.native()) body.retain(*s.timeline);
  }
  s.timeline->retain(s.borrowed);
  s.timeline->retain_native(
      *Access::binding(
           gpu::HostRegion{&s.parameters, 0, s.parameters.bytes().size()},
           s.device)
           .buffer);
  s.cancellation = std::make_unique<gpu::metal::Submission>(
      s.device, s.cancellation_queue,
      gpu::metal::TimelineLimits{1, command_budget});
  auto& cancel_commands = s.cancellation->begin_window(0);
  auto* encoder = cancel_commands.computeCommandEncoder();
  if (!encoder) throw std::runtime_error("Metal cancellation encoder failed");
  const auto cancel_binding =
      Access::binding(gpu::ConstDeviceRegion{&s.cancel, 0, 4}, s.device);
  s.dispatch(*encoder, *s.cancel_kernel.get(), std::span(&cancel_binding, 1));
  encoder->endEncoding();
  s.cancellation->retain_native(*cancel_binding.buffer);
  s.cancellation->end_window(0);
  s.complete = true;
}
PreparedSequenceResources DeviceSequence::prepared_resources() const {
  const auto& s = *impl_;
  require(s.complete, "sequence not prepared");
  std::size_t command =
      s.timeline->command_bytes() + s.cancellation->command_bytes();
  const auto allocation = [&](const gpu::DeviceBuffer& buffer) {
    return Access::binding(
               gpu::ConstDeviceRegion{&buffer, 0, buffer.size_bytes()},
               s.device)
        .buffer->allocatedSize();
  };
  const auto private_bytes =
      allocation(s.state) + allocation(s.cancel) + allocation(s.events);
  std::size_t shared_bytes =
      Access::binding(
          gpu::ConstHostRegion{&s.parameters, 0, s.parameters.bytes().size()},
          s.device)
          .buffer->allocatedSize();
  for (const auto& buffer : s.outputs) shared_bytes += buffer->allocatedSize();
  auto metadata =
      sizeof(Impl) +
      s.plan.steps.capacity() * sizeof(kernels::metal::SequenceStep) +
      s.plan.windows.capacity() * sizeof(kernels::metal::PublicationWindow) +
      s.tables.capacity() * sizeof(gpu::metal::NativePtr<MTL4::ArgumentTable>) +
      s.borrowed.capacity() * sizeof(gpu::ConstDeviceRegion) +
      s.outputs.capacity() * sizeof(gpu::metal::NativePtr<MTL::Buffer>) +
      s.timeline->host_bytes() + s.cancellation->host_bytes();
  for (const auto& body : s.bodies) {
    command += body.command_bytes();
    metadata += body.host_bytes();
  }
  return {{private_bytes + shared_bytes + command, shared_bytes + metadata,
           private_bytes + shared_bytes + command + metadata},
          command};
}
void DeviceSequence::start() {
  require(impl_->complete && !impl_->started, "invalid sequence start");
  impl_->timeline->start();
  impl_->started = true;
}
void DeviceSequence::request_stop() noexcept {
  if (!impl_->complete || impl_->stop_sent) return;
  try {
    impl_->cancellation->start();
    impl_->stop_sent = true;
  } catch (...) {
    Access::poison(impl_->device);
  }
}
void DeviceSequence::drain() {
  if (impl_->started) impl_->timeline->drain();
  if (impl_->stop_sent) impl_->cancellation->drain();
}
DeviceSequence::Event DeviceSequence::collect(std::size_t index,
                                              const std::stop_token& stop) {
  auto& s = *impl_;
  require(s.started && index == s.next_event, "invalid collection index");
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(gpu::metal::completion_timeout_ms);
  while (s.collected_window < s.outputs.size()) {
    while (s.timeline->completed_prefix() <= s.collected_window) {
      if (stop.stop_requested()) request_stop();
      std::this_thread::sleep_for(gpu::metal::feedback_poll_interval);
      if (std::chrono::steady_clock::now() >= deadline) {
        s.timeline->wait_prefix(s.collected_window + 1);
      }
    }
    s.timeline->wait_prefix(s.collected_window + 1);
    const auto* bytes = static_cast<const std::byte*>(
        s.outputs.at(s.collected_window)->contents());
    std::uint64_t count = 0;
    std::memcpy(&count, bytes, sizeof(count));
    if (count > s.plan.windows.at(s.collected_window).event_capacity) {
      throw std::runtime_error("Metal publication overflow");
    }
    if (index - s.collected_base < count) {
      Event event{};
      const auto offset =
          publication_header + (index - s.collected_base) * sizeof(Event);
      // Native mapped resource is immutable after this window's completion.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      std::memcpy(&event, bytes + offset, sizeof(event));
      ++s.next_event;
      return event;
    }
    s.collected_base += count;
    ++s.collected_window;
  }
  throw std::out_of_range("no further sequence events");
}
}  // namespace dolittle::inference
