/*
 * X4 private native ownership and completion helpers. Contract: retain native
 * objects through completion; feedback is published once by Metal, acquired by
 * waiters. No callback throws. Bounded waits fail rather than hiding a stopped
 * device. Algorithm (metal-port.md, Metal scheduling): commit prepared
 * commands, signal completion, acquire feedback, then release storage. O(1) per
 * commit; all callback storage is prepared before submission. Implementation
 * consultation: Core R.1 (#rr-raii), I.7 (#ri-post); performance GPU.7
 * (guidelines/gpu/GPU.7-pipeline-cpu-and-gpu-work.md), GPU.8
 * (guidelines/gpu/GPU.8-make-barriers-describe-real-hazards.md).
 */
#pragma once
#include <Metal/Metal.hpp>
#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
namespace dolittle::gpu::metal {
template <class T>
using NativePtr = std::shared_ptr<T>;
template <class T>
NativePtr<T> transfer(T* object) {
  if (!object) return {};
  return NativePtr<T>(object, [](T* value) noexcept { value->release(); });
}
template <class T>
NativePtr<T> retain(T* object) {
  if (object) object->retain();
  return transfer(object);
}
inline constexpr std::uint64_t completion_timeout_ms = 60000;
inline constexpr auto feedback_poll_interval = std::chrono::microseconds(100);
inline std::string describe(NS::Error* error) {
  return error ? error->localizedDescription()->utf8String()
               : "no native diagnostic";
}
template <class T>
gpu::metal::NativePtr<T> owned(T* object) {
  if (!object) throw std::runtime_error("Metal resource creation failed");
  return gpu::metal::transfer(object);
}
inline auto pool() { return owned(NS::AutoreleasePool::alloc()->init()); }
struct Feedback {
  std::atomic<bool> arrived{false};
  bool failed{};
};
inline auto commit_options(const std::shared_ptr<Feedback>& feedback) {
  auto options = owned(MTL4::CommitOptions::alloc()->init());
  options->addFeedbackHandler(
      [feedback](MTL4::CommitFeedback* result) noexcept {
        feedback->failed = !result || result->error();
        feedback->arrived.store(true, std::memory_order_release);
      });
  return options;
}
inline void wait_feedback(const Feedback& feedback) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(completion_timeout_ms);
  while (!feedback.arrived.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("Metal feedback timed out");
    }
    std::this_thread::sleep_for(feedback_poll_interval);
  }
  if (feedback.failed) {
    throw std::runtime_error("Metal command execution failed");
  }
}
inline auto residency(MTL::Device& device) {
  auto descriptor = owned(MTL::ResidencySetDescriptor::alloc()->init());
  NS::Error* error = nullptr;
  auto result =
      gpu::metal::transfer(device.newResidencySet(descriptor.get(), &error));
  if (!result) throw std::runtime_error("Metal residency: " + describe(error));
  return result;
}
}  // namespace dolittle::gpu::metal
