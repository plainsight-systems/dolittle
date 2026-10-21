/*
 * X4: instantiate Apple's pinned metal-cpp dispatch bindings once per binary.
 * No inference or storage algorithm. Core R.1 (#rr-raii) and performance GPU.6
 * (guidelines/gpu/GPU.6-batch-tiny-gpu-work.md), read for implementation, keep
 * native lifetime/dispatch in the backend. Clients borrow these API bindings.
 */
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
