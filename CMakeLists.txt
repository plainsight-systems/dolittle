cmake_minimum_required(VERSION 3.24)
# Export the actual target flags/includes for editor and analysis tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(dolittle LANGUAGES C CXX)
option(DOLITTLE_ALLOCATION_CHECKS "Reject C++/GPU allocations in live generation iterations" OFF)
if(DOLITTLE_ALLOCATION_CHECKS)
    add_compile_definitions(DOLITTLE_ALLOCATION_CHECKS)
endif()

# Shared host preparation and verified checkpoint loading.
include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)
find_package(OpenSSL 3 REQUIRED COMPONENTS Crypto)
# Unicode support only; token matching/BPE and model computation are our own code.
set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
set(UTF8PROC_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(utf8proc
    URL https://codeload.github.com/JuliaStrings/utf8proc/tar.gz/refs/tags/v2.10.0
    URL_HASH SHA256=6f4f1b639daa6dca9f80bc5db1233e9cbaa31a67790887106160b33ef743f136
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TEST OFF CACHE BOOL "" FORCE)
set(INSTALL_DOCUMENTATION OFF CACHE BOOL "" FORCE)
FetchContent_Declare(oniguruma
    URL https://codeload.github.com/kkos/oniguruma/tar.gz/refs/tags/v6.9.10
    URL_HASH SHA256=ad92309d0d13eebc27f6592e875f3efbfa3dda2bf6da5952e00f0a2120c921a8
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(utf8proc oniguruma)

add_library(dolittle_host STATIC
    src/server/wire.cpp
    src/nla/protocol.cpp
    src/nla/pairs/qwen3_8b.cpp
    src/experiment/record.cpp
    src/experiment/model_retention.cpp
    src/gpu/metal/buffer_views.cpp
    src/kernels/metal/sequence_plan.cpp
    src/tokenizers/byte_bpe.cpp
    src/text/qwen3_chat.cpp
    src/inference/sampling/categorical.cpp
    src/inference/progression.cpp
    src/tensors/tensor.cpp
    src/tensors/representations/bf16.cpp
    src/inference/observations.cpp
    src/inference/storage_plan.cpp
    src/kernels/operations.cpp
    src/models/model.cpp
    src/models/qwen3/model.cpp
    src/tensors/formats/format.cpp
    src/tensors/formats/safetensors.cpp
    src/models/checkpoint.cpp
    src/models/qwen3/checkpoint.cpp
)
# Match the explicit FP32 multiply/add boundaries in the frozen AV normalization.
if(NOT MSVC)
    set_source_files_properties(src/nla/protocol.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_include_directories(dolittle_host PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_compile_features(dolittle_host PUBLIC cxx_std_20)
target_link_libraries(dolittle_host PUBLIC nlohmann_json::nlohmann_json OpenSSL::Crypto utf8proc onig)
set_target_properties(dolittle_host PROPERTIES CXX_EXTENSIONS OFF)

function(dolittle_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()
dolittle_warnings(dolittle_host)

find_package(Threads REQUIRED)
target_link_libraries(dolittle_host PUBLIC Threads::Threads)
FetchContent_Declare(httplib
    URL https://codeload.github.com/yhirose/cpp-httplib/tar.gz/refs/tags/v0.62.0
    URL_HASH SHA256=7bba5949c56e772236432f69da03197fd4c12dc443455b9d81f6a05575431fc8
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(HTTPLIB_COMPILE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(httplib)
add_library(dolittle_http STATIC src/server/http.cpp)
target_link_libraries(dolittle_http PUBLIC dolittle_host httplib::httplib Threads::Threads)
dolittle_warnings(dolittle_http)

# Shared execution compiles in the host preset as well. Only final composition
# links a backend; this prevents platform headers leaking into orchestration.
add_library(dolittle_execution STATIC
    src/inference/executor.cpp src/inference/cache/contiguous.cpp
    src/inference/generation.cpp src/experiment/run.cpp src/experiment/session.cpp)
target_link_libraries(dolittle_execution PUBLIC dolittle_host Threads::Threads)
target_compile_features(dolittle_execution PUBLIC cxx_std_20)
dolittle_warnings(dolittle_execution)

include(CTest)
if(BUILD_TESTING)
    add_library(dolittle_numerical STATIC tests/support/numerical.cpp)
    target_include_directories(dolittle_numerical PUBLIC "${PROJECT_SOURCE_DIR}")
    target_compile_features(dolittle_numerical PUBLIC cxx_std_20)
    set_target_properties(dolittle_numerical PROPERTIES CXX_EXTENSIONS OFF)
    dolittle_warnings(dolittle_numerical)

    foreach(unit IN ITEMS model_retention buffer_views sequence_plan wire nla record bounds sampling progression byte_bpe qwen3_chat tensor bf16 observations storage_plan numerical
                          operations model_recipe qwen3_model safetensors qwen3_checkpoint)
        add_executable(test_${unit} tests/core/test_${unit}.cpp)
        target_include_directories(test_${unit} PRIVATE "${PROJECT_SOURCE_DIR}")
        target_link_libraries(test_${unit} PRIVATE dolittle_host dolittle_numerical)
        set_target_properties(test_${unit} PROPERTIES CXX_EXTENSIONS OFF)
        dolittle_warnings(test_${unit})
        add_test(NAME unit.${unit} COMMAND test_${unit})
        set_tests_properties(unit.${unit} PROPERTIES LABELS unit)
    endforeach()

    add_library(dolittle_forward_reference STATIC tests/gpu/forward_reference.cpp)
    target_link_libraries(dolittle_forward_reference PUBLIC dolittle_host dolittle_numerical)
    set_target_properties(dolittle_forward_reference PROPERTIES CXX_EXTENSIONS OFF)
    dolittle_warnings(dolittle_forward_reference)
    add_executable(test_forward_reference tests/core/test_forward_reference.cpp)
    target_link_libraries(test_forward_reference PRIVATE dolittle_forward_reference)
    set_target_properties(test_forward_reference PROPERTIES CXX_EXTENSIONS OFF)
    dolittle_warnings(test_forward_reference)
    add_test(NAME unit.forward_reference COMMAND test_forward_reference)
    set_tests_properties(unit.forward_reference PROPERTIES LABELS unit)

endif()

option(DOLITTLE_CUDA "Build native CUDA forward execution" OFF)
if(DOLITTLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit 13.2 REQUIRED)
    add_library(dolittle_cuda STATIC
        src/gpu/cuda/device.cpp
        src/kernels/cuda/operations.cpp
        src/kernels/cuda/projection.cu
        src/kernels/cuda/sampling.cu
        src/kernels/cuda/decode.cu
        src/kernels/cuda/injection.cu
        src/kernels/cuda/steering.cu
        src/kernels/cuda/normalization.cu
        src/kernels/cuda/rotary.cu
        src/kernels/cuda/attention.cu
        src/kernels/cuda/embedding.cu
        src/kernels/cuda/elementwise.cu)
    target_link_libraries(dolittle_cuda PUBLIC dolittle_host CUDA::cudart)
    # NVCC adds CCCL implicitly; expose the same SDK headers to Clang/editor tools.
    foreach(cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
        target_compile_options(dolittle_cuda PRIVATE "SHELL:-isystem ${cuda_include}/cccl")
    endforeach()
    target_compile_features(dolittle_cuda PUBLIC cxx_std_20)
    set_target_properties(dolittle_cuda PROPERTIES
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
        CXX_EXTENSIONS OFF CUDA_EXTENSIONS OFF
        CUDA_ARCHITECTURES "86-real;86-virtual")
    target_compile_options(dolittle_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
        # NVCC emits GNU line markers in generated host stubs; -Wpedantic
        # rejects those compiler-generated files, so it is CXX-only.
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--Werror=all-warnings;-Xcompiler=-Wall,-Wextra,-Werror>)
    add_executable(dolittle src/main.cpp)
    if(DOLITTLE_ALLOCATION_CHECKS)
        target_sources(dolittle PRIVATE src/diagnostics/allocation.cpp)
    endif()
    target_link_libraries(dolittle PRIVATE dolittle_execution dolittle_http dolittle_cuda)
    dolittle_warnings(dolittle)
    if(BUILD_TESTING)
        add_executable(test_device tests/gpu/test_device.cpp)
        target_include_directories(test_device PRIVATE "${PROJECT_SOURCE_DIR}")
        target_link_libraries(test_device PRIVATE dolittle_execution dolittle_cuda)
        dolittle_warnings(test_device)
        add_test(NAME integration.device COMMAND test_device)
        set_tests_properties(integration.device PROPERTIES LABELS integration RUN_SERIAL TRUE)
        add_executable(test_generation tests/gpu/test_generation.cpp tests/gpu/forward_reference.cpp tests/gpu/interpretation.cpp tests/gpu/operator_reference.cpp)
        if(DOLITTLE_ALLOCATION_CHECKS)
            target_sources(test_generation PRIVATE src/diagnostics/allocation.cpp)
            target_sources(test_device PRIVATE src/diagnostics/allocation.cpp)
        endif()
        target_include_directories(test_generation PRIVATE "${PROJECT_SOURCE_DIR}")
        target_link_libraries(test_generation PRIVATE dolittle_execution dolittle_numerical dolittle_cuda)
        dolittle_warnings(test_generation)
        set(DOLITTLE_GENERATION_FIXTURES "" CACHE PATH "Independent generation fixture root")
        if(DOLITTLE_GENERATION_FIXTURES)
            add_test(NAME integration.generation_tiny COMMAND test_generation
                "${DOLITTLE_GENERATION_FIXTURES}/tiny-fp32/fixture.json"
                "${DOLITTLE_GENERATION_FIXTURES}/tiny-bf16/fixture.json")
            add_test(NAME integration.generation_8b COMMAND test_generation
                "${DOLITTLE_GENERATION_FIXTURES}/qwen3-8b-bf16/fixture.json")
            set_tests_properties(integration.generation_tiny integration.generation_8b
                PROPERTIES LABELS integration RUN_SERIAL TRUE WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        endif()
    endif()
endif()

option(DOLITTLE_METAL "Build native Metal target execution" OFF)
if(DOLITTLE_METAL)
    if(NOT APPLE OR DOLITTLE_CUDA)
        message(FATAL_ERROR "Metal requires macOS and cannot share a binary with CUDA")
    endif()
    set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
    FetchContent_Declare(metal_cpp
        GIT_REPOSITORY https://github.com/apple/metal-cpp.git
        GIT_TAG f567ed836e4cbb85788c42115a2682bbe68097ee)
    FetchContent_MakeAvailable(metal_cpp)
    add_library(dolittle_metal STATIC
        src/gpu/metal/metal_cpp.cpp src/gpu/metal/device.cpp
        src/gpu/metal/submission.cpp src/gpu/metal/commands.cpp
        src/kernels/metal/decode.cpp src/kernels/metal/operations.cpp
        src/kernels/metal/sampling.cpp src/kernels/metal/injection.cpp
        src/kernels/metal/steering.cpp)
    target_include_directories(dolittle_metal SYSTEM PUBLIC "${metal_cpp_SOURCE_DIR}")
    target_link_libraries(dolittle_metal PUBLIC dolittle_host "-framework Metal" "-framework Foundation")
    target_compile_options(dolittle_metal PUBLIC -fblocks)
    find_program(XCRUN xcrun REQUIRED)
    set(METAL_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/dolittle.metallib")
    set(METAL_AIR_FILES)
    foreach(kernel decode embedding projection normalization rotary attention elementwise sampling injection steering)
        set(air "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.air")
        add_custom_command(OUTPUT "${air}"
            COMMAND "${XCRUN}" -sdk macosx metal -std=metal4.0 -Wall -Wextra -Werror
                -fno-fast-math -ffp-contract=off
                -c "${PROJECT_SOURCE_DIR}/src/kernels/metal/${kernel}.metal" -o "${air}"
            DEPENDS "src/kernels/metal/${kernel}.metal" src/kernels/metal/common.metal
                src/kernels/metal/numeric.metal src/kernels/metal/elementary.metal VERBATIM)
        list(APPEND METAL_AIR_FILES "${air}")
    endforeach()
    add_custom_command(OUTPUT "${METAL_LIBRARY}"
        COMMAND "${XCRUN}" -sdk macosx metallib ${METAL_AIR_FILES} -o "${METAL_LIBRARY}"
        DEPENDS ${METAL_AIR_FILES} VERBATIM)
    add_custom_target(dolittle_metallib DEPENDS "${METAL_LIBRARY}")
    add_dependencies(dolittle_metal dolittle_metallib)
    target_compile_definitions(dolittle_metal PRIVATE DOLITTLE_METAL_LIBRARY="${METAL_LIBRARY}")
    dolittle_warnings(dolittle_metal)
    add_executable(dolittle src/main.cpp)
    target_compile_definitions(dolittle PRIVATE DOLITTLE_METAL=1)
    target_link_libraries(dolittle PRIVATE dolittle_execution dolittle_http dolittle_metal)
    if(DOLITTLE_ALLOCATION_CHECKS)
        target_sources(dolittle PRIVATE src/diagnostics/allocation.cpp)
    endif()
    dolittle_warnings(dolittle)
    if(BUILD_TESTING)
        set(METAL_TEST_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/device-test.metallib")
        add_custom_command(OUTPUT "${METAL_TEST_LIBRARY}"
            COMMAND "${XCRUN}" -sdk macosx metal -std=metal4.0 -Wall -Wextra -Werror
                -fno-fast-math -ffp-contract=off -c "${PROJECT_SOURCE_DIR}/tests/gpu/device_metal.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/device-test.air"
            COMMAND "${XCRUN}" -sdk macosx metallib "${CMAKE_CURRENT_BINARY_DIR}/device-test.air" -o "${METAL_TEST_LIBRARY}"
            DEPENDS tests/gpu/device_metal.metal src/kernels/metal/numeric.metal src/kernels/metal/elementary.metal VERBATIM)
        add_custom_target(dolittle_test_metallib DEPENDS "${METAL_TEST_LIBRARY}")
        add_executable(test_device tests/gpu/test_device_metal.cpp tests/gpu/test_device.cpp)
        add_dependencies(test_device dolittle_test_metallib)
        target_include_directories(test_device PRIVATE "${PROJECT_SOURCE_DIR}")
        target_compile_definitions(test_device PRIVATE DOLITTLE_METAL_TEST_LIBRARY="${METAL_TEST_LIBRARY}")
        target_link_libraries(test_device PRIVATE dolittle_execution dolittle_metal)
        target_compile_definitions(test_device PRIVATE DOLITTLE_TEST_METAL=1)
        dolittle_warnings(test_device)
        add_test(NAME integration.device COMMAND test_device)
        set_tests_properties(integration.device PROPERTIES LABELS integration RUN_SERIAL TRUE TIMEOUT 120)
        add_executable(test_generation tests/gpu/test_generation.cpp tests/gpu/forward_reference.cpp tests/gpu/interpretation.cpp tests/gpu/operator_reference.cpp)
        target_include_directories(test_generation PRIVATE "${PROJECT_SOURCE_DIR}")
        target_compile_definitions(test_generation PRIVATE DOLITTLE_TEST_METAL=1)
        target_link_libraries(test_generation PRIVATE dolittle_execution dolittle_numerical dolittle_metal)
        dolittle_warnings(test_generation)
        set(DOLITTLE_GENERATION_FIXTURES "" CACHE PATH "Independent generation fixture root")
        if(DOLITTLE_GENERATION_FIXTURES)
            add_test(NAME integration.generation_tiny COMMAND test_generation
                "${DOLITTLE_GENERATION_FIXTURES}/tiny-fp32/fixture.json"
                "${DOLITTLE_GENERATION_FIXTURES}/tiny-bf16/fixture.json")
            add_test(NAME integration.generation_8b COMMAND test_generation
                "${DOLITTLE_GENERATION_FIXTURES}/qwen3-8b-bf16/fixture.json")
            set_tests_properties(integration.generation_tiny integration.generation_8b
                PROPERTIES LABELS integration RUN_SERIAL TRUE WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        endif()
    endif()
endif()

if(BUILD_TESTING AND TARGET dolittle)
    set(DOLITTLE_APPLICATION_CACHE "" CACHE PATH "Verified revision cache for live application integration")
    if(DOLITTLE_APPLICATION_CACHE)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        add_test(NAME integration.application COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/integration/test_application.py"
            --binary "$<TARGET_FILE:dolittle>" --cache "${DOLITTLE_APPLICATION_CACHE}")
        set_tests_properties(integration.application PROPERTIES LABELS integration RUN_SERIAL TRUE)
    endif()
endif()
