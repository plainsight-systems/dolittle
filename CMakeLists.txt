cmake_minimum_required(VERSION 3.24)
# Export the actual target flags/includes for editor and analysis tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(dolittle LANGUAGES C CXX)

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

include(CTest)
if(BUILD_TESTING)
    add_library(dolittle_numerical STATIC tests/support/numerical.cpp)
    target_include_directories(dolittle_numerical PUBLIC "${PROJECT_SOURCE_DIR}")
    target_compile_features(dolittle_numerical PUBLIC cxx_std_20)
    set_target_properties(dolittle_numerical PROPERTIES CXX_EXTENSIONS OFF)
    dolittle_warnings(dolittle_numerical)

    foreach(unit IN ITEMS sampling progression byte_bpe qwen3_chat tensor bf16 observations storage_plan numerical
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
    add_executable(test_checkpoint_archive tests/integration/test_checkpoint_archive.cpp)
    target_include_directories(test_checkpoint_archive PRIVATE "${PROJECT_SOURCE_DIR}")
    target_link_libraries(test_checkpoint_archive PRIVATE dolittle_host)
    set_target_properties(test_checkpoint_archive PROPERTIES CXX_EXTENSIONS OFF)
    dolittle_warnings(test_checkpoint_archive)
    add_test(NAME integration.checkpoint_archive COMMAND test_checkpoint_archive)
    set_tests_properties(integration.checkpoint_archive PROPERTIES LABELS integration)
endif()

option(DOLITTLE_CUDA "Build native CUDA forward execution" OFF)
if(DOLITTLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit 13.2 REQUIRED)
    add_library(dolittle_cuda STATIC
        src/gpu/cuda/device.cpp
        src/kernels/cuda/operations.cpp
        src/kernels/cuda/projection.cu
        src/kernels/cuda/normalization.cu
        src/kernels/cuda/rotary.cu
        src/kernels/cuda/attention.cu
        src/kernels/cuda/embedding.cu
        src/kernels/cuda/elementwise.cu
        src/inference/executor.cpp
        src/inference/cache/contiguous.cpp
        src/inference/generation.cpp)
    target_link_libraries(dolittle_cuda PUBLIC dolittle_host CUDA::cudart)
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
    if(BUILD_TESTING)
        add_executable(test_device tests/gpu/test_device.cpp)
        target_include_directories(test_device PRIVATE "${PROJECT_SOURCE_DIR}")
        target_link_libraries(test_device PRIVATE dolittle_cuda)
        dolittle_warnings(test_device)
        add_test(NAME integration.device COMMAND test_device)
        set_tests_properties(integration.device PROPERTIES LABELS integration RUN_SERIAL TRUE)
        add_executable(test_generation tests/gpu/test_generation.cpp tests/gpu/forward_reference.cpp)
        target_include_directories(test_generation PRIVATE "${PROJECT_SOURCE_DIR}")
        target_link_libraries(test_generation PRIVATE dolittle_cuda dolittle_numerical)
        dolittle_warnings(test_generation)
        set(DOLITTLE_GENERATION_FIXTURES "" CACHE PATH "Independent generation fixture root")
        if(DOLITTLE_GENERATION_FIXTURES)
            add_test(NAME integration.generation_tiny COMMAND test_generation
                "${DOLITTLE_GENERATION_FIXTURES}/tiny-fp32/fixture.json"
                "${DOLITTLE_GENERATION_FIXTURES}/tiny-bf16/fixture.json")
            add_test(NAME integration.generation_8b COMMAND test_generation
                "${DOLITTLE_GENERATION_FIXTURES}/qwen3-8b-bf16/fixture.json")
            set_tests_properties(integration.generation_tiny integration.generation_8b
                PROPERTIES LABELS integration RUN_SERIAL TRUE)
        endif()
    endif()
endif()
