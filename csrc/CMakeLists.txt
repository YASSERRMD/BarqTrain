cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(barqtrain_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# ---------------------------------------------------------------------------
# Auto-detect PyTorch cmake prefix from the active Python interpreter.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_PREFIX_PATH AND NOT DEFINED Torch_DIR)
    find_program(PYTHON_EXEC NAMES python3 python)
    if(PYTHON_EXEC)
        execute_process(
            COMMAND ${PYTHON_EXEC} -c
                "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _torch_result
        )
        if(_torch_result EQUAL 0 AND TORCH_CMAKE_PREFIX)
            message(STATUS "PyTorch cmake prefix: ${TORCH_CMAKE_PREFIX}")
            list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
        else()
            message(FATAL_ERROR
                "Cannot find PyTorch cmake prefix. Run:\n"
                "  cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') ..")
        endif()
    else()
        message(FATAL_ERROR "Python not found.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Auto-detect GPU compute capability for faster single-arch compilation.
# Compiling for 'native' (all arches) is the main source of slow builds.
# Colab T4=75, A100=80, L4=89, V100=70. We detect automatically.
# Users can override: cmake -DCUDA_ARCH=80 ..
# ---------------------------------------------------------------------------
if(NOT DEFINED CUDA_ARCH)
    find_program(PYTHON_EXEC NAMES python3 python)
    if(PYTHON_EXEC)
        execute_process(
            COMMAND ${PYTHON_EXEC} -c
                "import torch; print(torch.cuda.get_device_capability()[0]*10 + torch.cuda.get_device_capability()[1]) if torch.cuda.is_available() else print(75)"
            OUTPUT_VARIABLE _DETECTED_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _arch_result
        )
        if(_arch_result EQUAL 0 AND _DETECTED_ARCH)
            set(CUDA_ARCH "${_DETECTED_ARCH}")
            message(STATUS "Auto-detected CUDA arch: sm_${CUDA_ARCH}")
        else()
            set(CUDA_ARCH "75")  # Safe default (T4)
        endif()
    else()
        set(CUDA_ARCH "75")
    endif()
endif()

# Compile for exactly one architecture — 3-5x faster than 'native' multi-arch
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")

# Speed up nvcc: parallel threads during early compilation phases
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --threads 4")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3")

# ---------------------------------------------------------------------------
# Find Python development headers (provides Python.h)
# On Colab / Debian: apt-get install -y python3-dev
# ---------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")

# Find PyTorch and CUDA
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Source files
set(SRC_FILES
    src/bindings.cpp
    kernels/rmsnorm.cu
    kernels/flash_attention.cu
    kernels/chunked_cross_entropy.cu
    kernels/lora.cu
)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

target_link_libraries(${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    Python3::Python
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME ${PROJECT_NAME}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../build
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION .)
