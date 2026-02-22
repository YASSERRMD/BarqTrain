cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(barqtrain_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")  # Detect from system
set(CMAKE_BUILD_TYPE Release)

# ---------------------------------------------------------------------------
# Auto-detect PyTorch's CMake prefix from the active Python environment.
# This avoids having to manually set CMAKE_PREFIX_PATH or Torch_DIR.
# Users can still override by passing -DCMAKE_PREFIX_PATH=... to cmake.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_PREFIX_PATH AND NOT DEFINED Torch_DIR)
    find_program(PYTHON_EXEC NAMES python3 python)
    if(PYTHON_EXEC)
        execute_process(
            COMMAND ${PYTHON_EXEC} -c
                "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE TORCH_QUERY_RESULT
        )
        if(TORCH_QUERY_RESULT EQUAL 0 AND TORCH_CMAKE_PREFIX)
            message(STATUS "Auto-detected PyTorch cmake prefix: ${TORCH_CMAKE_PREFIX}")
            list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
        else()
            message(FATAL_ERROR
                "Could not determine PyTorch cmake prefix automatically.\n"
                "Please set it manually:\n"
                "  cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') .."
            )
        endif()
    else()
        message(FATAL_ERROR
            "Python not found. Install Python and PyTorch, then re-run cmake.\n"
            "Or pass the torch cmake prefix directly:\n"
            "  cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') .."
        )
    endif()
endif()

# Find PyTorch and CUDA
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(SRC_FILES
    src/bindings.cpp
    kernels/rmsnorm.cu
    kernels/flash_attention.cu
    kernels/chunked_cross_entropy.cu
    kernels/lora.cu
)

# Build the shared library
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

# Set output properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME ${PROJECT_NAME}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../build
)

# Compile definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xptxas -v")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Install rules (for wheel building)
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION .
)
