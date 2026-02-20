cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(barqtrain_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")  # Detect from system
set(CMAKE_BUILD_TYPE Release)

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
