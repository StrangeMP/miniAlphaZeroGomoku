cmake_minimum_required(VERSION 3.18)
# Set CUDA Toolkit path before project command
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" CACHE PATH "CUDA Toolkit root directory")
# Set TensorRT path
set(TENSORRT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/TensorRT-10.12.0.36" CACHE PATH "TensorRT root directory")
project(miniAlphaZeroGomoku LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default build type if none is specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Add compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /W3 /utf-8")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od /W3 /utf-8")
endif()

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -lineinfo")
endif()

# Apply flags based on the build type
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}})
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE}})

# Add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find and configure TensorRT
if(EXISTS "${TENSORRT_ROOT_DIR}")
    # Add TensorRT include directory
    include_directories("${TENSORRT_ROOT_DIR}/include")
    
    # Add TensorRT library directory
    link_directories("${TENSORRT_ROOT_DIR}/lib")
    
    # Find TensorRT libraries (with version suffix _10)
    find_library(TENSORRT_NVINFER nvinfer_10 HINTS "${TENSORRT_ROOT_DIR}/lib")
    find_library(TENSORRT_NVINFER_PLUGIN nvinfer_plugin_10 HINTS "${TENSORRT_ROOT_DIR}/lib")
    find_library(TENSORRT_NVONNXPARSER nvonnxparser_10 HINTS "${TENSORRT_ROOT_DIR}/lib")
    find_library(TENSORRT_NVPARSERS nvparsers_10 HINTS "${TENSORRT_ROOT_DIR}/lib")
    
    # Check if TensorRT libraries were found
    if(TENSORRT_NVINFER AND TENSORRT_NVINFER_PLUGIN AND TENSORRT_NVONNXPARSER)
        message(STATUS "TensorRT found at: ${TENSORRT_ROOT_DIR}")
        message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER}")
        message(STATUS "TensorRT nvinfer_plugin: ${TENSORRT_NVINFER_PLUGIN}")
        message(STATUS "TensorRT nvonnxparser: ${TENSORRT_NVONNXPARSER}")
        set(TENSORRT_FOUND TRUE)
        set(TENSORRT_LIBRARIES ${TENSORRT_NVINFER} ${TENSORRT_NVINFER_PLUGIN} ${TENSORRT_NVONNXPARSER})
        if(TENSORRT_NVPARSERS)
            message(STATUS "TensorRT nvparsers: ${TENSORRT_NVPARSERS}")
            list(APPEND TENSORRT_LIBRARIES ${TENSORRT_NVPARSERS})
        endif()
    else()
        message(WARNING "TensorRT libraries not found at: ${TENSORRT_ROOT_DIR}")
        message(WARNING "nvinfer_10: ${TENSORRT_NVINFER}")
        message(WARNING "nvinfer_plugin_10: ${TENSORRT_NVINFER_PLUGIN}")
        message(WARNING "nvonnxparser_10: ${TENSORRT_NVONNXPARSER}")
        set(TENSORRT_FOUND FALSE)
    endif()
else()
    message(WARNING "TensorRT root directory not found: ${TENSORRT_ROOT_DIR}")
    set(TENSORRT_FOUND FALSE)
endif()

# Automatically find all source files in the src directory
file(GLOB_RECURSE MAIN_SOURCES CONFIGURE_DEPENDS
    "src/main.cpp"
    "src/network.cpp"
    "src/dispatcher.cpp"
    "src/board.cpp"
    "src/AsyncLogger.cpp"
)

# Add the main executable
add_executable(main ${MAIN_SOURCES})

# Add compile definitions for Debug and Release after defining the targets
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(main PRIVATE DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(main PRIVATE NDEBUG)
endif()

# Add run_nn_inference executable
add_executable(run_nn_inference src/run_nn_inference.cpp src/network.cpp src/dispatcher.cpp src/board.cpp src/AsyncLogger.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(run_nn_inference PRIVATE DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(run_nn_inference PRIVATE NDEBUG)
endif()

# Add dummy executable (lightweight, no external libraries)
add_executable(dummy src/dummy.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(dummy PRIVATE DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(dummy PRIVATE NDEBUG)
endif()

# Find and link threading library
find_package(Threads REQUIRED)
# Link libraries
target_link_libraries(main PRIVATE Threads::Threads)
target_link_libraries(run_nn_inference PRIVATE Threads::Threads)

# Link TensorRT libraries if found
if(TENSORRT_FOUND)
    target_link_libraries(main PRIVATE ${TENSORRT_LIBRARIES})
    target_link_libraries(run_nn_inference PRIVATE ${TENSORRT_LIBRARIES})
    target_link_libraries(dummy PRIVATE ${TENSORRT_LIBRARIES})
    target_compile_definitions(main PRIVATE TENSORRT_ENABLED)
    target_compile_definitions(run_nn_inference PRIVATE TENSORRT_ENABLED)
    target_compile_definitions(dummy PRIVATE TENSORRT_ENABLED)
    message(STATUS "TensorRT libraries linked to executables")
endif()

target_link_libraries(run_nn_inference PRIVATE cudart)
target_link_libraries(main PRIVATE cudart)
target_link_libraries(dummy PRIVATE cudart)

# Specify output directory for the executable
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(run_nn_inference PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set_target_properties(dummy PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

