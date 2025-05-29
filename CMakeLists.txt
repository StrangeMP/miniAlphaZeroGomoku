cmake_minimum_required(VERSION 3.10)

project(AlphaZeroGomoku LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(Include, tensorflow)

# Add source files
set(SOURCES
    src/mcts_agent.cpp
    test/test.cpp
)

# Add executable
add_executable(AlphaZeroGomoku_test ${SOURCES})

# Add a message to indicate the build is complete (optional)
message(STATUS "CMake configuration complete. Build with your chosen generator.")

# Example for MSVC:
# mkdir build
# cd build
# cmake ..
# cmake --build .

# Example for MinGW/GCC/Clang:
# mkdir build
# cd build
# cmake .. -G "MinGW Makefiles"  # Or "Unix Makefiles" or "Ninja"
# cmake --build .
