cmake_minimum_required(VERSION 3.10)
project(LLM_Inference_Engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/tensor.cpp
    src/matmul.cpp
    src/model.cpp
    src/layers.cpp
    src/embedding.cpp
    src/forward_pass.cpp
    src/tokenizer.cpp
    src/inference_engine.cpp
    src/main.cpp
)

# Create executable
add_executable(llm_engine ${SOURCES})

# Link against filesystem library (needed for C++17 std::filesystem on some systems)
if(NOT MSVC)
    target_link_libraries(llm_engine PRIVATE stdc++fs)
endif()

# Optional: Enable optimization flags
if(MSVC)
    target_compile_options(llm_engine PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(llm_engine PRIVATE -O3 -march=native)
endif()

# Optional: Add OpenMP for multithreading
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(llm_engine PUBLIC OpenMP::OpenMP_CXX)
endif()
