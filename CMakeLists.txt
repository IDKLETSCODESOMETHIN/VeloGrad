cmake_minimum_required(VERSION 3.20)
project(velograd LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target modern NVIDIA Compute Capabilities:
# 80 = Ampere (A100), 89 = Ada Lovelace (RTX 4090), 90 = Hopper (H100), 100 = Blackwell (B200)
set(CMAKE_CUDA_ARCHITECTURES 80 89 90 100)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fPIC -lineinfo")

include_directories(include)

file(GLOB_RECURSE VELOGRAD_SOURCES 
    src/compiler/*.cpp
    src/distributed/*.cpp
    src/fault_tolerance/*.cpp
    src/io/*.cpp
    src/kernel/*.cpp
    src/runtime/*.cpp
    src/scheduler/*.cpp
    src/telemetry/*.cpp
)

add_library(velograd SHARED ${VELOGRAD_SOURCES})

set_target_properties(velograd PROPERTIES
    OUTPUT_NAME "velograd"
    VERSION "5.0.0"
    SOVERSION "5"
)

message(STATUS "VeloGrad v5.0 Enterprise Production Engine configured for C++20 / CUDA 9.0+ / 10.0+")
