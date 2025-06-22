cmake_minimum_required(VERSION 3.20)
project(AdaptystNVGPU
  DESCRIPTION "The NVIDIA GPU Nsight-based hardware module for Adaptyst")

set(ADAPTYST_MODULE_PATH "/opt/adaptyst" CACHE STRING "Path where Adaptyst modules are installed")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(adaptyst REQUIRED)

add_library(nvgpu SHARED
  src/nvgpu.cpp)

target_compile_definitions(nvgpu PRIVATE ADAPTYST_MODULE_PATH="${ADAPTYST_MODULE_PATH}")
target_include_directories(nvgpu PRIVATE src)
target_link_libraries(nvgpu PUBLIC adaptyst::adaptyst)

install(TARGETS nvgpu LIBRARY DESTINATION ${ADAPTYST_MODULE_PATH})
