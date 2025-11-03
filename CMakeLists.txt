# SPDX-FileCopyrightText: 2025 CERN
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)
project(adaptyst-nvgpu
  DESCRIPTION "The NVIDIA GPU hardware module for Adaptyst")

# include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(adaptyst REQUIRED)

message(STATUS "Adaptyst module path: ${ADAPTYST_MODULE_PATH}")
set(INSTALL_PATH "${ADAPTYST_MODULE_PATH}" CACHE STRING "Path where adaptyst-nvgpu should be installed")

find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json REQUIRED)

# FetchContent_Declare(concurrentqueue
#   GIT_REPOSITORY https://github.com/cameron314/concurrentqueue
#   GIT_SHALLOW FALSE)
# FetchContent_MakeAvailable(concurrentqueue)

add_library(nvgpu SHARED
  src/nvgpu.cpp)

add_library(nvgpu_inject SHARED
  src/nvgpu_inject.cpp)

target_compile_definitions(nvgpu PRIVATE MODULE_PATH="${INSTALL_PATH}/nvgpu")
target_include_directories(nvgpu PRIVATE src)
target_link_libraries(nvgpu PUBLIC adaptyst::adaptyst)
target_link_libraries(nvgpu_inject PUBLIC adaptyst::adaptyst_inject CUDA::cupti nlohmann_json::nlohmann_json)

install(TARGETS nvgpu nvgpu_inject LIBRARY DESTINATION ${INSTALL_PATH}/nvgpu)
