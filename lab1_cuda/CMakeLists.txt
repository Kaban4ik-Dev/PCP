cmake_minimum_required(VERSION 3.24)  # CUDA в MSVC лучше всего работает с 3.24+

project(lab1_cuda LANGUAGES CXX CUDA)

# --- —тандарты €зыка ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- јрхитектуры GPU ---
set(CMAKE_CUDA_ARCHITECTURES 75)

# --- ‘лаги ---
# --expt-relaxed-constexpr и т.п. добавл€й при необходимости
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/W3")

# --- —бор всех исходников из src/ ---
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

add_executable(${PROJECT_NAME} ${SOURCES})

# --- CUDA Runtime (cudart) ---
# STATIC или SHARED Ч по вкусу. SHARED требует cudart64_*.dll р€дом с exe.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

# --- ƒл€ MSVC: включаем /utf-8, чтобы русские строки в fprintf не ломались ---
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
    # NVCC тоже должен передать /utf-8 хосту:
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()