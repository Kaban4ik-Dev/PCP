cmake_minimum_required(VERSION 3.15)
project(cuda-lab1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA)

# Поиск пакетов
find_package(CUDAToolkit REQUIRED)

# Рекурсивно найти все с++ и cuda файлы в src и подпапках
file(GLOB_RECURSE SOURCE_FILES
    "src/*.cpp"
    "src/*.h"
    "src/*.cu"
    "src/*.cuh"
)

# Создание исполняемого файла
add_executable(${PROJECT_NAME} 
    ${SOURCE_FILES}
)

# Включаемые директории
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # Добавляем папку include
    ${CMAKE_CURRENT_SOURCE_DIR}/src      # Добавляем папку src
)

# Настройки компилятора для разных конфигураций
if(MSVC)
    # Настройки для MSVC
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Release>:/O2 /fp:fast>
        $<$<CONFIG:Debug>:/Od /Zi /RTC1>
    )
    
    # Настройка runtime библиотек
    set_target_properties(${PROJECT_NAME} PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
    
    # Отключение конкретных предупреждений C++
    target_compile_options(${PROJECT_NAME} PRIVATE
        /wd26495  # Отключение предупреждения о неинициализированных переменных
    )
else()
    # Настройки для других компиляторов (GCC, Clang)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:Debug>:-O0 -g3>
    )
endif()

# Выходные директории
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Отладочная информация
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG _DEBUG)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG _NDEBUG)
endif()

# Линковка
target_link_libraries(${PROJECT_NAME} PRIVATE 
    CUDA::cudart
)

# Отладочный вывод информации о настройках
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Generator: ${CMAKE_GENERATOR}")

# Предупреждение, если используется неправильная конфигурация
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(WARNING "No build type specified, defaulting to Release")
endif()