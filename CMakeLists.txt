cmake_minimum_required(VERSION 3.15)

project(scaueclib LANGUAGES CXX)

execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(scaueclib
    src/module.cpp
    src/online_cpa.cpp
    src/online_tvla.cpp
)

target_include_directories(scaueclib PRIVATE
    include
)

target_compile_features(scaueclib PRIVATE
    cxx_std_20
)

target_compile_options(scaueclib PRIVATE
    -O3
    -march=native
)



