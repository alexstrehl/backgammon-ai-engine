# Open Sage's engine library (bgbot_lib, the release's cpp/CMakeLists_cpu.txt
# with its -O3 -ffast-math -funroll-loops -mavx2 -mfma) and the harness,
# compiled as the release compiles its C API (mobile/CMakeLists.txt).
cmake_minimum_required(VERSION 3.18)
project(sage_bench LANGUAGES CXX)
set(BGSAGE_SRC "" CACHE PATH "bgsage release source")
set(BGBOT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(BGBOT_BUILD_APPS OFF CACHE BOOL "" FORCE)
include(${BGSAGE_SRC}/cpp/CMakeLists_cpu.txt)
add_library(sage_bench SHARED ${CMAKE_CURRENT_LIST_DIR}/sage_bench.cpp)
target_link_libraries(sage_bench PRIVATE bgbot_lib)
target_compile_definitions(sage_bench PRIVATE NOMINMAX)
