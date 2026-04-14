cmake_minimum_required(VERSION 3.15)
project(im2col_comparison LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Проверка поддержки AVX2 компилятором
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2)
if(NOT COMPILER_HAS_AVX2)
    message(FATAL_ERROR )
endif()

# Зависимости через FetchContent
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable gtest in benchmark" FORCE)
FetchContent_MakeAvailable(Catch2 benchmark)

# Исходники проекта
set(PROJECT_SOURCES src/tensor.cpp src/gemm.cpp src/conv_naive.cpp src/conv_im2col.cpp)

# 1. Основной исполняемый файл (ручной ввод параметров)
add_executable(conv_compare src/main.cpp ${PROJECT_SOURCES})
target_include_directories(conv_compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(conv_compare PRIVATE -mavx2 -O3 -Wall -Wextra)

# 2. Тесты (Catch2)
add_executable(test_conv test/test_conv.cpp ${PROJECT_SOURCES})
target_include_directories(test_conv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_conv PRIVATE Catch2::Catch2WithMain)
target_compile_options(test_conv PRIVATE -mavx2 -O3 -Wall -Wextra)

# 3. Бенчмарки (Google Benchmark)
add_executable(bench_conv benchmark/bench_conv.cpp ${PROJECT_SOURCES})
target_include_directories(bench_conv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_conv PRIVATE benchmark::benchmark)
target_compile_options(bench_conv PRIVATE -mavx2 -O3 -DNDEBUG)

# Интеграция с CTest 
include(CTest)
add_test(NAME ConvTests COMMAND test_conv)
