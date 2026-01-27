cmake_minimum_required(VERSION 3.20)
project(TensorLib VERSION 3 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()


# ---------------- GoogleTest -----------------------
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# ---------------- Google Benchmark -----------------
FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
)
# Disable benchmark's own tests
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)


# ---------------- library --------------------------
add_library(tensor_lib STATIC
        src/ops/ops.cpp
        src/tensor/tensor.cpp
        src/tensor/tensor_RNG.cpp
)
target_include_directories(tensor_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------- test executable ------------------
enable_testing()
add_executable(tensor_test src/tests/test_tensor.cpp)
target_include_directories(tensor_test SYSTEM PRIVATE /opt/homebrew/include/eigen3 ${gtest_SOURCE_DIR}/googletest/include)
target_link_libraries(tensor_test PRIVATE tensor_lib GTest::gtest GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(tensor_test)

# ---------------- benchmark executable ------------------
add_executable(tensor_bench
        src/tests/bench_tensor.cpp
)

target_include_directories(tensor_bench
        SYSTEM PRIVATE /opt/homebrew/include/eigen3
)

target_link_libraries(tensor_bench
        PRIVATE
        tensor_lib
        benchmark::benchmark_main
)

# main executable

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE tensor_lib)

# ---------------- warnings = errors -----------------

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(PROJECT_WARNINGS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wreorder
    )

    target_compile_options(tensor_lib PRIVATE ${PROJECT_WARNINGS} -Werror)
    target_compile_options(tensor_test PRIVATE ${PROJECT_WARNINGS} -Werror)
    target_compile_options(tensor_bench PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(tensor_lib PRIVATE /W4 /WX)
    target_compile_options(tensor_test PRIVATE /W4 /WX)
endif()


