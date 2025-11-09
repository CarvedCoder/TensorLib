cmake_minimum_required(VERSION 3.25)
project(minitensor VERSION 3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------- warnings = errors -----------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wsign-conversion )
elseif(MSVC)
    add_compile_options(/W4 /WX)
endif()

# ---------------- sanitizers (Debug only) ----------
string(APPEND CMAKE_CXX_FLAGS_DEBUG
        " -fsanitize=address,undefined -fno-omit-frame-pointer")

# ---------------- GoogleTest -----------------------
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# ---------------- library --------------------------
add_library(tensor_lib STATIC
        src/ops.cpp
        src/tensor.cpp
        src/autograd.cpp
)
target_include_directories(tensor_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------- test executable ------------------
enable_testing()
add_executable(tensor_test src/test_tensor.cpp)
target_link_libraries(tensor_test PRIVATE tensor_lib GTest::gtest GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(tensor_test)

# ---------------- optional bench -------------------
# add_executable(bench bench.cpp)
# target_link_libraries(bench PRIVATE tensor_lib)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE tensor_lib)