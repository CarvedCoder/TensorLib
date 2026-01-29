enable_testing()

add_executable(test_tensor
    test_tensor.cpp
)

target_include_directories(test_tensor SYSTEM PRIVATE
    /opt/homebrew/include/eigen3
)

target_link_libraries(test_tensor PRIVATE
    tensor_lib
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_tensor)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_tensor PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

