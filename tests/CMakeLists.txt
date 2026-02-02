enable_testing()

find_package(Eigen3 QUIET)

add_executable(test_tensor
    test_tensor.cpp
)

if(EIGEN3_FOUND)
    target_link_libraries(test_tensor PRIVATE Eigen3::Eigen)
else()
    # Fallback for systems where Eigen is in standard include paths
    target_include_directories(test_tensor SYSTEM PRIVATE
        /usr/include/eigen3
        /opt/homebrew/include/eigen3
    )
endif()

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

