find_package(Eigen3 QUIET)

add_executable(bench_tensor
    bench_tensor.cpp
)

if(EIGEN3_FOUND)
    target_link_libraries(bench_tensor PRIVATE Eigen3::Eigen)
else()
    # Fallback for systems where Eigen is in standard include paths
    target_include_directories(bench_tensor SYSTEM PRIVATE
        /usr/include/eigen3
        /opt/homebrew/include/eigen3
    )
endif()

target_link_libraries(bench_tensor PRIVATE
    tensor_lib
    benchmark::benchmark_main
)

target_compile_options(bench_tensor PRIVATE -O3 -march=native)

