add_executable(bench_tensor
    bench_tensor.cpp
)

target_link_libraries(bench_tensor PRIVATE
    tensorlib
    benchmark::benchmark_main
    Eigen3::Eigen
)

target_compile_options(bench_tensor PRIVATE -O3 -march=native)

