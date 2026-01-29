add_executable(bench_tensor
    bench_tensor.cpp
)

target_include_directories(bench_tensor SYSTEM PRIVATE
    /opt/homebrew/include/eigen3
)

target_link_libraries(bench_tensor PRIVATE
    tensor_lib
    benchmark::benchmark_main
)

target_compile_options(bench_tensor PRIVATE -O3 -march=native)

