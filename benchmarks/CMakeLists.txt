# =============================================================================
# benchmarks/CMakeLists.txt
#
# Drop this file into your existing benchmarks/ directory and replace the
# original add_executable block.  The rest of the root CMakeLists.txt is
# unchanged.
# =============================================================================

add_executable(bench_tensor
    bench_tensor.cpp
)

target_link_libraries(bench_tensor PRIVATE
    tensorlib
    benchmark::benchmark_main
    Eigen3::Eigen
)

# ─── Compiler flags ───────────────────────────────────────────────────────────
# Use C++23 to match the library.
target_compile_features(bench_tensor PRIVATE cxx_std_23)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_tensor PRIVATE
        # Optimisation
        -O3
        -march=native          # auto-detect SIMD width (AVX2/AVX-512/NEON)
        -funroll-loops
        -fno-math-errno        # allow vectorisation of libm calls

        # Strict aliasing (safe — we never alias float* with non-float*)
        -fstrict-aliasing

        # Suppress benchmark-induced warnings
        -Wno-unused-result
    )
elseif(MSVC)
    target_compile_options(bench_tensor PRIVATE
        /O2
        /arch:AVX2
        /fp:fast
        /GS-                   # disable buffer security check (perf)
    )
endif()

# ─── Linker flags ─────────────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT APPLE)
    # Link-time optimisation (optional but recommended for Release builds)
    # Uncomment if you want cross-TU inlining:
    # target_compile_options(bench_tensor PRIVATE -flto)
    # target_link_options(bench_tensor PRIVATE -flto)
endif()

# ─── Optional: perf-stat friendly binary ─────────────────────────────────────
# Keep frame pointers so perf/vtune can walk the call stack.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_tensor PRIVATE -fno-omit-frame-pointer)
endif()

# ─── CTest integration ────────────────────────────────────────────────────────
# Running the full benchmark suite as a CTest is too slow; register a quick
# smoke test that just lists the benchmark names.
enable_testing()
add_test(
    NAME bench_tensor_smoke
    COMMAND bench_tensor --benchmark_list_tests=true
)
