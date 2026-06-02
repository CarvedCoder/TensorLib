/**
 * =============================================================================
 * TensorLib vs Eigen — Professional Benchmark Suite
 * =============================================================================
 *
 * Design principles:
 *  - Every TensorLib benchmark has a 1:1 Eigen counterpart at the same size.
 *  - Data is initialised OUTSIDE the timed loop so allocation is never counted
 *    unless the benchmark is explicitly measuring allocation.
 *  - Identical random seeds across TensorLib / Eigen helpers guarantee the
 *    same floating-point inputs.
 *  - GFLOP/s is reported for every compute-bound benchmark (matmul, dot, etc.)
 *    using the conventional 2·M·N·K formula.
 *  - Bandwidth (GB/s) is reported for every bandwidth-bound benchmark via
 *    SetBytesProcessed().
 *  - DoNotOptimize() is used on every output; ClobberMemory() is inserted after
 *    in-place writes so the compiler cannot dead-code them.
 *  - Benchmarks are grouped by category; names follow the pattern
 *      BM_{Library}_{Category}[_{Variant}]
 *    so they can be filtered with --benchmark_filter.
 *  - C++23 is required (span, ranges, etc. already used in TensorLib).
 *
 * Build:  see CMakeLists_bench.txt (provided alongside this file)
 * Run:    ./bench_tensor --benchmark_counters_tabular=true
 *         ./bench_tensor --benchmark_filter="MatMul"
 *         ./bench_tensor --benchmark_out=results.json --benchmark_out_format=json
 * =============================================================================
 */

// ─── Standard library ────────────────────────────────────────────────────────
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <vector>

// ─── Third-party ─────────────────────────────────────────────────────────────
#include <Eigen/Dense>
#include <benchmark/benchmark.h>

// ─── TensorLib ───────────────────────────────────────────────────────────────
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>

// =============================================================================
// §0  SHARED HELPERS
// =============================================================================

// Fixed seed used everywhere so TensorLib and Eigen receive identical data.
static constexpr uint32_t kSeed = 0xDEAD'BEEF;

// ---------------------------------------------------------------------------
// TensorLib factory helpers
// ---------------------------------------------------------------------------

/// Fill a 2-D tensor with a constant value (allocation outside the hot loop).
[[nodiscard]] static Tensor tl_make_const(size_t r, size_t c, float v = 1.0f) {
    auto t = Tensor::createZeros({r, c});
    float* p = t.getMutableDataPtr();
    std::fill_n(p, r * c, v);
    return t;
}

/// Fill a 1-D tensor with a constant value.
[[nodiscard]] static Tensor tl_make_const1d(size_t n, float v = 1.0f) {
    auto t = Tensor::createZeros({n});
    float* p = t.getMutableDataPtr();
    std::fill_n(p, n, v);
    return t;
}

/// Fill a 2-D tensor with uniform random floats in [-1, 1].
[[nodiscard]] static Tensor tl_make_random(size_t r, size_t c, uint32_t seed = kSeed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto t = Tensor::createZeros({r, c});
    float* p = t.getMutableDataPtr();
    for (size_t i = 0; i < r * c; ++i)
        p[i] = dist(gen);
    return t;
}

/// Fill a 1-D tensor with uniform random floats in [-1, 1].
[[nodiscard]] static Tensor tl_make_random1d(size_t n, uint32_t seed = kSeed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto t = Tensor::createZeros({n});
    float* p = t.getMutableDataPtr();
    for (size_t i = 0; i < n; ++i)
        p[i] = dist(gen);
    return t;
}

// ---------------------------------------------------------------------------
// Eigen factory helpers   (row-major to match TensorLib's row-major layout)
// ---------------------------------------------------------------------------
using ERowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

[[nodiscard]] static ERowMat eg_make_const(size_t r, size_t c, float v = 1.0f) {
    return ERowMat::Constant(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c), v);
}

[[nodiscard]] static Eigen::VectorXf eg_make_const1d(size_t n, float v = 1.0f) {
    return Eigen::VectorXf::Constant(static_cast<Eigen::Index>(n), v);
}

[[nodiscard]] static ERowMat eg_make_random(size_t r, size_t c, uint32_t seed = kSeed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    ERowMat m(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
    for (size_t i = 0; i < r * c; ++i)
        m.data()[i] = dist(gen);
    return m;
}

[[nodiscard]] static Eigen::VectorXf eg_make_random1d(size_t n, uint32_t seed = kSeed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Eigen::VectorXf v(static_cast<Eigen::Index>(n));
    for (size_t i = 0; i < n; ++i)
        v.data()[i] = dist(gen);
    return v;
}

// ---------------------------------------------------------------------------
// GFLOP/s counter helpers
// ---------------------------------------------------------------------------
static void set_matmul_gflops(benchmark::State& st, double m, double k, double n) {
    // 2·M·K·N multiply-adds; divide by 1e9 so the reported value is in GFLOP/s
    const double gflops = 2.0 * m * k * n / 1e9;
    st.counters["GFLOP/s"] =
        benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
}

static void set_dot_gflops(benchmark::State& st, double n) {
    // n multiplications + n−1 additions ≈ 2n; divide by 1e9 for GFLOP/s
    st.counters["GFLOP/s"] =
        benchmark::Counter(2.0 * n / 1e9, benchmark::Counter::kIsIterationInvariantRate);
}

// =============================================================================
// §1  MEMORY ALLOCATION & INITIALIZATION
// =============================================================================

// ─── §1.1  Tensor allocation ─────────────────────────────────────────────────

static void BM_TL_Alloc_Zeros(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    for (auto _ : st) {
        auto t = Tensor::createZeros({N});
        benchmark::DoNotOptimize(t);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * sizeof(float)));
    st.SetLabel("createZeros 1-D");
}
BENCHMARK(BM_TL_Alloc_Zeros)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

static void BM_TL_Alloc_Ones(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    for (auto _ : st) {
        auto t = Tensor::createOnes({N});
        benchmark::DoNotOptimize(t);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * sizeof(float)));
    st.SetLabel("createOnes 1-D");
}
BENCHMARK(BM_TL_Alloc_Ones)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

static void BM_TL_Alloc_Matrix(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    for (auto _ : st) {
        auto t = Tensor::createZeros({N, N});
        benchmark::DoNotOptimize(t);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * N * sizeof(float)));
}
BENCHMARK(BM_TL_Alloc_Matrix)->RangeMultiplier(2)->Range(32, 1024)->Unit(benchmark::kMicrosecond);

// ─── §1.2  Eigen allocation (baseline) ───────────────────────────────────────

static void BM_EG_Alloc_Zeros(benchmark::State& st) {
    const auto N = static_cast<Eigen::Index>(st.range(0));
    for (auto _ : st) {
        ERowMat m = ERowMat::Zero(N, N);
        benchmark::DoNotOptimize(m);
    }
    st.SetBytesProcessed(
        static_cast<int64_t>(st.iterations() * static_cast<size_t>(N * N) * sizeof(float)));
}
BENCHMARK(BM_EG_Alloc_Zeros)->RangeMultiplier(2)->Range(32, 1024)->Unit(benchmark::kMicrosecond);

static void BM_Baseline_New_Delete(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    for (auto _ : st) {
        auto* p = new float[N](); // value-initialised (zero)
        benchmark::DoNotOptimize(p);
        delete[] p;
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * sizeof(float)));
    st.SetLabel("raw new[]/delete[]");
}
BENCHMARK(BM_Baseline_New_Delete)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// §2  MEMORY ACCESS PATTERNS
// =============================================================================

// ─── §2.1  Sequential read ───────────────────────────────────────────────────

static void BM_TL_Read_Sequential(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = tl_make_random1d(N);
    const float* p = t.getDataPtr();

    for (auto _ : st) {
        float acc = 0.0f;
        for (size_t i = 0; i < N; ++i)
            acc += p[i];
        benchmark::DoNotOptimize(acc);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * sizeof(float)));
}
BENCHMARK(BM_TL_Read_Sequential)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 24)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Read_Sequential(benchmark::State& st) {
    const auto N = static_cast<Eigen::Index>(st.range(0));
    Eigen::VectorXf v = eg_make_random1d(static_cast<size_t>(N));
    const float* p = v.data();

    for (auto _ : st) {
        float acc = 0.0f;
        for (Eigen::Index i = 0; i < N; ++i)
            acc += p[i];
        benchmark::DoNotOptimize(acc);
    }
    st.SetBytesProcessed(
        static_cast<int64_t>(st.iterations() * static_cast<size_t>(N) * sizeof(float)));
}
BENCHMARK(BM_EG_Read_Sequential)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 24)
    ->Unit(benchmark::kMicrosecond);

// ─── §2.2  Sequential write ──────────────────────────────────────────────────

static void BM_TL_Write_Sequential(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = Tensor::createZeros({N});
    float* p = t.getMutableDataPtr();

    for (auto _ : st) {
        for (size_t i = 0; i < N; ++i)
            p[i] = static_cast<float>(i) * 0.001f;
        benchmark::ClobberMemory();
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * sizeof(float)));
}
BENCHMARK(BM_TL_Write_Sequential)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 24)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Write_Sequential(benchmark::State& st) {
    const auto N = static_cast<Eigen::Index>(st.range(0));
    Eigen::VectorXf v = Eigen::VectorXf::Zero(N);
    float* p = v.data();

    for (auto _ : st) {
        for (Eigen::Index i = 0; i < N; ++i)
            p[i] = static_cast<float>(i) * 0.001f;
        benchmark::ClobberMemory();
    }
    st.SetBytesProcessed(
        static_cast<int64_t>(st.iterations() * static_cast<size_t>(N) * sizeof(float)));
}
BENCHMARK(BM_EG_Write_Sequential)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 24)
    ->Unit(benchmark::kMicrosecond);

// ─── §2.3  2-D row-major vs column-major traversal ───────────────────────────

static void BM_TL_Access_RowMajor(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = tl_make_random(N, N);
    const float* p = t.getDataPtr();

    for (auto _ : st) {
        float acc = 0.0f;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                acc += p[i * N + j]; // stride-1 (cache-friendly)
        benchmark::DoNotOptimize(acc);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * N * sizeof(float)));
    st.SetLabel("row-major (cache friendly)");
}
BENCHMARK(BM_TL_Access_RowMajor)->RangeMultiplier(2)->Range(64, 1024);

static void BM_TL_Access_ColMajor(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = tl_make_random(N, N);
    const float* p = t.getDataPtr();

    for (auto _ : st) {
        float acc = 0.0f;
        for (size_t j = 0; j < N; ++j)
            for (size_t i = 0; i < N; ++i)
                acc += p[i * N + j]; // stride-N (cache-unfriendly)
        benchmark::DoNotOptimize(acc);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * N * sizeof(float)));
    st.SetLabel("col-major (cache unfriendly)");
}
BENCHMARK(BM_TL_Access_ColMajor)->RangeMultiplier(2)->Range(64, 1024);

// ─── §2.4  memcpy baseline ───────────────────────────────────────────────────

static void BM_Baseline_Memcpy(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    std::vector<float> src(N, 1.0f), dst(N, 0.0f);

    for (auto _ : st) {
        std::memcpy(dst.data(), src.data(), N * sizeof(float));
        benchmark::ClobberMemory();
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * sizeof(float)));
}
BENCHMARK(BM_Baseline_Memcpy)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 26)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// §3  ELEMENT-WISE OPERATIONS
// =============================================================================
//
// NOTE: TensorLib's binaryKernel goes through computeBroadcast() which runs an
// N-dimensional multi-index loop even for the same-shape case.  These
// benchmarks expose that overhead clearly.

// ─── §3.1  Vector addition ───────────────────────────────────────────────────

static void BM_TL_EW_Add(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = tl_make_random1d(N, kSeed);
    auto b = tl_make_random1d(N, kSeed + 1);

    for (auto _ : st) {
        auto r = TensorOps::operator+(a, b);
        benchmark::DoNotOptimize(r);
    }
    // 2 reads + 1 write per element
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 3 * sizeof(float)));
}
BENCHMARK(BM_TL_EW_Add)->RangeMultiplier(4)->Range(1 << 10, 1 << 22)->Unit(benchmark::kMicrosecond);

static void BM_EG_EW_Add(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = eg_make_random1d(N, kSeed);
    auto b = eg_make_random1d(N, kSeed + 1);

    for (auto _ : st) {
        Eigen::VectorXf r = a + b;
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 3 * sizeof(float)));
}
BENCHMARK(BM_EG_EW_Add)->RangeMultiplier(4)->Range(1 << 10, 1 << 22)->Unit(benchmark::kMicrosecond);

// ─── §3.2  Vector subtraction ────────────────────────────────────────────────

static void BM_TL_EW_Sub(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = tl_make_random1d(N, kSeed);
    auto b = tl_make_random1d(N, kSeed + 1);

    for (auto _ : st) {
        auto r = TensorOps::operator-(a, b);
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 3 * sizeof(float)));
}
BENCHMARK(BM_TL_EW_Sub)->RangeMultiplier(4)->Range(1 << 10, 1 << 22)->Unit(benchmark::kMicrosecond);

static void BM_EG_EW_Sub(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = eg_make_random1d(N, kSeed);
    auto b = eg_make_random1d(N, kSeed + 1);

    for (auto _ : st) {
        Eigen::VectorXf r = a - b;
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 3 * sizeof(float)));
}
BENCHMARK(BM_EG_EW_Sub)->RangeMultiplier(4)->Range(1 << 10, 1 << 22)->Unit(benchmark::kMicrosecond);

// ─── §3.3  Coefficient-wise multiply ─────────────────────────────────────────

static void BM_TL_EW_Mul(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = tl_make_random1d(N, kSeed);
    auto b = tl_make_random1d(N, kSeed + 1);

    for (auto _ : st) {
        auto r = TensorOps::operator*(a, b);
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 3 * sizeof(float)));
}
BENCHMARK(BM_TL_EW_Mul)->RangeMultiplier(4)->Range(1 << 10, 1 << 22)->Unit(benchmark::kMicrosecond);

static void BM_EG_EW_Mul(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = eg_make_random1d(N, kSeed);
    auto b = eg_make_random1d(N, kSeed + 1);

    for (auto _ : st) {
        Eigen::VectorXf r = a.cwiseProduct(b);
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 3 * sizeof(float)));
}
BENCHMARK(BM_EG_EW_Mul)->RangeMultiplier(4)->Range(1 << 10, 1 << 22)->Unit(benchmark::kMicrosecond);

// ─── §3.4  Element-wise divide ────────────────────────────────────────────────

static void BM_TL_EW_Div(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = tl_make_random1d(N, kSeed);
    auto b = tl_make_const1d(N, 2.0f); // avoid div-by-zero

    for (auto _ : st) {
        auto r = TensorOps::operator/(a, b);
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 3 * sizeof(float)));
}
BENCHMARK(BM_TL_EW_Div)->RangeMultiplier(4)->Range(1 << 10, 1 << 22)->Unit(benchmark::kMicrosecond);

static void BM_EG_EW_Div(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = eg_make_random1d(N, kSeed);
    Eigen::VectorXf b = eg_make_const1d(N, 2.0f);

    for (auto _ : st) {
        Eigen::VectorXf r = a.cwiseQuotient(b);
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 3 * sizeof(float)));
}
BENCHMARK(BM_EG_EW_Div)->RangeMultiplier(4)->Range(1 << 10, 1 << 22)->Unit(benchmark::kMicrosecond);

// ─── §3.5  Scalar multiply ───────────────────────────────────────────────────

static void BM_TL_EW_ScalarMul(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = tl_make_random1d(N, kSeed);

    for (auto _ : st) {
        auto r = TensorOps::operator*(a, 3.14159f);
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
}
BENCHMARK(BM_TL_EW_ScalarMul)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_EW_ScalarMul(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = eg_make_random1d(N, kSeed);

    for (auto _ : st) {
        Eigen::VectorXf r = a * 3.14159f;
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
}
BENCHMARK(BM_EG_EW_ScalarMul)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

// ─── §3.6  Broadcasting: 2-D matrix + 1-D vector (TensorLib supports this) ──

static void BM_TL_EW_Broadcast_MatVec(benchmark::State& st) {
    const size_t R = static_cast<size_t>(st.range(0));
    const size_t C = 256;
    auto M = tl_make_random(R, C, kSeed);
    auto v = tl_make_random1d(C, kSeed + 1); // broadcast over rows

    for (auto _ : st) {
        auto r = TensorOps::operator+(M, v);
        benchmark::DoNotOptimize(r);
    }
    // Output R*C writes + R*C reads from M + C reads from v (broadcast)
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * (R * C * 2 + C) * sizeof(float)));
    st.SetLabel("(R x 256) + (256,) broadcast");
}
BENCHMARK(BM_TL_EW_Broadcast_MatVec)->Arg(1)->Arg(32)->Arg(128)->Arg(512);

static void BM_EG_EW_Broadcast_MatVec(benchmark::State& st) {
    const auto R = static_cast<Eigen::Index>(st.range(0));
    const Eigen::Index C = 256;
    auto M = eg_make_random(static_cast<size_t>(R), 256, kSeed);
    Eigen::RowVectorXf v = eg_make_random1d(256, kSeed + 1).transpose();

    for (auto _ : st) {
        ERowMat r = M.rowwise() + v;
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(
        static_cast<int64_t>(st.iterations() * static_cast<size_t>(R * C * 2 + C) * sizeof(float)));
    st.SetLabel("(R x 256) + (256,) broadcast");
}
BENCHMARK(BM_EG_EW_Broadcast_MatVec)->Arg(1)->Arg(32)->Arg(128)->Arg(512);

// =============================================================================
// §4  DOT PRODUCT & REDUCTIONS
// =============================================================================

// TensorLib has no built-in dot/sum, so we implement the reference manually
// on the raw pointer to give a fair apples-to-apples comparison.

static void BM_TL_Dot_Manual(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = tl_make_random1d(N, kSeed);
    auto b = tl_make_random1d(N, kSeed + 1);
    const float* pa = a.getDataPtr();
    const float* pb = b.getDataPtr();

    for (auto _ : st) {
        float acc = 0.0f;
        for (size_t i = 0; i < N; ++i)
            acc += pa[i] * pb[i];
        benchmark::DoNotOptimize(acc);
    }
    set_dot_gflops(st, static_cast<double>(N));
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
}
BENCHMARK(BM_TL_Dot_Manual)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Dot(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = eg_make_random1d(N, kSeed);
    auto b = eg_make_random1d(N, kSeed + 1);

    for (auto _ : st) {
        float r = a.dot(b);
        benchmark::DoNotOptimize(r);
    }
    set_dot_gflops(st, static_cast<double>(N));
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
}
BENCHMARK(BM_EG_Dot)->RangeMultiplier(4)->Range(1 << 10, 1 << 22)->Unit(benchmark::kMicrosecond);

// ─── §4.2  L2-norm ───────────────────────────────────────────────────────────

static void BM_TL_Norm_Manual(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = tl_make_random1d(N, kSeed);
    const float* pa = a.getDataPtr();

    for (auto _ : st) {
        float acc = 0.0f;
        for (size_t i = 0; i < N; ++i)
            acc += pa[i] * pa[i];
        float result = std::sqrt(acc);
        benchmark::DoNotOptimize(result);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * sizeof(float)));
}
BENCHMARK(BM_TL_Norm_Manual)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Norm(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = eg_make_random1d(N, kSeed);

    for (auto _ : st) {
        float r = a.norm();
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * sizeof(float)));
}
BENCHMARK(BM_EG_Norm)->RangeMultiplier(4)->Range(1 << 10, 1 << 22)->Unit(benchmark::kMicrosecond);

// ─── §4.3  SSE / MSE loss (TensorLib unique operation) ───────────────────────

static void BM_TL_Loss_MSE(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto pred = tl_make_random1d(N, kSeed);
    auto truth = tl_make_random1d(N, kSeed + 1);

    for (auto _ : st) {
        float r = TensorOps::calcCost(pred, truth, LossType::MSE);
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
    st.SetLabel("MSE loss");
}
BENCHMARK(BM_TL_Loss_MSE)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Loss_MSE(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto pred = eg_make_random1d(N, kSeed);
    auto truth = eg_make_random1d(N, kSeed + 1);

    for (auto _ : st) {
        float r = (pred - truth).squaredNorm() / static_cast<float>(N);
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
    st.SetLabel("MSE loss");
}
BENCHMARK(BM_EG_Loss_MSE)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// §5  TRANSPOSE
// =============================================================================

static void BM_TL_Transpose_Square(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = tl_make_random(N, N);

    for (auto _ : st) {
        auto tT = TensorOps::transpose2D(t);
        benchmark::DoNotOptimize(tT);
    }
    // Read N² floats, write N² floats
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * N * 2 * sizeof(float)));
}
BENCHMARK(BM_TL_Transpose_Square)
    ->RangeMultiplier(2)
    ->Range(32, 2048)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Transpose_Square(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto m = eg_make_random(N, N);

    for (auto _ : st) {
        ERowMat mT = m.transpose();
        benchmark::DoNotOptimize(mT);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * N * 2 * sizeof(float)));
}
BENCHMARK(BM_EG_Transpose_Square)
    ->RangeMultiplier(2)
    ->Range(32, 2048)
    ->Unit(benchmark::kMicrosecond);

// ─── §5.2  Rectangular transpose ─────────────────────────────────────────────

static void BM_TL_Transpose_Rect(benchmark::State& st) {
    const size_t R = static_cast<size_t>(st.range(0));
    const size_t C = static_cast<size_t>(st.range(1));
    auto t = tl_make_random(R, C);

    for (auto _ : st) {
        auto tT = TensorOps::transpose2D(t);
        benchmark::DoNotOptimize(tT);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * R * C * 2 * sizeof(float)));
}
BENCHMARK(BM_TL_Transpose_Rect)
    ->Args({128, 1024})
    ->Args({1024, 128})
    ->Args({512, 2048})
    ->Args({2048, 512})
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Transpose_Rect(benchmark::State& st) {
    const size_t R = static_cast<size_t>(st.range(0));
    const size_t C = static_cast<size_t>(st.range(1));
    auto m = eg_make_random(R, C);

    for (auto _ : st) {
        ERowMat mT = m.transpose();
        benchmark::DoNotOptimize(mT);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * R * C * 2 * sizeof(float)));
}
BENCHMARK(BM_EG_Transpose_Rect)
    ->Args({128, 1024})
    ->Args({1024, 128})
    ->Args({512, 2048})
    ->Args({2048, 512})
    ->Unit(benchmark::kMicrosecond);

// ─── §5.3  Double-transpose identity check (perf only) ───────────────────────

static void BM_TL_Transpose_Double(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = tl_make_random(N, N);

    for (auto _ : st) {
        auto t1 = TensorOps::transpose2D(t);
        auto t2 = TensorOps::transpose2D(t1);
        benchmark::DoNotOptimize(t2);
    }
}
BENCHMARK(BM_TL_Transpose_Double)->RangeMultiplier(2)->Range(64, 512);

// =============================================================================
// §6  MATRIX MULTIPLICATION
// =============================================================================
//
// TensorLib matmul uses ikj loop ordering — better than naive ijk but has no
// tiling, no SIMD, and calls createZeros (heap alloc) every invocation.
// Eigen uses a highly optimised BLAS-like kernel (Householder-blocked GEMM).
// These benchmarks make that gap visible at every size regime.

// ─── §6.1  Square — tiny (fits fully in L1) ───────────────────────────────────

static void BM_TL_MatMul_Tiny(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = tl_make_random(N, N, kSeed);
    auto B = tl_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_TL_MatMul_Tiny)->DenseRange(2, 16, 2)->Unit(benchmark::kNanosecond);

static void BM_EG_MatMul_Tiny(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = eg_make_random(N, N, kSeed);
    auto B = eg_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        ERowMat C = A * B;
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_EG_MatMul_Tiny)->DenseRange(2, 16, 2)->Unit(benchmark::kNanosecond);

// ─── §6.2  Square — small (L1–L2) ────────────────────────────────────────────

static void BM_TL_MatMul_Small(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = tl_make_random(N, N, kSeed);
    auto B = tl_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_TL_MatMul_Small)->RangeMultiplier(2)->Range(16, 128)->Unit(benchmark::kMicrosecond);

static void BM_EG_MatMul_Small(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = eg_make_random(N, N, kSeed);
    auto B = eg_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        ERowMat C = A * B;
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_EG_MatMul_Small)->RangeMultiplier(2)->Range(16, 128)->Unit(benchmark::kMicrosecond);

// ─── §6.3  Square — medium (L2–L3) ───────────────────────────────────────────

static void BM_TL_MatMul_Medium(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = tl_make_random(N, N, kSeed);
    auto B = tl_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_TL_MatMul_Medium)
    ->Arg(128)
    ->Arg(192)
    ->Arg(256)
    ->Arg(384)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond);

static void BM_EG_MatMul_Medium(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = eg_make_random(N, N, kSeed);
    auto B = eg_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        ERowMat C = A * B;
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_EG_MatMul_Medium)
    ->Arg(128)
    ->Arg(192)
    ->Arg(256)
    ->Arg(384)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond);

// ─── §6.4  Square — large (exceeds L3) ───────────────────────────────────────

static void BM_TL_MatMul_Large(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = tl_make_random(N, N, kSeed);
    auto B = tl_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_TL_MatMul_Large)->Arg(512)->Arg(768)->Arg(1024)->Unit(benchmark::kMillisecond);

static void BM_EG_MatMul_Large(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = eg_make_random(N, N, kSeed);
    auto B = eg_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        ERowMat C = A * B;
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_EG_MatMul_Large)->Arg(512)->Arg(768)->Arg(1024)->Unit(benchmark::kMillisecond);

// ─── §6.5  Power-of-2 vs non-power-of-2 ─────────────────────────────────────

static void BM_TL_MatMul_Pow2(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = tl_make_random(N, N, kSeed);
    auto B = tl_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_TL_MatMul_Pow2)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

static void BM_EG_MatMul_Pow2(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = eg_make_random(N, N, kSeed);
    auto B = eg_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        ERowMat C = A * B;
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_EG_MatMul_Pow2)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

static void BM_TL_MatMul_NonPow2(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = tl_make_random(N, N, kSeed);
    auto B = tl_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_TL_MatMul_NonPow2)->Arg(63)->Arg(127)->Arg(255)->Arg(511);

static void BM_EG_MatMul_NonPow2(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = eg_make_random(N, N, kSeed);
    auto B = eg_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        ERowMat C = A * B;
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_EG_MatMul_NonPow2)->Arg(63)->Arg(127)->Arg(255)->Arg(511);

// ─── §6.6  Rectangular — ML-style (batch × features) ────────────────────────

struct NNShape {
    size_t M, K, N;
};
static constexpr std::array<NNShape, 7> kNNShapes = {{
    {1, 512, 512},    // single-sample inference
    {32, 512, 512},   // small batch
    {128, 512, 512},  // medium batch
    {32, 784, 128},   // MNIST-like encoder
    {64, 2048, 1024}, // wide layer
    {256, 512, 256},  // ResNet-like
    {128, 64, 1},     // matrix-vector (output layer)
}};

static void BM_TL_MatMul_NeuralNet(benchmark::State& st) {
    const size_t M = static_cast<size_t>(st.range(0));
    const size_t K = static_cast<size_t>(st.range(1));
    const size_t N = static_cast<size_t>(st.range(2));
    auto A = tl_make_random(M, K, kSeed);
    auto B = tl_make_random(K, N, kSeed + 1);

    for (auto _ : st) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, M, K, N);
}
BENCHMARK(BM_TL_MatMul_NeuralNet)
    ->Args({1, 512, 512})
    ->Args({32, 512, 512})
    ->Args({128, 512, 512})
    ->Args({32, 784, 128})
    ->Args({64, 2048, 1024})
    ->Args({256, 512, 256})
    ->Args({128, 64, 1})
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_MatMul_NeuralNet(benchmark::State& st) {
    const size_t M = static_cast<size_t>(st.range(0));
    const size_t K = static_cast<size_t>(st.range(1));
    const size_t N = static_cast<size_t>(st.range(2));
    auto A = eg_make_random(M, K, kSeed);
    auto B = eg_make_random(K, N, kSeed + 1);

    for (auto _ : st) {
        ERowMat C = A * B;
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, M, K, N);
}
BENCHMARK(BM_EG_MatMul_NeuralNet)
    ->Args({1, 512, 512})
    ->Args({32, 512, 512})
    ->Args({128, 512, 512})
    ->Args({32, 784, 128})
    ->Args({64, 2048, 1024})
    ->Args({256, 512, 256})
    ->Args({128, 64, 1})
    ->Unit(benchmark::kMicrosecond);

// ─── §6.7  Matrix-vector product ─────────────────────────────────────────────

static void BM_TL_MatMul_MatVec(benchmark::State& st) {
    const size_t M = static_cast<size_t>(st.range(0));
    const size_t K = static_cast<size_t>(st.range(1));
    auto A = tl_make_random(M, K, kSeed);
    auto x = tl_make_random(K, 1, kSeed + 1);

    for (auto _ : st) {
        auto y = TensorOps::matmul(A, x);
        benchmark::DoNotOptimize(y);
    }
    set_matmul_gflops(st, M, K, 1);
}
BENCHMARK(BM_TL_MatMul_MatVec)
    ->Args({64, 64})
    ->Args({128, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_MatMul_MatVec(benchmark::State& st) {
    const size_t M = static_cast<size_t>(st.range(0));
    const size_t K = static_cast<size_t>(st.range(1));
    auto A = eg_make_random(M, K, kSeed);
    auto x = eg_make_random1d(K, kSeed + 1);

    for (auto _ : st) {
        Eigen::VectorXf y = A * x;
        benchmark::DoNotOptimize(y);
    }
    set_matmul_gflops(st, M, K, 1);
}
BENCHMARK(BM_EG_MatMul_MatVec)
    ->Args({64, 64})
    ->Args({128, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// §7  CHAINED OPERATIONS (expression / fusion)
// =============================================================================
//
// TensorLib materialises a new tensor for each binary op, so a chain of k ops
// allocates k intermediate buffers.  Eigen uses lazy expression templates and
// fuses the chain into a single pass.  This section makes that difference
// visible.

static void BM_TL_Chain_AddMulSub(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = tl_make_random1d(N, kSeed);
    auto b = tl_make_random1d(N, kSeed + 1);
    auto c = tl_make_random1d(N, kSeed + 2);
    auto d = tl_make_random1d(N, kSeed + 3);

    // (a + b) * c - d  →  3 allocations, 3 passes
    for (auto _ : st) {
        auto t1 = TensorOps::operator+(a, b);
        auto t2 = TensorOps::operator*(t1, c);
        auto t3 = TensorOps::operator-(t2, d);
        benchmark::DoNotOptimize(t3);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 7 * sizeof(float)));
    st.SetLabel("(a+b)*c-d  [3 allocs, 3 passes]");
}
BENCHMARK(BM_TL_Chain_AddMulSub)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Chain_AddMulSub(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = eg_make_random1d(N, kSeed);
    auto b = eg_make_random1d(N, kSeed + 1);
    auto c = eg_make_random1d(N, kSeed + 2);
    auto d = eg_make_random1d(N, kSeed + 3);

    // Eigen evaluates lazily in a single pass (expression templates)
    for (auto _ : st) {
        Eigen::VectorXf r = (a + b).cwiseProduct(c) - d;
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 7 * sizeof(float)));
    st.SetLabel("(a+b)*c-d  [1 alloc, 1 pass via expr templates]");
}
BENCHMARK(BM_EG_Chain_AddMulSub)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

// Longer chain: a + b + c + d  (4 terms)
static void BM_TL_Chain_Sum4(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = tl_make_random1d(N, kSeed);
    auto b = tl_make_random1d(N, kSeed + 1);
    auto c = tl_make_random1d(N, kSeed + 2);
    auto d = tl_make_random1d(N, kSeed + 3);

    for (auto _ : st) {
        auto t1 = TensorOps::operator+(a, b);
        auto t2 = TensorOps::operator+(t1, c);
        auto t3 = TensorOps::operator+(t2, d);
        benchmark::DoNotOptimize(t3);
    }
    st.SetLabel("a+b+c+d  [3 allocs]");
}
BENCHMARK(BM_TL_Chain_Sum4)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Chain_Sum4(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto a = eg_make_random1d(N, kSeed);
    auto b = eg_make_random1d(N, kSeed + 1);
    auto c = eg_make_random1d(N, kSeed + 2);
    auto d = eg_make_random1d(N, kSeed + 3);

    for (auto _ : st) {
        Eigen::VectorXf r = a + b + c + d;
        benchmark::DoNotOptimize(r);
    }
    st.SetLabel("a+b+c+d  [1 alloc, 1 pass]");
}
BENCHMARK(BM_EG_Chain_Sum4)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

// Chained matmul: A * B * C
static void BM_TL_Chain_MatMul3(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = tl_make_random(N, N, kSeed);
    auto B = tl_make_random(N, N, kSeed + 1);
    auto C = tl_make_random(N, N, kSeed + 2);

    for (auto _ : st) {
        auto AB = TensorOps::matmul(A, B);
        auto ABC = TensorOps::matmul(AB, C);
        benchmark::DoNotOptimize(ABC);
    }
    set_matmul_gflops(st, N, N, N); // approximate (two equal-size matmuls)
}
BENCHMARK(BM_TL_Chain_MatMul3)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Unit(benchmark::kMicrosecond);

static void BM_EG_Chain_MatMul3(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = eg_make_random(N, N, kSeed);
    auto B = eg_make_random(N, N, kSeed + 1);
    auto C = eg_make_random(N, N, kSeed + 2);

    for (auto _ : st) {
        ERowMat ABC = (A * B) * C;
        benchmark::DoNotOptimize(ABC);
    }
    set_matmul_gflops(st, N, N, N);
}
BENCHMARK(BM_EG_Chain_MatMul3)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Unit(benchmark::kMicrosecond);

// =============================================================================
// §8  REALISTIC ML WORKLOADS
// =============================================================================

// ─── §8.1  Linear regression gradient step ───────────────────────────────────
//
// Simulates a single SGD update:
//   y_pred  = X * w + b
//   loss    = MSE(y_pred, y_true)
//   grad_w  = X^T * (y_pred - y_true) / n
//   w      -= lr * grad_w

static void BM_TL_LinearRegression_Step(benchmark::State& st) {
    const size_t B = static_cast<size_t>(st.range(0)); // batch size
    const size_t F = 128;                              // features

    auto X = tl_make_random(B, F, kSeed);
    auto w = tl_make_random(F, 1, kSeed + 1);
    auto y = tl_make_random(B, 1, kSeed + 2);
    auto b_t = tl_make_const(B, 1, 0.0f); // bias placeholder

    for (auto _ : st) {
        // Forward
        auto y_pred = TensorOps::matmul(X, w);

        // Error
        auto err = TensorOps::operator-(y_pred, y);

        // Gradient w.r.t. w:  X^T @ err / B
        auto X_T = TensorOps::transpose2D(X);
        auto grad_w = TensorOps::matmul(X_T, err);
        auto scaled = TensorOps::operator*(grad_w, 1.0f / static_cast<float>(B));

        benchmark::DoNotOptimize(scaled);
    }

    double gflops = (2.0 * B * F * 1    // matmul (forward)
                     + B                // subtraction
                     + 2.0 * F * B * 1) // matmul (backward)
                    / 1e9;
    st.counters["GFLOP/s"] =
        benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
    st.SetLabel("linear regression SGD step");
}
BENCHMARK(BM_TL_LinearRegression_Step)->Arg(32)->Arg(128)->Arg(512)->Unit(benchmark::kMicrosecond);

static void BM_EG_LinearRegression_Step(benchmark::State& st) {
    const auto B = static_cast<Eigen::Index>(st.range(0));
    const Eigen::Index F = 128;

    auto X = eg_make_random(static_cast<size_t>(B), static_cast<size_t>(F), kSeed);
    Eigen::VectorXf w = eg_make_random1d(static_cast<size_t>(F), kSeed + 1);
    Eigen::VectorXf y = eg_make_random1d(static_cast<size_t>(B), kSeed + 2);

    for (auto _ : st) {
        Eigen::VectorXf y_pred = X * w;
        Eigen::VectorXf err = y_pred - y;
        Eigen::VectorXf grad_w = (X.transpose() * err) / static_cast<float>(B);
        benchmark::DoNotOptimize(grad_w);
    }

    double gflops =
        (2.0 * static_cast<double>(B) * static_cast<double>(F) + static_cast<double>(B) +
         2.0 * static_cast<double>(F) * static_cast<double>(B)) /
        1e9;
    st.counters["GFLOP/s"] =
        benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
    st.SetLabel("linear regression SGD step");
}
BENCHMARK(BM_EG_LinearRegression_Step)->Arg(32)->Arg(128)->Arg(512)->Unit(benchmark::kMicrosecond);

// ─── §8.2  Three-layer MLP forward pass (MNIST-like) ─────────────────────────

static void BM_TL_MLP_Forward(benchmark::State& st) {
    const size_t B = static_cast<size_t>(st.range(0));

    auto input = tl_make_random(B, 784, kSeed);
    auto w1 = tl_make_random(784, 256, kSeed + 1);
    auto w2 = tl_make_random(256, 128, kSeed + 2);
    auto w3 = tl_make_random(128, 10, kSeed + 3);

    for (auto _ : st) {
        auto h1 = TensorOps::matmul(input, w1);
        auto h2 = TensorOps::matmul(h1, w2);
        auto output = TensorOps::matmul(h2, w3);
        benchmark::DoNotOptimize(output);
    }

    double gflops = 2.0 * static_cast<double>(B) * (784.0 * 256 + 256.0 * 128 + 128.0 * 10) / 1e9;
    st.counters["GFLOP/s"] =
        benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
    st.SetLabel("3-layer MLP (784→256→128→10)");
}
BENCHMARK(BM_TL_MLP_Forward)->Arg(1)->Arg(32)->Arg(128)->Arg(256)->Unit(benchmark::kMicrosecond);

static void BM_EG_MLP_Forward(benchmark::State& st) {
    const size_t B = static_cast<size_t>(st.range(0));

    auto input = eg_make_random(B, 784, kSeed);
    auto w1 = eg_make_random(784, 256, kSeed + 1);
    auto w2 = eg_make_random(256, 128, kSeed + 2);
    auto w3 = eg_make_random(128, 10, kSeed + 3);

    for (auto _ : st) {
        ERowMat output = ((input * w1) * w2) * w3;
        benchmark::DoNotOptimize(output);
    }

    double gflops = 2.0 * static_cast<double>(B) * (784.0 * 256 + 256.0 * 128 + 128.0 * 10) / 1e9;
    st.counters["GFLOP/s"] =
        benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
    st.SetLabel("3-layer MLP (784→256→128→10)");
}
BENCHMARK(BM_EG_MLP_Forward)->Arg(1)->Arg(32)->Arg(128)->Arg(256)->Unit(benchmark::kMicrosecond);

// ─── §8.3  Transformer scaled dot-product attention ──────────────────────────
//
// scores   = Q @ K^T / sqrt(d_k)      [seq×seq]
// output   = scores @ V               [seq×d_model]

static void BM_TL_Attention(benchmark::State& st) {
    const size_t S = static_cast<size_t>(st.range(0));
    const size_t D = 64; // d_k = d_v per head

    auto Q = tl_make_random(S, D, kSeed);
    auto K = tl_make_random(S, D, kSeed + 1);
    auto V = tl_make_random(S, D, kSeed + 2);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (auto _ : st) {
        auto K_T = TensorOps::transpose2D(K);
        auto scores = TensorOps::matmul(Q, K_T);
        auto scaled = TensorOps::operator*(scores, scale);
        // softmax skipped (not implemented in TensorLib)
        auto output = TensorOps::matmul(scaled, V);
        benchmark::DoNotOptimize(output);
    }

    // Q@K^T: 2*S*S*D,  scaled@V: 2*S*S*D
    double gflops = 4.0 * S * S * D / 1e9;
    st.counters["GFLOP/s"] =
        benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
    st.SetLabel("scaled dot-product attention (d_k=64)");
}
BENCHMARK(BM_TL_Attention)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Attention(benchmark::State& st) {
    const size_t S = static_cast<size_t>(st.range(0));
    const size_t D = 64;

    auto Q = eg_make_random(S, D, kSeed);
    auto K = eg_make_random(S, D, kSeed + 1);
    auto V = eg_make_random(S, D, kSeed + 2);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (auto _ : st) {
        ERowMat scores = (Q * K.transpose()) * scale;
        ERowMat output = scores * V;
        benchmark::DoNotOptimize(output);
    }

    double gflops = 4.0 * S * S * D / 1e9;
    st.counters["GFLOP/s"] =
        benchmark::Counter(gflops, benchmark::Counter::kIsIterationInvariantRate);
    st.SetLabel("scaled dot-product attention (d_k=64)");
}
BENCHMARK(BM_EG_Attention)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Unit(benchmark::kMicrosecond);

// ─── §8.4  Min-Max scaler (TensorLib unique op via Scaler class) ─────────────

static void BM_TL_MinMaxScaler(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = tl_make_random1d(N, kSeed);
    const float* p = t.getDataPtr();
    auto result = Tensor::createZeros({N});
    float* out = result.getMutableDataPtr();

    for (auto _ : st) {
        // Replicate the Scaler::minMaxScaler logic on raw pointers
        float mn = *std::min_element(p, p + N);
        float mx = *std::max_element(p, p + N);
        float range = mx - mn;
        for (size_t i = 0; i < N; ++i)
            out[i] = (p[i] - mn) / range;
        benchmark::DoNotOptimize(out[0]);
        benchmark::ClobberMemory();
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
    st.SetLabel("minmax scaler");
}
BENCHMARK(BM_TL_MinMaxScaler)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_MinMaxScaler(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto v = eg_make_random1d(N, kSeed);

    for (auto _ : st) {
        float mn = v.minCoeff();
        float mx = v.maxCoeff();
        Eigen::VectorXf r = (v.array() - mn) / (mx - mn);
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
    st.SetLabel("minmax scaler");
}
BENCHMARK(BM_EG_MinMaxScaler)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// §9  ACTIVATION FUNCTIONS
// =============================================================================
//
// TensorLib exposes sigmoid/relu/leakyRelu/tanh as scalar functions.
// We benchmark them applied over a vector via a raw pointer loop.
// Eigen applies them elementwise via .unaryExpr() or built-in Array methods.

static void BM_TL_Activation_Sigmoid(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = tl_make_random1d(N, kSeed);
    const float* in = t.getDataPtr();
    auto out_t = Tensor::createZeros({N});
    float* out = out_t.getMutableDataPtr();

    for (auto _ : st) {
        for (size_t i = 0; i < N; ++i)
            out[i] = TensorOps::sigmoid(in[i]);
        benchmark::ClobberMemory();
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
    st.SetLabel("sigmoid (scalar loop)");
}
BENCHMARK(BM_TL_Activation_Sigmoid)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Activation_Sigmoid(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto v = eg_make_random1d(N, kSeed);

    for (auto _ : st) {
        Eigen::VectorXf r = (1.0f / (1.0f + (-v.array()).exp())).matrix();
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
    st.SetLabel("sigmoid (Eigen::Array vectorised)");
}
BENCHMARK(BM_EG_Activation_Sigmoid)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_TL_Activation_ReLU(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = tl_make_random1d(N, kSeed);
    const float* in = t.getDataPtr();
    auto out_t = Tensor::createZeros({N});
    float* out = out_t.getMutableDataPtr();

    for (auto _ : st) {
        for (size_t i = 0; i < N; ++i)
            out[i] = TensorOps::relu(in[i]);
        benchmark::ClobberMemory();
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
    st.SetLabel("relu (scalar loop)");
}
BENCHMARK(BM_TL_Activation_ReLU)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Activation_ReLU(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto v = eg_make_random1d(N, kSeed);

    for (auto _ : st) {
        Eigen::VectorXf r = v.cwiseMax(0.0f);
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
    st.SetLabel("relu (Eigen cwiseMax)");
}
BENCHMARK(BM_EG_Activation_ReLU)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_TL_Activation_Tanh(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = tl_make_random1d(N, kSeed);
    const float* in = t.getDataPtr();
    auto out_t = Tensor::createZeros({N});
    float* out = out_t.getMutableDataPtr();

    for (auto _ : st) {
        for (size_t i = 0; i < N; ++i)
            out[i] = TensorOps::m_tanh(in[i]);
        benchmark::ClobberMemory();
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
    st.SetLabel("tanh (scalar loop)");
}
BENCHMARK(BM_TL_Activation_Tanh)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_EG_Activation_Tanh(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto v = eg_make_random1d(N, kSeed);

    for (auto _ : st) {
        Eigen::VectorXf r = v.array().tanh().matrix();
        benchmark::DoNotOptimize(r);
    }
    st.SetBytesProcessed(static_cast<int64_t>(st.iterations() * N * 2 * sizeof(float)));
    st.SetLabel("tanh (Eigen::Array vectorised)");
}
BENCHMARK(BM_EG_Activation_Tanh)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// §10  CACHE BEHAVIOUR
// =============================================================================

// ─── §10.1  Cache-cold matmul (flush with large irrelevant allocation) ────────

static void BM_TL_MatMul_CacheCold(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = tl_make_random(N, N, kSeed);
    auto B = tl_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        benchmark::ClobberMemory(); // attempt to evict caches
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
    st.SetLabel("cache cold (ClobberMemory before)");
}
BENCHMARK(BM_TL_MatMul_CacheCold)->Arg(256)->Arg(512)->Arg(1024)->Unit(benchmark::kMillisecond);

static void BM_EG_MatMul_CacheCold(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = eg_make_random(N, N, kSeed);
    auto B = eg_make_random(N, N, kSeed + 1);

    for (auto _ : st) {
        benchmark::ClobberMemory();
        ERowMat C = A * B;
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
    st.SetLabel("cache cold (ClobberMemory before)");
}
BENCHMARK(BM_EG_MatMul_CacheCold)->Arg(256)->Arg(512)->Arg(1024)->Unit(benchmark::kMillisecond);

// ─── §10.2  Repeated same matmul (warm cache) ────────────────────────────────

static void BM_TL_MatMul_CacheWarm(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = tl_make_random(N, N, kSeed);
    auto B = tl_make_random(N, N, kSeed + 1);
    // Warm-up run outside the measurement loop
    {
        auto tmp = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(tmp);
    }

    for (auto _ : st) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
    st.SetLabel("cache warm (pre-warmed A, B)");
}
BENCHMARK(BM_TL_MatMul_CacheWarm)->Arg(128)->Arg(256)->Arg(512)->Unit(benchmark::kMicrosecond);

static void BM_EG_MatMul_CacheWarm(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto A = eg_make_random(N, N, kSeed);
    auto B = eg_make_random(N, N, kSeed + 1);
    {
        ERowMat tmp = A * B;
        benchmark::DoNotOptimize(tmp);
    }

    for (auto _ : st) {
        ERowMat C = A * B;
        benchmark::DoNotOptimize(C);
    }
    set_matmul_gflops(st, N, N, N);
    st.SetLabel("cache warm (pre-warmed A, B)");
}
BENCHMARK(BM_EG_MatMul_CacheWarm)->Arg(128)->Arg(256)->Arg(512)->Unit(benchmark::kMicrosecond);

// =============================================================================
// §11  NUMERICAL EDGE CASES  (correctness is not tested here — only perf)
// =============================================================================

static void BM_TL_EdgeCase_MixedMagnitude(benchmark::State& st) {
    const size_t N = 1 << 16;
    auto large = tl_make_const1d(N, 1e10f);
    auto small = tl_make_const1d(N, 1e-10f);

    for (auto _ : st) {
        auto r = TensorOps::operator+(large, small);
        benchmark::DoNotOptimize(r);
    }
    st.SetLabel("large (1e10) + small (1e-10)");
}
BENCHMARK(BM_TL_EdgeCase_MixedMagnitude);

static void BM_EG_EdgeCase_MixedMagnitude(benchmark::State& st) {
    const size_t N = 1 << 16;
    Eigen::VectorXf large = eg_make_const1d(N, 1e10f);
    Eigen::VectorXf small = eg_make_const1d(N, 1e-10f);

    for (auto _ : st) {
        Eigen::VectorXf r = large + small;
        benchmark::DoNotOptimize(r);
    }
    st.SetLabel("large (1e10) + small (1e-10)");
}
BENCHMARK(BM_EG_EdgeCase_MixedMagnitude);

static void BM_TL_EdgeCase_RepeatedAdd(benchmark::State& st) {
    // 1000 accumulations of a small delta — exercises allocation pressure
    const size_t N = 1 << 12;
    auto t = tl_make_const1d(N, 0.0f);
    auto delta = tl_make_const1d(N, 1e-4f);

    for (auto _ : st) {
        auto acc = tl_make_const1d(N, 0.0f);
        for (int i = 0; i < 1000; ++i)
            acc = TensorOps::operator+(acc, delta);
        benchmark::DoNotOptimize(acc);
    }
    st.SetLabel("1000 × acc += delta  [1000 allocs]");
}
BENCHMARK(BM_TL_EdgeCase_RepeatedAdd);

static void BM_EG_EdgeCase_RepeatedAdd(benchmark::State& st) {
    const size_t N = 1 << 12;
    Eigen::VectorXf delta = eg_make_const1d(N, 1e-4f);

    for (auto _ : st) {
        Eigen::VectorXf acc = Eigen::VectorXf::Zero(static_cast<Eigen::Index>(N));
        for (int i = 0; i < 1000; ++i)
            acc += delta;
        benchmark::DoNotOptimize(acc);
    }
    st.SetLabel("1000 × acc += delta  [in-place, 0 allocs]");
}
BENCHMARK(BM_EG_EdgeCase_RepeatedAdd);

// =============================================================================
// §12  RESHAPE (TensorLib metadata-only — should be near-zero)
// =============================================================================

static void BM_TL_Reshape(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto t = tl_make_random(N, N);

    for (auto _ : st) {
        t.reshape({N * N});
        t.reshape({N, N});
        benchmark::DoNotOptimize(t.getShape().data());
    }
    st.SetLabel("reshape (metadata only, no copy)");
}
BENCHMARK(BM_TL_Reshape)->Arg(64)->Arg(256)->Arg(1024);

// Eigen Map (zero-copy view): analogous to reshape — no data movement
static void BM_EG_Reshape_Map(benchmark::State& st) {
    const size_t N = static_cast<size_t>(st.range(0));
    auto m = eg_make_random(N, N);
    float* data = m.data();
    const auto total = static_cast<Eigen::Index>(N * N);

    for (auto _ : st) {
        Eigen::Map<Eigen::VectorXf> flat(data, total);
        benchmark::DoNotOptimize(flat.data());
        Eigen::Map<ERowMat> mat(data, static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(N));
        benchmark::DoNotOptimize(mat.data());
    }
    st.SetLabel("Map<>() remapping (metadata only)");
}
BENCHMARK(BM_EG_Reshape_Map)->Arg(64)->Arg(256)->Arg(1024);

// =============================================================================
// BENCHMARK_MAIN  — Google Benchmark entry point
// =============================================================================
BENCHMARK_MAIN();
