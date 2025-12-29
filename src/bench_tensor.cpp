#include "../include/ops.h"
#include "../include/tensor.h"
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <random>

// ============================================================
// HELPERS
// ============================================================
static Tensor::Tensorptr make_tensor(size_t r, size_t c, float v = 1.0f) {
    auto t = Tensor::createZeros({r, c});
    for (size_t i = 0; i < r * c; ++i)
        t->setDataElem(i, v);
    return t;
}

static Tensor::Tensorptr make_random_tensor(size_t r, size_t c) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    auto t = Tensor::createZeros({r, c});
    for (size_t i = 0; i < r * c; ++i)
        t->setDataElem(i, dist(gen));
    return t;
}

static Eigen::MatrixXf make_eigen(size_t r, size_t c, float v = 1.0f) {
    return Eigen::MatrixXf::Constant(r, c, v);
}

static Eigen::MatrixXf make_random_eigen(size_t r, size_t c) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    Eigen::MatrixXf m(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            m(i, j) = dist(gen);
    return m;
}

// ============================================================
// SECTION 1: MEMORY ALLOCATION & INITIALIZATION
// ============================================================
static void BM_CreateZeros(benchmark::State &state) {
    const size_t N = state.range(0);
    for (auto _ : state) {
        auto t = Tensor::createZeros({N});
        benchmark::DoNotOptimize(t);
    }
    state.SetBytesProcessed(state.iterations() * N * sizeof(float));
}
BENCHMARK(BM_CreateZeros)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

static void BM_CreateOnes(benchmark::State &state) {
    const size_t N = state.range(0);
    for (auto _ : state) {
        auto t = Tensor::createOnes({N});
        benchmark::DoNotOptimize(t);
    }
    state.SetBytesProcessed(state.iterations() * N * sizeof(float));
}
BENCHMARK(BM_CreateOnes)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

static void BM_CreateScalar(benchmark::State &state) {
    for (auto _ : state) {
        auto t = Tensor::createScalar(3.14159f);
        benchmark::DoNotOptimize(t);
    }
}
BENCHMARK(BM_CreateScalar);

// ============================================================
// SECTION 2: MEMORY ACCESS PATTERNS
// ============================================================
static void BM_SequentialRead(benchmark::State &state) {
    const size_t N = state.range(0);
    auto t = Tensor::createOnes({N});
    const float *data = t->getDataPtr();
    
    for (auto _ : state) {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i)
            sum += data[i];
        benchmark::DoNotOptimize(sum);
    }
    state.SetBytesProcessed(state.iterations() * N * sizeof(float));
}
BENCHMARK(BM_SequentialRead)->Range(1<<10, 1<<22)->Unit(benchmark::kMicrosecond);

static void BM_SequentialWrite(benchmark::State &state) {
    const size_t N = state.range(0);
    auto t = Tensor::createZeros({N});
    float *data = t->getMutableDataPtr();
    
    for (auto _ : state) {
        for (size_t i = 0; i < N; ++i)
            data[i] = static_cast<float>(i);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * N * sizeof(float));
}
BENCHMARK(BM_SequentialWrite)->Range(1<<10, 1<<22)->Unit(benchmark::kMicrosecond);

static void BM_StridedAccess_RowMajor(benchmark::State &state) {
    const size_t N = state.range(0);
    auto t = Tensor::createZeros({N, N});
    
    for (auto _ : state) {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                sum += (*t)(i, j);
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_StridedAccess_RowMajor)->Range(32, 512);

static void BM_StridedAccess_ColMajor(benchmark::State &state) {
    const size_t N = state.range(0);
    auto t = Tensor::createZeros({N, N});
    
    for (auto _ : state) {
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j)  // Column-first (cache unfriendly)
            for (size_t i = 0; i < N; ++i)
                sum += (*t)(i, j);
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_StridedAccess_ColMajor)->Range(32, 512);

static void BM_RandomAccess(benchmark::State &state) {
    const size_t N = state.range(0);
    auto t = Tensor::createZeros({N, N});
    
    // Pre-generate random indices
    std::vector<size_t> indices(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> dist(0, N - 1);
    for (auto &idx : indices) idx = dist(gen);
    
    for (auto _ : state) {
        float sum = 0.0f;
        for (size_t k = 0; k < indices.size(); k += 2)
            sum += (*t)(indices[k], indices[k + 1]);
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_RandomAccess)->Range(128, 2048);

// ============================================================
// SECTION 3: ELEMENTWISE OPERATIONS (Tensor vs Eigen)
// ============================================================
static void BM_Tensor_Add(benchmark::State &state) {
    const size_t N = state.range(0);
    auto a = Tensor::createOnes({N});
    auto b = Tensor::createOnes({N});
    
    for (auto _ : state) {
        auto r = TensorOps::operator+(a, b);
        benchmark::DoNotOptimize(r);
    }
    state.SetBytesProcessed(state.iterations() * N * 3 * sizeof(float));
}
BENCHMARK(BM_Tensor_Add)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

static void BM_Eigen_Add(benchmark::State &state) {
    const size_t N = state.range(0);
    Eigen::VectorXf a = Eigen::VectorXf::Ones(N);
    Eigen::VectorXf b = Eigen::VectorXf::Ones(N);
    
    for (auto _ : state) {
        Eigen::VectorXf r = a + b;
        benchmark::DoNotOptimize(r);
    }
    state.SetBytesProcessed(state.iterations() * N * 3 * sizeof(float));
}
BENCHMARK(BM_Eigen_Add)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

static void BM_Tensor_Mul(benchmark::State &state) {
    const size_t N = state.range(0);
    auto a = Tensor::createOnes({N});
    auto b = Tensor::createOnes({N});
    
    for (auto _ : state) {
        auto r = TensorOps::operator*(a, b);
        benchmark::DoNotOptimize(r);
    }
    state.SetBytesProcessed(state.iterations() * N * 3 * sizeof(float));
}
BENCHMARK(BM_Tensor_Mul)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

static void BM_Eigen_Mul(benchmark::State &state) {
    const size_t N = state.range(0);
    Eigen::VectorXf a = Eigen::VectorXf::Ones(N);
    Eigen::VectorXf b = Eigen::VectorXf::Ones(N);
    
    for (auto _ : state) {
        Eigen::VectorXf r = a.cwiseProduct(b);
        benchmark::DoNotOptimize(r);
    }
    state.SetBytesProcessed(state.iterations() * N * 3 * sizeof(float));
}
BENCHMARK(BM_Eigen_Mul)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

static void BM_Tensor_Sub(benchmark::State &state) {
    const size_t N = state.range(0);
    auto a = Tensor::createOnes({N});
    auto b = Tensor::createOnes({N});
    
    for (auto _ : state) {
        auto r = TensorOps::operator-(a, b);
        benchmark::DoNotOptimize(r);
    }
    state.SetBytesProcessed(state.iterations() * N * 3 * sizeof(float));
}
BENCHMARK(BM_Tensor_Sub)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

// ============================================================
// SECTION 4: MATRIX MULTIPLICATION (Core Performance)
// ============================================================

// Small matrices (fits in L1 cache)
static void BM_Tensor_MatMul_Tiny(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_tensor(N, N);
    auto B = make_tensor(N, N);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_MatMul_Tiny)->DenseRange(4, 16, 4);

static void BM_Eigen_MatMul_Tiny(benchmark::State &state) {
    const size_t N = state.range(0);
    Eigen::MatrixXf A = make_eigen(N, N);
    Eigen::MatrixXf B = make_eigen(N, N);
    
    for (auto _ : state) {
        Eigen::MatrixXf C = A * B;
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Eigen_MatMul_Tiny)->DenseRange(4, 16, 4);

// Medium matrices (L2/L3 cache)
static void BM_Tensor_MatMul_Medium(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_tensor(N, N);
    auto B = make_tensor(N, N);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_MatMul_Medium)->RangeMultiplier(2)->Range(32, 256);

static void BM_Eigen_MatMul_Medium(benchmark::State &state) {
    const size_t N = state.range(0);
    Eigen::MatrixXf A = make_eigen(N, N);
    Eigen::MatrixXf B = make_eigen(N, N);
    
    for (auto _ : state) {
        Eigen::MatrixXf C = A * B;
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Eigen_MatMul_Medium)->RangeMultiplier(2)->Range(32, 256);

// Large matrices (exceeds L3 cache)
static void BM_Tensor_MatMul_Large(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_tensor(N, N);
    auto B = make_tensor(N, N);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_MatMul_Large)->Arg(512)->Arg(768)->Arg(1024);

static void BM_Eigen_MatMul_Large(benchmark::State &state) {
    const size_t N = state.range(0);
    Eigen::MatrixXf A = make_eigen(N, N);
    Eigen::MatrixXf B = make_eigen(N, N);
    
    for (auto _ : state) {
        Eigen::MatrixXf C = A * B;
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Eigen_MatMul_Large)->Arg(512)->Arg(768)->Arg(1024);

// ============================================================
// SECTION 5: RECTANGULAR MATRICES (ML Workloads)
// ============================================================

// Typical neural network layer shapes
static void BM_Tensor_MatMul_NeuralNet(benchmark::State &state) {
    const size_t batch = state.range(0);
    const size_t in_features = state.range(1);
    const size_t out_features = state.range(2);
    
    auto input = make_tensor(batch, in_features);
    auto weight = make_tensor(in_features, out_features);
    
    for (auto _ : state) {
        auto output = TensorOps::matmul(input, weight);
        benchmark::DoNotOptimize(output);
    }
    
    double flops = 2.0 * batch * in_features * out_features;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_MatMul_NeuralNet)
    ->Args({1, 512, 512})      // Single input
    ->Args({32, 512, 512})     // Small batch
    ->Args({128, 512, 512})    // Medium batch
    ->Args({32, 784, 128})     // MNIST-like
    ->Args({64, 2048, 1024})   // Large layer
    ->Args({256, 512, 256});   // ResNet-like

static void BM_Eigen_MatMul_NeuralNet(benchmark::State &state) {
    const size_t batch = state.range(0);
    const size_t in_features = state.range(1);
    const size_t out_features = state.range(2);
    
    Eigen::MatrixXf input = make_eigen(batch, in_features);
    Eigen::MatrixXf weight = make_eigen(in_features, out_features);
    
    for (auto _ : state) {
        Eigen::MatrixXf output = input * weight;
        benchmark::DoNotOptimize(output);
    }
    
    double flops = 2.0 * batch * in_features * out_features;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Eigen_MatMul_NeuralNet)
    ->Args({1, 512, 512})
    ->Args({32, 512, 512})
    ->Args({128, 512, 512})
    ->Args({32, 784, 128})
    ->Args({64, 2048, 1024})
    ->Args({256, 512, 256});

// ============================================================
// SECTION 6: TRANSPOSE OPERATIONS
// ============================================================
static void BM_Tensor_Transpose(benchmark::State &state) {
    const size_t N = state.range(0);
    auto t = make_tensor(N, N);
    
    for (auto _ : state) {
        auto t_T = TensorOps::transpose2D(t);
        benchmark::DoNotOptimize(t_T);
    }
    
    state.SetBytesProcessed(state.iterations() * N * N * sizeof(float) * 2);
}
BENCHMARK(BM_Tensor_Transpose)->RangeMultiplier(2)->Range(32, 1024);

static void BM_Eigen_Transpose(benchmark::State &state) {
    const size_t N = state.range(0);
    Eigen::MatrixXf m = make_eigen(N, N);
    
    for (auto _ : state) {
        Eigen::MatrixXf m_T = m.transpose();
        benchmark::DoNotOptimize(m_T);
    }
    
    state.SetBytesProcessed(state.iterations() * N * N * sizeof(float) * 2);
}
BENCHMARK(BM_Eigen_Transpose)->RangeMultiplier(2)->Range(32, 1024);

// Non-square transpose
static void BM_Tensor_Transpose_Rectangular(benchmark::State &state) {
    const size_t M = state.range(0);
    const size_t N = state.range(1);
    auto t = make_tensor(M, N);
    
    for (auto _ : state) {
        auto t_T = TensorOps::transpose2D(t);
        benchmark::DoNotOptimize(t_T);
    }
}
BENCHMARK(BM_Tensor_Transpose_Rectangular)
    ->Args({1024, 256})
    ->Args({256, 1024})
    ->Args({2048, 512})
    ->Args({512, 2048});

// ============================================================
// SECTION 7: CHAINED OPERATIONS (Expression Trees)
// ============================================================
static void BM_Tensor_ChainedOps_Simple(benchmark::State &state) {
    const size_t N = state.range(0);
    auto a = Tensor::createOnes({N});
    auto b = Tensor::createOnes({N});
    auto c = Tensor::createOnes({N});
    
    for (auto _ : state) {
        // (a + b) * c
        auto r = TensorOps::operator*(TensorOps::operator+(a, b), c);
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_Tensor_ChainedOps_Simple)->Range(1<<10, 1<<20);

static void BM_Tensor_ChainedOps_Complex(benchmark::State &state) {
    const size_t N = state.range(0);
    auto a = Tensor::createOnes({N});
    auto b = Tensor::createOnes({N});
    auto c = Tensor::createOnes({N});
    auto d = Tensor::createOnes({N});
    
    for (auto _ : state) {
        // ((a + b) * c) - d
        auto t1 = TensorOps::operator+(a, b);
        auto t2 = TensorOps::operator*(t1, c);
        auto t3 = TensorOps::operator-(t2, d);
        benchmark::DoNotOptimize(t3);
    }
}
BENCHMARK(BM_Tensor_ChainedOps_Complex)->Range(1<<10, 1<<20);

static void BM_Tensor_MatMul_Chain(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_tensor(N, N);
    auto B = make_tensor(N, N);
    auto C = make_tensor(N, N);
    
    for (auto _ : state) {
        // A * B * C
        auto AB = TensorOps::matmul(A, B);
        auto ABC = TensorOps::matmul(AB, C);
        benchmark::DoNotOptimize(ABC);
    }
    
    double flops = 2.0 * N * N * N * 2; // Two matmuls
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_MatMul_Chain)->Range(32, 256);

// ============================================================
// SECTION 8: BATCH OPERATIONS
// ============================================================
static void BM_Tensor_BatchMatMul(benchmark::State &state) {
    const size_t batch = state.range(0);
    const size_t N = state.range(1);
    
    std::vector<Tensor::Tensorptr> As, Bs;
    for (size_t i = 0; i < batch; ++i) {
        As.push_back(make_tensor(N, N));
        Bs.push_back(make_tensor(N, N));
    }
    
    for (auto _ : state) {
        for (size_t i = 0; i < batch; ++i) {
            auto C = TensorOps::matmul(As[i], Bs[i]);
            benchmark::DoNotOptimize(C);
        }
    }
    
    double flops = 2.0 * batch * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_BatchMatMul)
    ->Args({8, 64})
    ->Args({16, 64})
    ->Args({32, 64})
    ->Args({8, 128})
    ->Args({16, 128});

static void BM_Tensor_BatchAdd(benchmark::State &state) {
    const size_t batch = state.range(0);
    const size_t N = state.range(1);
    
    std::vector<Tensor::Tensorptr> As, Bs;
    for (size_t i = 0; i < batch; ++i) {
        As.push_back(Tensor::createOnes({N}));
        Bs.push_back(Tensor::createOnes({N}));
    }
    
    for (auto _ : state) {
        for (size_t i = 0; i < batch; ++i) {
            auto C = TensorOps::operator+(As[i], Bs[i]);
            benchmark::DoNotOptimize(C);
        }
    }
}
BENCHMARK(BM_Tensor_BatchAdd)
    ->Args({100, 1000})
    ->Args({1000, 100})
    ->Args({10000, 10});

// ============================================================
// SECTION 9: CACHE BEHAVIOR TESTS
// ============================================================
static void BM_Tensor_CacheThrashing(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_tensor(N, N);
    auto B = make_tensor(N, N);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_CacheThrashing)->Arg(256)->Arg(512)->Arg(1024);

static void BM_Tensor_CacheWarm(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_tensor(N, N);
    auto B = make_tensor(N, N);
    
    // Warm up cache
    auto warmup = TensorOps::matmul(A, B);
    benchmark::DoNotOptimize(warmup);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_CacheWarm)->Arg(256)->Arg(512);

// ============================================================
// SECTION 10: NUMERICAL STABILITY TESTS
// ============================================================
static void BM_Tensor_MixedMagnitude(benchmark::State &state) {
    const size_t N = 10000;
    auto large = Tensor::createOnes({N});
    auto small = Tensor::createOnes({N});
    
    // Set contrasting magnitudes
    for (size_t i = 0; i < N; ++i) {
        large->setDataElem(i, 1e10f);
        small->setDataElem(i, 1e-10f);
    }
    
    for (auto _ : state) {
        auto sum = TensorOps::operator+(large, small);
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_Tensor_MixedMagnitude);

static void BM_Tensor_RepeatedOps(benchmark::State &state) {
    const size_t N = state.range(0);
    auto t = Tensor::createOnes({N});
    auto delta = Tensor::createOnes({N});
    
    // Set small delta
    for (size_t i = 0; i < N; ++i)
        delta->setDataElem(i, 0.001f);
    
    for (auto _ : state) {
        for (size_t i = 0; i < 1000; ++i) {
            t = TensorOps::operator+(t, delta);
        }
        benchmark::DoNotOptimize(t);
    }
}
BENCHMARK(BM_Tensor_RepeatedOps)->Arg(100)->Arg(1000);

// ============================================================
// SECTION 11: MEMORY BANDWIDTH TESTS
// ============================================================
static void BM_Memcpy_Baseline(benchmark::State &state) {
    const size_t N = state.range(0);
    auto src = Tensor::createOnes({N});
    auto dst = Tensor::createZeros({N});
    
    for (auto _ : state) {
        std::memcpy(dst->getMutableDataPtr(), src->getDataPtr(), 
                    N * sizeof(float));
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * N * sizeof(float));
}
BENCHMARK(BM_Memcpy_Baseline)
    ->Range(1<<10, 1<<24)
    ->Unit(benchmark::kMicrosecond);

static void BM_Tensor_Copy_Elementwise(benchmark::State &state) {
    const size_t N = state.range(0);
    auto src = Tensor::createOnes({N});
    auto dst = Tensor::createZeros({N});
    
    for (auto _ : state) {
        for (size_t i = 0; i < N; ++i)
            dst->setDataElem(i, src->getDataPtr()[i]);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * N * sizeof(float));
}
BENCHMARK(BM_Tensor_Copy_Elementwise)
    ->Range(1<<10, 1<<20)
    ->Unit(benchmark::kMicrosecond);

// ============================================================
// SECTION 12: SPECIAL CASES & EDGE CONDITIONS
// ============================================================
static void BM_Tensor_VerySmallMatMul(benchmark::State &state) {
    auto A = make_tensor(2, 2);
    auto B = make_tensor(2, 2);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
}
BENCHMARK(BM_Tensor_VerySmallMatMul);

static void BM_Tensor_SingleRowMatMul(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_tensor(1, N);
    auto B = make_tensor(N, N);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
}
BENCHMARK(BM_Tensor_SingleRowMatMul)->Range(64, 1024);

static void BM_Tensor_SingleColMatMul(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_tensor(N, N);
    auto B = make_tensor(N, 1);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
}
BENCHMARK(BM_Tensor_SingleColMatMul)->Range(64, 1024);

// ============================================================
// SECTION 13: POWER-OF-2 vs NON-POWER-OF-2 SIZES
// ============================================================
static void BM_Tensor_MatMul_PowerOf2(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_tensor(N, N);
    auto B = make_tensor(N, N);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_MatMul_PowerOf2)
    ->Arg(64)->Arg(128)->Arg(256)->Arg(512);

static void BM_Tensor_MatMul_NonPowerOf2(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_tensor(N, N);
    auto B = make_tensor(N, N);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_MatMul_NonPowerOf2)
    ->Arg(63)->Arg(127)->Arg(255)->Arg(511);

// ============================================================
// SECTION 14: OVERHEAD & SHARED_PTR COSTS
// ============================================================
static void BM_SharedPtr_Creation(benchmark::State &state) {
    for (auto _ : state) {
        auto t = Tensor::createScalar(1.0f);
        benchmark::DoNotOptimize(t);
    }
}
BENCHMARK(BM_SharedPtr_Creation);

static void BM_SharedPtr_Copy(benchmark::State &state) {
    auto original = Tensor::createScalar(1.0f);
    
    for (auto _ : state) {
        auto copy = original;  // Shared pointer copy
        benchmark::DoNotOptimize(copy);
    }
}
BENCHMARK(BM_SharedPtr_Copy);

static void BM_RawPosize_ter_Creation(benchmark::State &state) {
    for (auto _ : state) {
        float *ptr = new float(1.0f);
        benchmark::DoNotOptimize(ptr);
        delete ptr;
    }
}
BENCHMARK(BM_RawPosize_ter_Creation);

// ============================================================
// SECTION 15: COMPARISON WITH RANDOM DATA
// ============================================================
static void BM_Tensor_MatMul_Random(benchmark::State &state) {
    const size_t N = state.range(0);
    auto A = make_random_tensor(N, N);
    auto B = make_random_tensor(N, N);
    
    for (auto _ : state) {
        auto C = TensorOps::matmul(A, B);
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_MatMul_Random)->Range(64, 512);

static void BM_Eigen_MatMul_Random(benchmark::State &state) {
    const size_t N = state.range(0);
    Eigen::MatrixXf A = make_random_eigen(N, N);
    Eigen::MatrixXf B = make_random_eigen(N, N);
    
    for (auto _ : state) {
        Eigen::MatrixXf C = A * B;
        benchmark::DoNotOptimize(C);
    }
    
    double flops = 2.0 * N * N * N;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Eigen_MatMul_Random)->Range(64, 512);

// ============================================================
// SECTION 16: REALISTIC ML WORKLOAD SIMULATION
// ============================================================

// Simulates forward pass through a simple neural network
static void BM_Tensor_MLForwardPass(benchmark::State &state) {
    const size_t batch = state.range(0);
    
    // Input: batch x 784 (MNIST-like)
    auto input = make_tensor(batch, 784);
    
    // Layer 1: 784 -> 256
    auto w1 = make_tensor(784, 256);
    
    // Layer 2: 256 -> 128
    auto w2 = make_tensor(256, 128);
    
    // Layer 3: 128 -> 10
    auto w3 = make_tensor(128, 10);
    
    for (auto _ : state) {
        auto h1 = TensorOps::matmul(input, w1);
        auto h2 = TensorOps::matmul(h1, w2);
        auto output = TensorOps::matmul(h2, w3);
        benchmark::DoNotOptimize(output);
    }
    
    // Total FLOPs for all 3 layers
    double flops = 2.0 * batch * (784 * 256 + 256 * 128 + 128 * 10);
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_MLForwardPass)
    ->Arg(1)->Arg(32)->Arg(64)->Arg(128)->Arg(256);

// Simulates transformer attention mechanism (simplified)
static void BM_Tensor_AttentionMechanism(benchmark::State &state) {
    const size_t seq_len = state.range(0);
    const size_t d_model = 512;
    
    // Q, K, V matrices
    auto Q = make_tensor(seq_len, d_model);
    auto K = make_tensor(seq_len, d_model);
    auto V = make_tensor(seq_len, d_model);
    
    for (auto _ : state) {
        // Attention scores: Q @ K^T
        auto K_T = TensorOps::transpose2D(K);
        auto scores = TensorOps::matmul(Q, K_T);
        
        // Attention output: scores @ V
        auto output = TensorOps::matmul(scores, V);
        benchmark::DoNotOptimize(output);
    }
    
    double flops = 2.0 * seq_len * seq_len * d_model * 2;
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_Tensor_AttentionMechanism)
    ->Arg(16)->Arg(32)->Arg(64)->Arg(128);

BENCHMARK_MAIN();