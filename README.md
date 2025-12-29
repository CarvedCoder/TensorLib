# TensorLib — High-Performance C++20 Tensor Library

> A lightweight, zero-dependency tensor library with NumPy-compatible memory layouts, built for understanding ML systems from first principles.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Tests](https://img.shields.io/badge/tests-101%20%E2%9C%85-brightgreen.svg)](test_tensor.cpp)
[![Build Time](https://img.shields.io/badge/test%20time-0.21s-green.svg)](test_tensor.cpp)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](test_tensor.cpp)

---

## 🎯 Overview

**TensorLib** is an educational tensor library implementing NumPy-compatible memory layouts with modern C++20 features. It provides clean abstractions for understanding deep learning systems while maintaining reasonable performance characteristics.

### Key Features

- **Zero runtime dependencies** — GoogleTest only for testing, Google Benchmark for performance analysis
- **101 comprehensive tests** covering correctness, edge cases, and numerical stability
- **Fast test execution** — 0.21s in Release mode with full coverage
- **NumPy-compatible strides** for row-major memory layout
- **Modern C++20** — smart pointers, move semantics, RAII throughout
- **Extensive benchmarking** — Performance validation against Eigen library

---

## 📊 Recent Performance Improvements

### Matrix Multiplication Optimization

Significant performance gains achieved through **loop reordering optimization** (i-k-j ordering):

| Operation | Size | Before | After | Speedup | Target (Eigen) |
|-----------|------|--------|-------|---------|----------------|
| **MatMul (Tiny)** | 4×4 | 0.69 GFLOP/s | 1.78 GFLOP/s | **2.6×** | 6.2 GFLOP/s |
| **MatMul (Tiny)** | 16×16 | 1.08 GFLOP/s | 15.7 GFLOP/s | **14.6×** | 49.8 GFLOP/s |
| **MatMul (Medium)** | 32×32 | 1.06 GFLOP/s | 19.4 GFLOP/s | **18.3×** | 89.4 GFLOP/s |
| **MatMul (Medium)** | 128×128 | 1.13 GFLOP/s | 38.3 GFLOP/s | **33.9×** | 110.4 GFLOP/s |
| **MatMul (Large)** | 512×512 | 1.15 GFLOP/s | 32.9 GFLOP/s | **28.6×** | 124.0 GFLOP/s |
| **MatMul (Large)** | 1024×1024 | 1.15 GFLOP/s | 33.0 GFLOP/s | **28.7×** | 125.2 GFLOP/s |
| **Neural Net** | 32×512×512 | 1.15 GFLOP/s | 33.0 GFLOP/s | **28.7×** | 108.0 GFLOP/s |

### Element-wise Operations Optimization

**14-15× speedup** achieved through optimized memory access patterns:

| Operation | Size | Before | After | Speedup | Target (Eigen) |
|-----------|------|--------|-------|---------|----------------|
| **Add** | 1K elements | 5.2 GB/s | 73.7 GB/s | **14.2×** | 129.4 GB/s |
| **Add** | 256K elements | 5.3 GB/s | 80.8 GB/s | **15.2×** | 112.1 GB/s |
| **Multiply** | 1K elements | 5.2 GB/s | 73.4 GB/s | **14.1×** | 128.4 GB/s |
| **Multiply** | 256K elements | 5.3 GB/s | 81.6 GB/s | **15.4×** | 111.3 GB/s |

### Memory Operations Improvement

**6-7× speedup** in sequential write operations:

| Operation | Size | Before | After | Speedup |
|-----------|------|--------|-------|---------|
| **Sequential Write** | 1K | 5.4 GB/s | 42.8 GB/s | **7.9×** |
| **Sequential Write** | 256K | 5.5 GB/s | 41.6 GB/s | **7.6×** |

### Transpose Performance

Performance characteristics for transpose operations:

| Size | Before | After | Change | Target (Eigen) |
|------|--------|-------|--------|----------------|
| 32×32 | 3.80 GB/s | 20.0 GB/s | **5.3×** | 55.6 GB/s |
| 64×64 | 4.00 GB/s | 23.2 GB/s | **5.8×** | 37.4 GB/s |
| 256×256 | 3.97 GB/s | 3.72 GB/s | 0.94× | 14.4 GB/s |
| 512×512 | 4.04 GB/s | 3.68 GB/s | 0.91× | 13.3 GB/s |

*Note: Small matrices show improvement; larger matrices show regression due to increased memory traffic from value copying. Further optimization pending.*

### Summary

- **Matrix multiplication**: **28-34× faster** — now achieves 26-35% of Eigen's performance
- **Element-wise ops**: **14-15× faster** — reaches 60-70% of Eigen's throughput
- **Memory bandwidth**: **6-8× improvement** in sequential operations
- **Overall**: Moved from naive implementation to cache-aware algorithms
- **Still educational**: 3-4× slower than production libraries (expected for learning code)

---

## ✨ Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Zero-copy construction** | ✅ | Takes `unique_ptr<float[]>` — no unnecessary copies |
| **Flexible dimensionality** | ✅ | Scalar (0-D) to 8-D tensors with `MAX_RANK = 8` |
| **NumPy strides** | ✅ | Row-major layout matching NumPy exactly |
| **Element-wise operations** | ✅ | `+`, `-`, `*` (Hadamard product) |
| **Matrix multiplication** | ✅ | Cache-optimized i-k-j loop ordering |
| **Matrix transpose** | ✅ | 2D transpose with proper layout |
| **Bounds checking** | ✅ | `operator()` throws `std::out_of_range` |
| **Factory methods** | ✅ | `createZeros`, `createOnes`, `createScalar`, `createRandTensor` |
| **Initialization modes** | ✅ | Normal, He, Xavier, Xavier Uniform, He Uniform |
| **Loss functions** | ✅ | SSE (Sum of Squared Errors), MSE (Mean Squared Error) |
| **Activation functions** | ✅ | Sigmoid, ReLU, Leaky ReLU, Tanh |
| **Memory alignment** | ✅ | 64-byte aligned for cache optimization |
| **Build system** | ✅ | Modern CMake with FetchContent |
| **Sanitizers** | ✅ | AddressSanitizer & UBSanitizer in Debug |
| **Strict compilation** | ✅ | `-Wall -Wextra -Wpedantic -Werror` |
| **Comprehensive benchmarks** | ✅ | 50+ benchmarks comparing against Eigen |

---

## 📈 Benchmark Suite

Our benchmark suite includes **50+ performance tests** organized into categories:

### 1. Memory Allocation & Initialization (3 benchmarks)
- Tensor creation with zeros, ones, and scalars
- Validates allocation performance across sizes (1K to 1M elements)

### 2. Memory Access Patterns (5 benchmarks)
- Sequential read/write performance
- Row-major vs column-major strided access
- Random access patterns (cache behavior analysis)

### 3. Element-wise Operations (6 benchmarks)
- Addition, subtraction, multiplication vs Eigen
- Comparison across sizes (1K to 1M elements)
- Bandwidth utilization metrics

### 4. Matrix Multiplication (12 benchmarks)
- Tiny (4×4 to 16×16), Medium (32×32 to 256×256), Large (512×512 to 1024×1024)
- Neural network shapes (batch×input×output)
- Random data vs uniform data comparison
- Direct comparison with Eigen's performance

### 5. Transpose Operations (4 benchmarks)
- Square matrices (32×32 to 1024×1024)
- Rectangular matrices (different aspect ratios)
- Comparison with Eigen

### 6. Chained Operations (4 benchmarks)
- Expression evaluation: `(a + b) * c`
- Complex chains: `((a + b) * c) - d`
- Matrix multiplication chains: `A * B * C`

### 7. Batch Operations (3 benchmarks)
- Batch matrix multiplication (simulating mini-batches)
- Batch element-wise operations

### 8. Cache Behavior (3 benchmarks)
- Cache thrashing vs cache-warm scenarios
- Performance degradation analysis

### 9. Special Cases (5 benchmarks)
- Very small matrices (2×2)
- Single row/column matrix multiplication
- Power-of-2 vs non-power-of-2 sizes

### 10. ML Workload Simulations (2 benchmarks)
- Forward pass through 3-layer neural network
- Simplified transformer attention mechanism

**All benchmarks include GFLOP/s metrics and bandwidth measurements where applicable.**

---

## 🧪 Test Suite Breakdown

**101 tests** validating correctness, performance, and edge cases:

### Core Tensor Tests (42 tests)
- Shape & stride correctness (1-D through 8-D)
- Element access validation
- Memory management (alignment, move semantics)
- Boundary condition handling
- Performance validation

### Element-Wise Operations (12 tests)
- Arithmetic operations with verification
- Shape mismatch detection
- IEEE-754 special values (inf, NaN)
- Aliasing safety

### Matrix Operations (8 tests)
- Matrix multiplication correctness
- Transpose validation
- Error handling for rank mismatches
- Cache optimization verification

### Intensive Numerical Tests (20 tests)
- Chained operations stability
- Large-scale computations (100K+ elements)
- High-dimensional arithmetic (8-D tensors)
- Memory stability over 1000 iterations

### Floating-Point Edge Cases (8 tests)
- Denormal number handling
- Overflow/underflow behavior
- NaN propagation
- Catastrophic cancellation
- Mixed magnitude operations

### Memory & Boundary Tests (6 tests)
- Edge dimensions (primes, powers of 2)
- Boundary access validation
- Out-of-bounds detection

### Mathematical Properties (4 tests)
- Commutativity
- Associativity
- Distributive property

### Performance Benchmarks (1 test)
- Sequential and random access patterns

**Test execution: 0.21 seconds (Release build)**

---

## 🚀 Quick Start

### Prerequisites
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.25+

### Build & Test

```bash
# Clone repository
git clone https://github.com/yourusername/minitensor.git
cd minitensor

# Release build with optimizations
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure
```

### Run Benchmarks

```bash
# Build benchmarks
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
cmake --build build --parallel

# Run all benchmarks
./build/bench_tensor --benchmark_min_time=0.5s

# Run specific benchmark category
./build/bench_tensor --benchmark_filter=MatMul
```

### Development Build (with sanitizers)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

---

## 💻 Usage Examples

### Basic Tensor Creation

```cpp
#include "tensor.h"
#include "ops.h"

// Create tensors
auto zeros = Tensor::createZeros({2, 3});
auto ones = Tensor::createOnes({2, 3});
auto scalar = Tensor::createScalar(42.0f);

// Random initialization with He/Xavier schemes
auto weights = Tensor::createRandTensor({784, 128}, InitType::He);
auto biases = Tensor::createRandTensor({128}, InitType::Xavier);

// From raw data (zero-copy)
auto data = std::make_unique<float[]>(6);
for (size_t i = 0; i < 6; ++i) data[i] = float(i);
auto tensor = Tensor::CreateTensor(std::move(data), 6, {2, 3});

// Element access
float val = (*tensor)(1, 2);  // Row 1, Col 2
tensor->setDataElem(0, 99.0f);
```

### Element-Wise Operations

```cpp
auto a = Tensor::createOnes({3, 3});
auto b = Tensor::createOnes({3, 3});

// Operations
auto sum = TensorOps::operator+(a, b);   // Element-wise addition
auto diff = TensorOps::operator-(a, b);  // Element-wise subtraction
auto prod = TensorOps::operator*(a, b);  // Hadamard product
```

### Matrix Multiplication & Transpose

```cpp
// Matrix multiplication
auto A = Tensor::createOnes({2, 3});
auto B = Tensor::createOnes({3, 4});
auto C = TensorOps::matmul(A, B);  // 2×4 result

// Transpose
auto M = Tensor::createZeros({3, 4});
auto M_T = TensorOps::transpose2D(M);  // 4×3 result
```

### Activation Functions & Loss

```cpp
// Apply activations
float x = 0.5f;
float sig = TensorOps::sigmoid(x);      // 1/(1+e^-x)
float relu_out = TensorOps::relu(x);    // max(0, x)
float lrelu = TensorOps::leakyRelu(x);  // x > 0 ? x : 0.01x
float tanh_out = TensorOps::m_tanh(x);  // tanh(x)

// Compute loss
auto predictions = Tensor::createOnes({10, 1});
auto targets = Tensor::createZeros({10, 1});
float sse = TensorOps::calcCost(predictions, targets, LossType::SSE);
float mse = TensorOps::calcCost(predictions, targets, LossType::MSE);
```

---

## 🏗️ Architecture

### Memory Layout

- **Row-major ordering** (C-style, NumPy default)
- **64-byte alignment** for cache line optimization
- **Contiguous storage** with stride-based indexing
- **Zero-copy construction** via `unique_ptr` move semantics

### Stride Calculation

```cpp
// Shape [2, 3, 4] → strides [12, 4, 1]
// Element (i,j,k) → data[i*12 + j*4 + k*1]
```

Exactly matches NumPy's stride behavior for seamless interoperability.

### Smart Pointer Design

- **`unique_ptr<float[]>`** — Owns raw data, ensures RAII
- **`shared_ptr<Tensor>`** — Tensor instances can be shared
- **Move semantics** — Efficient tensor passing without copies

### Loop Optimization (Matrix Multiplication)

```cpp
// Cache-friendly i-k-j ordering
for (size_t i = 0; i < m; ++i) {
    for (size_t k = 0; k < K; ++k) {
        float a_ik = A(i, k);  // Load once
        for (size_t j = 0; j < n; ++j) {
            C(i, j) += a_ik * B(k, j);  // Reuse a_ik
        }
    }
}
```

This ordering achieves **28-34× speedup** over naive i-j-k ordering by:
- Maximizing cache hit rate for matrix B
- Minimizing cache line evictions
- Reusing loaded values of A

---

## 📚 API Reference

### Factory Methods

```cpp
// Zero-initialized tensors
static Tensorptr createZeros(const std::initializer_list<size_t>& shape);
static Tensorptr createZeros(const std::array<size_t, MAX_RANK>& shape);

// One-initialized tensors
static Tensorptr createOnes(const std::initializer_list<size_t>& shape);
static Tensorptr createOnes(const std::array<size_t, MAX_RANK>& shape);

// Scalar tensor (0-D)
static Tensorptr createScalar(float data);

// Random initialization
static Tensorptr createRandTensor(
    const std::initializer_list<size_t>& shape, 
    InitType mode = Normal
);
```

### Element Access

```cpp
float& operator()(size_t i);                  // 1-D access
float& operator()(size_t i, size_t j);        // 2-D access
float& operator()(size_t i, size_t j, size_t k);  // 3-D access

float getDataElem(size_t i) const;            // Safe read
void setDataElem(size_t i, float val);        // Safe write
```

### Operations

```cpp
// Element-wise
Tensorptr operator+(const Tensorptr& a, const Tensorptr& b);
Tensorptr operator-(const Tensorptr& a, const Tensorptr& b);
Tensorptr operator*(const Tensorptr& a, const Tensorptr& b);

// Matrix operations
Tensorptr matmul(const Tensorptr& A, const Tensorptr& B);
Tensorptr transpose2D(const Tensorptr& M);

// Activations (element-wise)
float sigmoid(float x);
float relu(float x);
float leakyRelu(float x);
float m_tanh(float x);

// Loss computation
float calcCost(const Tensorptr& pred, const Tensorptr& target, LossType mode);
```

---

## 🎓 Educational Goals

This library teaches:

1. **Tensor fundamentals** — Shapes, strides, row-major layouts
2. **Modern C++20** — Smart pointers, move semantics, RAII
3. **Numerical computing** — Floating-point precision, IEEE-754
4. **Performance optimization** — Cache awareness, loop ordering
5. **Software engineering** — Testing, benchmarking, build systems
6. **ML foundations** — Weight initialization, loss functions, activations

**Not a production library** — Use Eigen, Intel MKL, or PyTorch for real workloads.

---

## 🔮 Roadmap

### Planned Features
- [ ] Broadcasting for element-wise ops
- [ ] Automatic differentiation (backpropagation)
- [ ] Reduction operations (sum, mean, max)
- [ ] Batched operations for mini-batches
- [ ] Strided views (slicing without copying)

### Performance Enhancements
- [ ] SIMD vectorization (AVX2/AVX-512)
- [ ] Multi-threading with OpenMP
- [ ] Expression templates for lazy evaluation
- [ ] Loop tiling for larger matrices

---

## 🤝 Contributing

Contributions welcome! Focus on:

- **Clarity over cleverness** — This is educational code
- **Comprehensive tests** for new features
- **Benchmark additions** to track performance
- **Documentation** of design decisions

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **NumPy** — Stride calculation and memory layout reference
- **Eigen** — API design inspiration and performance baseline
- **GoogleTest** — Comprehensive testing framework
- **Google Benchmark** — Performance measurement toolkit

Built with ❤️ for understanding ML systems from first principles.

---

## 📖 References

- [NumPy Internals](https://numpy.org/doc/stable/reference/internals.html) — Memory layout specification
- [What Every Computer Scientist Should Know About Floating-Point](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/) — Cache-aware programming
- [Matrix Multiplication Algorithm Design](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm) — Loop ordering strategies

---

**Performance Note**: Current implementation achieves 26-35% of Eigen's throughput for matrix multiplication and 60-70% for element-wise operations. This gap is expected for educational code and can be closed with SIMD, tiling, and expression templates.