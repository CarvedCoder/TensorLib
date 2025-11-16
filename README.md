# MiniTensor — A Lightweight C++20 Tensor Library

> A high-performance, zero-dependency tensor library built for understanding ML systems from the ground up.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Tests](https://img.shields.io/badge/tests-101%20%E2%9C%85-brightgreen.svg)](test_tensor.cpp)
[![Build Time](https://img.shields.io/badge/test%20time-0.21s-green.svg)](test_tensor.cpp)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](test_tensor.cpp)

---

## 🎯 Overview

**MiniTensor** is a tensor library implementing NumPy-compatible memory layouts with modern C++20 features. Designed for educational purposes and lightweight ML projects, it provides clean abstractions without sacrificing performance fundamentals.

### Key Highlights

- **101 comprehensive tests** covering edge cases, numerical stability, and performance
- **0.21 second test suite** execution in Release mode
- **Zero runtime dependencies** — GoogleTest only for testing
- **100% test coverage** with sanitizers enabled in Debug builds
- **NumPy-compatible strides** for seamless interoperability
- **Modern C++20** — smart pointers, move semantics, RAII principles

---

## ✨ Features

| Feature                      | Status | Description                                              |
| ---------------------------- | ------ | -------------------------------------------------------- |
| **Zero-copy construction**   | ✅      | Takes `unique_ptr<float[]>` — no unnecessary memcpy      |
| **Flexible dimensionality**  | ✅      | Scalar (0-D) to 8-D tensors with `MAX_RANK = 8`         |
| **NumPy strides**            | ✅      | Row-major layout calculated exactly like NumPy           |
| **Element-wise operations**  | ✅      | `+`, `-`, `*` (Hadamard product)                         |
| **Matrix multiplication**    | ✅      | Optimized i-k-j loop ordering for cache efficiency      |
| **Matrix transpose**         | ✅      | 2D transpose with proper memory layout                   |
| **Bounds checking**          | ✅      | `operator()` throws `std::out_of_range` on violations    |
| **Factory methods**          | ✅      | `createZeros`, `createOnes`, `createScalar`              |
| **Memory alignment**         | ✅      | 64-byte aligned data for SIMD and cache optimization     |
| **Build system**             | ✅      | Modern CMake with FetchContent, sanitizers in Debug      |
| **Strict compilation**       | ✅      | `-Wall -Wextra -Wpedantic -Werror` (GCC/Clang), `/W4 /WX` (MSVC) |
| **UB detection**             | ✅      | AddressSanitizer & UBSanitizer auto-enabled in Debug     |

---

## 📊 Test Suite Breakdown

**101 tests** organized into comprehensive suites validating correctness, performance, and edge cases:

### Core Tensor Tests (42 tests)
- **Shape & stride correctness**: 1-D through 8-D tensors, mixed dimensions
- **Element access**: Flat indexing vs multi-dimensional equivalence
- **Memory management**: 64-byte alignment, move semantics, smart pointers
- **Boundary conditions**: Out-of-bounds detection, overflow handling
- **Performance validation**: 1M element initialization < 50ms

### Element-Wise Operations (12 tests)
- **Arithmetic ops**: Addition, subtraction, multiplication with full verification
- **Shape validation**: Mismatched shapes throw `invalid_argument`
- **Numerical edge cases**: IEEE-754 special values (inf, nan)
- **Aliasing safety**: `t * t` creates new tensor without mutation

### Matrix Operations (8 tests)
- **Matrix multiplication**: Correctness on various sizes (2×2 to large matrices)
- **Transpose**: 2D matrix transposition with stride verification
- **Error handling**: Rank and dimension mismatch detection
- **Cache optimization**: i-k-j loop ordering validated

### Intensive Numerical Tests (20 tests)
- **Chained operations**: Numerical stability through multiple ops
- **Large-scale computations**: 100k+ element operations with performance benchmarks
- **High-dimensional tensors**: 8-D tensor arithmetic
- **Repeated operations**: Memory stability over 1000 iterations

### Floating-Point Edge Cases (8 tests)
- **Denormal numbers**: Subnormal value handling
- **Overflow/underflow**: Max float values and infinity arithmetic
- **NaN propagation**: Correct NaN behavior across all operations
- **Precision loss**: Catastrophic cancellation demonstrations
- **Mixed magnitudes**: 1e20 + 1e-20 precision tests

### Memory & Boundary Tests (6 tests)
- **Edge dimensions**: 1, 2, primes, powers of 2, large primes (101×103)
- **Boundary access**: Corner element validation in 2D/3D tensors
- **Out-of-bounds**: SIZE_MAX and large index validation

### Mathematical Properties (4 tests)
- **Commutativity**: Addition and multiplication order independence
- **Associativity**: Operation grouping equivalence
- **Distributive property**: a × (b + c) = a×b + a×c

### Performance Benchmarks (1 test)
- **Sequential access**: 1M element write operations
- **Random access**: 10k 2D indexing operations

**All tests pass in 0.21 seconds (Release build)**

---

## 🚀 Quick Start

### Prerequisites
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.25+

### Build & Test

```bash
# Clone the repository
git clone https://github.com/yourusername/minitensor.git
cd minitensor

# Configure with Release optimizations
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure
```

**Expected output:**
```
100% tests passed, 0 tests failed out of 101
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

// Create a 2×3 tensor filled with zeros
auto t1 = Tensor::createZeros({2, 3});

// Create from raw data
auto data = std::make_unique<float[]>(6);
for (size_t i = 0; i < 6; ++i) data[i] = static_cast<float>(i);
auto t2 = Tensor::CreateTensor(std::move(data), 6, {2, 3});

// Access elements
float val = (*t2)(1, 2);  // Row 1, Col 2
t2->setDataElem(0, 42.0f);
```

### Element-Wise Operations

```cpp
auto a = Tensor::createOnes({3, 3});
auto b = Tensor::createOnes({3, 3});

// Element-wise operations
auto sum = TensorOps::operator+(a, b);     // [3, 3] all 2.0
auto diff = TensorOps::operator-(a, b);    // [3, 3] all 0.0
auto prod = TensorOps::operator*(a, b);    // [3, 3] all 1.0 (Hadamard)
```

### Matrix Multiplication

```cpp
auto A = Tensor::createOnes({2, 3});  // 2×3 matrix
auto B = Tensor::createOnes({3, 4});  // 3×4 matrix

auto C = TensorOps::matmul(A, B);     // 2×4 matrix
// C(i,j) = Σ A(i,k) * B(k,j)
```

### Matrix Transpose

```cpp
auto M = Tensor::createZeros({3, 4});
auto M_T = TensorOps::transpose2D(M);  // 4×3 matrix
```

---

## 🏗️ Architecture

### Memory Layout

- **Row-major ordering** (C-style, NumPy default)
- **64-byte alignment** for cache line optimization
- **Contiguous storage** with stride-based indexing
- **Zero-copy construction** using `unique_ptr` move semantics

### Stride Calculation

```cpp
// For shape [2, 3, 4]:
// strides = [12, 4, 1]
// Element at (i, j, k) → data[i*12 + j*4 + k*1]
```

Matches NumPy's stride behavior exactly for interoperability.

### Smart Pointer Usage

- **`unique_ptr<float[]>`** for raw data ownership
- **`shared_ptr<Tensor>`** for tensor instances (enables shared semantics)
- **Move semantics** throughout to avoid unnecessary copies

---

## 🧪 Testing Philosophy

### Comprehensive Coverage

- **Correctness**: Every operation validated with known outputs
- **Edge cases**: Boundary conditions, special values (inf, nan, denormals)
- **Performance**: Benchmarks ensure no regressions
- **Memory safety**: Sanitizers catch leaks, overflows, undefined behavior

### Test Categories

1. **Unit tests**: Individual operations in isolation
2. **Integration tests**: Chained operations and complex expressions
3. **Stress tests**: Large tensors, repeated operations, numerical stability
4. **Property tests**: Mathematical identities (commutativity, associativity)

---

## 🎓 Educational Goals

This library is designed to teach:

1. **Tensor fundamentals**: Shapes, strides, memory layouts
2. **Modern C++**: Smart pointers, move semantics, RAII
3. **Numerical computing**: Floating-point edge cases, precision
4. **Systems thinking**: Cache optimization, memory alignment
5. **Software engineering**: Testing, build systems, API design

---

## 🔮 Roadmap

### Planned Features
- [ ] Broadcasting for element-wise operations
- [ ] Automatic differentiation (autodiff) for backpropagation
- [ ] Reduction operations (sum, mean, max along axes)
- [ ] Advanced matrix operations (determinant, inverse, eigenvalues)
- [ ] Batched matrix multiplication for mini-batches
- [ ] Expression templates for lazy evaluation
- [ ] SIMD vectorization (AVX2/AVX-512)

### Performance Optimizations
- [ ] Loop tiling for better cache utilization
- [ ] Parallel operations using OpenMP/TBB
- [ ] In-place operations to reduce allocations

---

## 📈 Performance Notes

- **Matrix multiplication**: Naive O(n³) with i-k-j loop ordering for cache efficiency
- **Not optimized**: This is an educational library, not production BLAS
- **Comparison**: ~100-1000× slower than Eigen/MKL (expected for learning implementation)
- **Good for**: Understanding fundamentals, prototyping, small-scale experiments

For production workloads, use Eigen, Intel MKL, or cuBLAS.

---

## 🤝 Contributing

Contributions welcome! This is primarily an educational project, so focus on:

- Clarity and readability over micro-optimizations
- Comprehensive tests for new features
- Documentation of design decisions

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NumPy**: Memory layout and stride calculation reference
- **GoogleTest**: Comprehensive testing framework
- **Eigen**: Inspiration for API design

Built with ❤️ for learning ML systems from first principles.

---

## 📚 References

For understanding the concepts:
- [NumPy internals documentation](https://numpy.org/doc/stable/reference/internals.html)
- [Efficient C++ Performance Programming](https://www.agner.org/optimize/)
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
