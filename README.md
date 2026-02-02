# TensorLib

A minimal, dependency-free C++20 tensor computation library designed for low-level ML infrastructure development and numerical computing research.

---

## 1. Project Overview

### What TensorLib Is

TensorLib is a statically-typed, header-accessible tensor library implementing core numerical primitives required for machine learning and scientific computing workloads. It provides:

- N-dimensional tensor abstraction (up to rank-8)
- Row-major contiguous memory layout with explicit stride computation
- Broadcasting-aware binary operations
- Matrix multiplication and 2D transpose operations
- Weight initialization schemes (He, Xavier, uniform variants)
- Move-only ownership semantics preventing accidental deep copies

### Why It Exists

TensorLib exists to demonstrate fundamental tensor computation principles without the abstraction layers and dependencies present in production frameworks (PyTorch, TensorFlow, Eigen). It serves as:

1. **Educational infrastructure** for understanding how tensor libraries work internally
2. **Prototyping platform** for exploring custom numerical kernels
3. **Benchmark baseline** for comparing performance against optimized implementations

### Intended Audience

- Systems engineers building ML runtimes
- Researchers prototyping custom tensor operations
- Developers studying low-level numerical computing patterns
- Engineers evaluating performance characteristics of naive vs. optimized implementations

### Non-Goals

TensorLib intentionally does **not** provide:

| Non-Goal | Rationale |
|----------|-----------|
| SIMD vectorization | Keeps implementation readable; users can benchmark against vectorized libraries |
| Multi-threading | Single-threaded baseline enables clear performance attribution |
| GPU backends | CPU-only focus for simplicity |
| Automatic differentiation | Planned but not implemented (see `autograd/GradFunc.h` stub) |
| Expression templates | Explicit temporary allocation for predictable memory behavior |
| Mixed precision | Float32 only; simplifies numerical analysis |
| Memory pooling | Each tensor owns its allocation independently |

---

## 2. Key Features

### Tensor Abstraction
- Fixed maximum rank of 8 dimensions via `std::array<size_t, MAX_RANK>`
- Dynamic rank computed from non-zero shape dimensions
- Scalar tensors supported (rank-0, total size 1)
- Factory functions: `createTensor`, `createZeros`, `createOnes`, `createScalar`, `createRandTensor`

### Memory Efficiency
- `std::unique_ptr<float[]>` ownership eliminates reference counting overhead
- Move-only semantics (copy constructor/assignment deleted)
- Contiguous row-major storage enables cache-efficient sequential access
- No hidden allocations in indexing operations

### Operation Support
| Category | Operations |
|----------|------------|
| Element-wise | `+`, `-`, `*`, `/` with broadcasting |
| Linear Algebra | `matmul` (2D only), `transpose2D` |
| Reductions | `calcCost` (SSE/MSE loss) |
| Activations | `sigmoid`, `relu`, `leakyRelu`, `m_tanh` (scalar functions) |
| Shape | `reshape`, `getShape`, `getStrides` |

### Performance Focus
- `ikj` loop ordering in matmul for improved cache locality
- Broadcasting computed once, then applied via stride manipulation
- Direct pointer access available via `getDataPtr()` / `getMutableDataPtr()`

### Extensibility
- Template-based `binaryKernel` accepts arbitrary binary functors
- Clean separation between tensor storage and operations (`Tensor` vs `TensorOps`)
- CSV data loading via companion `csvlib` library

---

## 3. Design Philosophy

### Why C++

C++ was chosen for:

1. **Deterministic memory management** — `unique_ptr` provides clear ownership without GC pauses
2. **Zero-overhead abstractions** — Inline functions, templates without virtual dispatch
3. **Low-level control** — Direct pointer arithmetic for kernel implementations
4. **Ecosystem compatibility** — Integration with Google Test, Google Benchmark, Eigen for comparison

### Data-Oriented Design Choices

| Decision | Rationale |
|----------|-----------|
| Fixed `MAX_RANK = 8` | Avoids heap allocation for shape/stride arrays; covers 99% of ML use cases |
| Row-major layout | Matches C/C++ array semantics; cache-efficient for last-dimension iteration |
| Contiguous storage | Enables `memcpy`, SIMD-ready (future), predictable cache behavior |
| Separate `TensorOps` namespace | Operations don't pollute `Tensor` class; enables future GPU/CPU dispatch |

### Tradeoffs Made

| Choice | Simplicity Gained | Performance Cost |
|--------|-------------------|------------------|
| No expression templates | Readable binary operation implementation | Intermediate tensor allocation per operation |
| Single-threaded | Predictable execution, simpler debugging | No parallelism |
| Naive matmul loops | Clear algorithm, easy to understand | 10-100x slower than optimized BLAS |
| No SIMD intrinsics | Portable across architectures | Leaves vectorization on the table |

---

## 4. Architecture Overview

### Tensor Class Structure

```cpp
class Tensor {
    std::unique_ptr<float[]> m_data;      // Owning pointer to contiguous float buffer
    std::array<size_t, MAX_RANK> m_shape; // Dimension sizes (0 = unused dimension)
    std::array<size_t, MAX_RANK> m_stride;// Precomputed strides for indexing
    size_t m_total_size;                  // Product of non-zero dimensions
    size_t m_rank;                        // Number of non-zero dimensions
    std::unique_ptr<float[]> m_grad;      // Gradient storage (future autograd)
};
```

### Shape, Stride, and Indexing Model

**Row-major strides** are computed right-to-left:

```
Shape:   [2, 3, 4]
Strides: [12, 4, 1]   // stride[i] = product(shape[i+1:])
```

**Indexing formula** for `tensor(i, j, k)`:

```
offset = i * stride[0] + j * stride[1] + k * stride[2]
       = i * 12 + j * 4 + k * 1
```

### Broadcasting Mechanism

`computeBroadcast()` produces a `BroadcastInfo` struct containing:

- Output shape (element-wise max of input shapes)
- Adjusted strides for each input (0 for broadcast dimensions)

Binary operations iterate over output indices, using adjusted strides to read inputs.

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Code                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Tensor::createTensor()  │  Tensor::createZeros()  │  createRandTensor()  │
├──────────────────────────┴─────────────────────────┴────────────────┤
│                         Tensor Class                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ m_data       │  │ m_shape      │  │ m_stride                 │   │
│  │ unique_ptr   │  │ [8] array    │  │ [8] array (precomputed)  │   │
│  │ float[]      │  │              │  │                          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                       TensorOps Namespace                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐   │
│  │ operator+   │  │ operator*   │  │ matmul      │  │transpose2D│   │
│  │ operator-   │  │ operator/   │  │ (ikj loop)  │  │           │   │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘  └───────────┘   │
│         │                │                                           │
│         └────────────────┴──────────┐                               │
│                                     ▼                               │
│                          ┌──────────────────┐                       │
│                          │  binaryKernel<>  │                       │
│                          │  (template)      │                       │
│                          └────────┬─────────┘                       │
│                                   ▼                                 │
│                          ┌──────────────────┐                       │
│                          │ computeBroadcast │                       │
│                          │ (stride adjust)  │                       │
│                          └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Repository Structure

```
TensorLib/
├── CMakeLists.txt          # Root build configuration (C++20, GTest, Benchmark)
├── README.md               # This file
├── .gitignore              # Build artifacts exclusion
│
├── tensor/                 # Core tensor library
│   ├── CMakeLists.txt      # Static library: tensor_lib
│   ├── include/
│   │   └── tensorlib/
│   │       ├── tensor.h        # Public API (includes tensor/tensor.h)
│   │       ├── ops.h           # Public API (includes ops/ops.h)
│   │       ├── loader.h        # CSV-to-Tensor loading
│   │       ├── tensor_RNG.h    # RNG singleton wrapper
│   │       ├── tensor/
│   │       │   ├── tensor.h    # Tensor class definition
│   │       │   └── tensor_RNG.h# TensorRNG class
│   │       ├── ops/
│   │       │   └── ops.h       # TensorOps namespace, BroadcastInfo
│   │       ├── data/
│   │       │   └── loader.h    # Data::toTensor()
│   │       └── autograd/
│   │           └── GradFunc.h  # Stub for future autograd
│   └── src/
│       ├── tensor/
│       │   ├── tensor.cpp      # Tensor implementation
│       │   └── tensor_RNG.cpp  # RNG engine singleton
│       ├── ops/
│       │   └── ops.cpp         # Operations implementation
│       └── data/
│           └── loader.cpp      # Empty (data loading not yet implemented)
│
├── csvlib/                 # CSV parsing library
│   ├── CMakeLists.txt      # Static library: csvlib
│   ├── include/csvlib/
│   │   └── csv.h           # CSVData struct, CSVParser class
│   └── src/
│       └── csv.cpp         # Fast CSV parsing with from_chars
│
├── tests/                  # Unit tests (Google Test)
│   ├── CMakeLists.txt
│   └── test_tensor.cpp     # Comprehensive test suite (80+ tests)
│
├── benchmarks/             # Performance benchmarks (Google Benchmark)
│   ├── CMakeLists.txt
│   └── bench_tensor.cpp    # Tensor vs Eigen comparisons
│
└── apps/                   # Example applications
    ├── CMakeLists.txt
    └── main.cpp            # CSV loading demonstration
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `tensor_lib` | Core tensor abstraction, operations, memory management |
| `csvlib` | High-performance CSV parsing with column-major storage |
| `tests` | Correctness verification via property-based and edge-case tests |
| `benchmarks` | Performance profiling against Eigen baseline |
| `apps` | Usage examples and integration demonstrations |

---

## 6. Building the Library

### Requirements

| Requirement | Version |
|-------------|---------|
| C++ Standard | C++20 |
| CMake | ≥ 3.20 |
| Compiler | GCC 10+, Clang 12+, MSVC 2019+ |
| Eigen (optional) | For benchmark comparisons |

### Build Commands

```bash
# Clone repository
git clone https://github.com/CarvedCoder/TensorLib.git
cd TensorLib

# Create build directory
mkdir build && cd build

# Configure (Release)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build all targets
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure

# Run benchmarks
./benchmarks/bench_tensor
```

### Debug Build (with AddressSanitizer + UBSan)

```bash
mkdir build-debug && cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . -j$(nproc)
ctest --output-on-failure
```

Debug builds automatically enable `-fsanitize=address,undefined`.

### Compiler Flags

| Build Type | Flags |
|------------|-------|
| Debug | `-fsanitize=address,undefined -fno-omit-frame-pointer` |
| Release | (user-specified; benchmarks use `-O3 -march=native`) |
| Warnings | `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror` |

---

## 7. Basic Usage Examples

### Creating Tensors

```cpp
#include <tensorlib/tensor.h>
#include <tensorlib/ops.h>

// From raw data
auto data = std::make_unique<float[]>(6);
for (size_t i = 0; i < 6; ++i) data[i] = static_cast<float>(i);
auto tensor = Tensor::createTensor(std::move(data), {2, 3});

// Factory functions
auto zeros = Tensor::createZeros({3, 4, 5});
auto ones = Tensor::createOnes({10, 10});
auto scalar = Tensor::createScalar(3.14f);

// Random initialization (He, Xavier, etc.)
auto weights = Tensor::createRandTensor({784, 256}, InitType::He);
```

### Accessing Elements

```cpp
auto t = Tensor::createOnes({3, 4});

// Read access (bounds-checked)
float val = t(1, 2);

// Direct pointer access (no bounds checking)
const float* ptr = t.getDataPtr();
float* mut_ptr = t.getMutableDataPtr();

// Modify element
t.setDataElem(5, 42.0f);

// View as span
std::span<const float> view = t.view();
```

### Common Operations

```cpp
auto A = Tensor::createOnes({100, 100});
auto B = Tensor::createOnes({100, 100});

// Element-wise (with broadcasting)
auto sum = TensorOps::operator+(A, B);
auto diff = TensorOps::operator-(A, B);
auto prod = TensorOps::operator*(A, B);
auto quot = TensorOps::operator/(A, B);

// Scalar multiplication
auto scaled = TensorOps::operator*(A, 2.5f);

// Matrix multiplication
auto C = TensorOps::matmul(A, B);

// Transpose
auto A_T = TensorOps::transpose2D(A);

// Loss computation
float mse = TensorOps::calcCost(A, B, LossType::MSE);
```

### Shape Inspection and Reshaping

```cpp
auto t = Tensor::createZeros({2, 3, 4});

size_t rank = t.getRank();           // 3
size_t total = t.getTotalSize();     // 24
auto shape = t.getShape();           // span{2, 3, 4}
auto strides = t.getStrides();       // span{12, 4, 1}

t.reshape({6, 4});                   // Now 2D: 6×4
t.reshape({24});                     // Now 1D: 24 elements
```

---

## 8. Supported Operations

### Element-wise Operations

| Operation | Syntax | Broadcasting | Complexity |
|-----------|--------|--------------|------------|
| Addition | `A + B` | ✓ | O(n) |
| Subtraction | `A - B` | ✓ | O(n) |
| Multiplication | `A * B` | ✓ | O(n) |
| Division | `A / B` | ✓ | O(n) |
| Scalar multiply | `A * scalar` | — | O(n) |

### Linear Algebra

| Operation | Function | Constraints | Complexity |
|-----------|----------|-------------|------------|
| Matrix multiply | `matmul(A, B)` | 2D tensors, inner dims match | O(m·n·k) |
| Transpose | `transpose2D(A)` | 2D tensors only | O(m·n) |

### Reductions

| Operation | Function | Output |
|-----------|----------|--------|
| SSE Loss | `calcCost(A, B, LossType::SSE)` | Σ(a-b)² |
| MSE Loss | `calcCost(A, B, LossType::MSE)` | Σ(a-b)² / n |

### Activation Functions (Scalar)

| Function | Formula |
|----------|---------|
| `sigmoid(x)` | 1 / (1 + e^(-x)) |
| `relu(x)` | max(0, x) |
| `leakyRelu(x)` | x > 0 ? x : 0.01 (Note: current impl returns constant; typical leaky ReLU uses 0.01x) |
| `m_tanh(x)` | tanh(x) |

### Shape Operations

| Operation | Function | Notes |
|-----------|----------|-------|
| Get shape | `getShape()` | Returns `span<const size_t>` |
| Get strides | `getStrides()` | Returns `span<const size_t>` |
| Reshape | `reshape({...})` | Total elements must match |

---

## 9. Memory Management

### Ownership Model

TensorLib uses **exclusive ownership** via `std::unique_ptr<float[]>`:

```cpp
// Tensor owns its data exclusively
std::unique_ptr<float[]> m_data;

// Copy is DELETED — prevents accidental deep copies
Tensor(const Tensor&) = delete;
Tensor& operator=(const Tensor&) = delete;

// Move is DEFAULT — transfers ownership efficiently
Tensor(Tensor&&) noexcept = default;
Tensor& operator=(Tensor&&) noexcept = default;
```

### Heap vs Stack Usage

| Component | Location | Rationale |
|-----------|----------|-----------|
| `m_data` buffer | Heap | Tensor data can be arbitrarily large |
| `m_shape[8]` | Stack (embedded) | Fixed size, no allocation |
| `m_stride[8]` | Stack (embedded) | Fixed size, no allocation |
| Scalar metadata | Stack (embedded) | `m_rank`, `m_total_size` |

### Copy vs Move Semantics

```cpp
// ❌ COMPILE ERROR: Copy deleted
Tensor copy = original;

// ✓ OK: Move transfers ownership
Tensor moved = std::move(original);  // original is now empty

// Operations return by value (move elision applies)
auto result = TensorOps::operator+(A, B);  // No copy, move/RVO
```

### Avoiding Unnecessary Allocations

1. **Factory functions** allocate once: `createZeros`, `createOnes`
2. **Reshape** modifies metadata only, no reallocation
3. **Direct pointer access** avoids copy: `getDataPtr()`
4. **Broadcasting** computes adjusted strides, doesn't copy inputs

---

## 10. Performance Considerations

### Cache Friendliness

**Row-major layout** ensures sequential memory access when iterating over the last dimension:

```cpp
// GOOD: Cache-friendly (sequential access)
for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
        process(tensor(i, j));

// BAD: Cache-unfriendly (strided access)
for (size_t j = 0; j < cols; ++j)
    for (size_t i = 0; i < rows; ++i)
        process(tensor(i, j));
```

### Loop Ordering in Matmul

TensorLib uses **ikj ordering** for matrix multiplication:

```cpp
for (size_t i = 0; i < m; i++) {
    for (size_t k = 0; k < K; k++) {
        float a_ik = A[i][k];  // Load once
        for (size_t j = 0; j < n; j++) {
            C[i][j] += a_ik * B[k][j];  // Sequential access in B and C
        }
    }
}
```

This provides better cache utilization than naive `ijk` ordering by:
- Loading `A[i][k]` once per inner loop
- Accessing `B[k][j]` sequentially (row-major)
- Writing `C[i][j]` sequentially

### Avoided Abstractions

| Abstraction Avoided | Benefit |
|---------------------|---------|
| Virtual dispatch | No vtable indirection |
| Expression templates | Simpler implementation, predictable allocation |
| Reference counting | No atomic operations, clear ownership |
| Dynamic type erasure | Compile-time known types |

### Known Bottlenecks

| Bottleneck | Impact | Mitigation Path |
|------------|--------|-----------------|
| No SIMD | 4-8x slower than vectorized | Future: AVX2/NEON intrinsics |
| No threading | Single-core only | Future: OpenMP/thread pool |
| Naive matmul | 10-100x slower than BLAS | Future: Blocked/tiled implementation |
| Intermediate allocations | Memory bandwidth limited | Future: Expression templates |

---

## 11. Error Handling & Safety

### Bounds Checking Strategy

| Access Method | Bounds Checked | Throws |
|---------------|----------------|--------|
| `operator()(i)` | ✓ | `std::out_of_range` |
| `operator()(i, j)` | ✓ | `std::out_of_range` |
| `operator()(i, j, k)` | ✓ | `std::out_of_range` |
| `getDataPtr()[i]` | ✗ | (undefined behavior) |
| `setDataElem(i, v)` | ✗ | (undefined behavior) |

### Assertions vs Runtime Errors

TensorLib uses **exceptions** for recoverable errors:

```cpp
// Shape validation
if (shape_list.size() > MAX_RANK)
    throw std::invalid_argument("Shape passed is greater than 8");

// Dimension mismatch
if (K != s2[0])
    throw std::invalid_argument("Inner ranks in matmul aren't the same");

// Reshape validation
if (total_elem != m_total_size)
    throw std::invalid_argument("reshape doesn't contain all the elements");
```

### Debug vs Release Behavior

| Feature | Debug | Release |
|---------|-------|---------|
| AddressSanitizer | Enabled | Disabled |
| UndefinedBehaviorSanitizer | Enabled | Disabled |
| Bounds checking in `operator()` | Always | Always |
| Compiler warnings as errors | Enabled | Enabled |

---

## 12. Extending TensorLib

### Adding New Operations

1. **Declare** in `tensor/include/tensorlib/ops/ops.h`:
```cpp
namespace TensorOps {
Tensor exp(const Tensor& t);
}
```

2. **Implement** in `tensor/src/ops/ops.cpp`:
```cpp
Tensor TensorOps::exp(const Tensor& t) {
    auto result = Tensor::createZeros(t.getShape());
    auto out = result.getMutableDataPtr();
    auto in = t.getDataPtr();
    for (size_t i = 0; i < t.getTotalSize(); ++i)
        out[i] = std::exp(in[i]);
    return result;
}
```

3. **Test** in `tests/test_tensor.cpp`:
```cpp
TEST(UnaryOpsTest, Exp) {
    auto t = Tensor::createScalar(1.0f);
    auto result = TensorOps::exp(t);
    EXPECT_FLOAT_EQ(result(0), std::exp(1.0f));
}
```

### Using binaryKernel for Custom Binary Ops

```cpp
// In ops.cpp
Tensor TensorOps::pow(const Tensor& base, const Tensor& exp) {
    return binaryKernel(base, exp, [](float a, float b) {
        return std::pow(a, b);
    });
}
```

### Where to Plug in Optimizations

| Optimization | Location | Approach |
|--------------|----------|----------|
| SIMD loops | `binaryKernel`, `matmul` | Add intrinsics, keep scalar fallback |
| Blocked matmul | `matmul()` | Tile loops for cache blocking |
| Thread parallelism | `binaryKernel` | OpenMP pragmas on outer loop |

### Contributor Guidelines

1. Maintain move-only semantics — never add copy constructors
2. Use factory functions for tensor creation — constructors are private
3. Add tests for new operations — follow existing test patterns
4. Run benchmarks — ensure no performance regressions
5. Enable all warnings — code must compile with `-Werror`

---

## 13. Testing

### Test Structure

Tests are organized by tensor rank and operation category:

```cpp
// Rank-based tests
TEST(ScalarTest, ...)       // Rank-0
TEST(Tensor1DTest, ...)     // Rank-1
TEST(Tensor2DTest, ...)     // Rank-2
TEST(Tensor3DTest, ...)     // Rank-3
TEST(HighRankTest, ...)     // Rank-7, Rank-8

// Operation tests
TEST(ElementwiseOpsTest, ...)
TEST(TransposeTest, ...)
TEST(MatMulTest, ...)

// Edge cases
TEST(FloatEdgeCaseTest, ...)   // Infinity, NaN, epsilon
TEST(NumericalStabilityTest, ...)
TEST(PropertyTest, ...)        // Commutativity, associativity
```

### Correctness Verification

| Verification Type | Example |
|-------------------|---------|
| Invariant checks | `getRank()`, `getTotalSize()` match shape |
| Bounds checking | `EXPECT_THROW(tensor(5), std::out_of_range)` |
| Numerical correctness | `EXPECT_FLOAT_EQ(result, expected)` |
| Property testing | Addition commutativity: `a+b == b+a` |
| Edge cases | `inf + finite`, `nan * anything` |

### Running Tests

```bash
# Run all tests
cd build
ctest --output-on-failure

# Run specific test
./tests/test_tensor --gtest_filter="MatMulTest.*"

# Run with verbose output
./tests/test_tensor --gtest_list_tests
```

---

## 14. Limitations & Known Gaps

### Missing Features

| Feature | Status | Notes |
|---------|--------|-------|
| Autograd | Stub only | `GradFunc.h` is empty |
| Data loader | Not implemented | `loader.cpp` is empty |
| Slicing/views | Not supported | Tensors are always contiguous |
| Type templates | Float32 only | No `double`, `int`, `half` |
| Batched matmul | Not supported | Must loop manually |
| Higher-rank matmul | 2D only | `matmul` throws for rank > 2 |

### Scalability Constraints

| Constraint | Limit | Workaround |
|------------|-------|------------|
| Maximum rank | 8 | None (hardcoded `MAX_RANK`) |
| Data type | float32 | None (requires templates) |
| Thread count | 1 | None (single-threaded) |
| Memory | System RAM | None (no out-of-core support) |

### Numeric Limitations

| Issue | Behavior |
|-------|----------|
| Float precision | Standard IEEE 754 float32 |
| Overflow | Produces `inf` |
| Underflow | Produces `0` or denormal |
| NaN propagation | Standard IEEE 754 semantics |
| Catastrophic cancellation | Not handled (user responsibility) |

---

## 15. Roadmap

### Near-term

- [ ] **SIMD vectorization** — AVX2/NEON for element-wise ops and matmul
- [ ] **Blocked matmul** — Cache-aware tiling for large matrices
- [ ] **Tensor slicing** — Non-owning views into existing tensors

### Medium-term

- [ ] **Multi-threading** — OpenMP or thread pool for parallel loops
- [ ] **Expression templates** — Fused operations to reduce allocations
- [ ] **Type templates** — Support for `double`, `int32`, `float16`

### Long-term

- [ ] **GPU backend** — CUDA/Metal/OpenCL compute kernels
- [ ] **Autograd** — Reverse-mode automatic differentiation
- [ ] **Graph compilation** — Operation fusion and scheduling

---

## 16. Why This Project Matters

### Technical Competency Demonstrated

| Skill | Evidence |
|-------|----------|
| Modern C++ (C++20) | `std::span`, `std::unique_ptr`, move semantics |
| Memory management | RAII, no raw `new`/`delete`, move-only types |
| Numerical computing | Broadcasting, stride computation, cache-aware loops |
| Testing discipline | 80+ unit tests, property-based testing, edge cases |
| Performance awareness | Benchmarks vs Eigen, GFLOP/s measurement |
| Build systems | CMake, sanitizers, cross-platform warnings |

### Systems + ML Relevance

TensorLib demonstrates understanding of:

1. **How tensor frameworks work internally** — not just API usage
2. **Memory layout decisions** that affect ML performance
3. **Trade-offs between abstraction and performance**
4. **Testing numerical software** for correctness and stability

This is the foundation that production tensor libraries (XLA, TVM, Triton, oneDNN) build upon.

---

## 17. License

No license file is currently present in the repository.

**Note**: Without an explicit license, default copyright applies (all rights reserved). Consider adding MIT, Apache 2.0, or BSD-3-Clause for open-source use.

---

## Acknowledgments

- [Google Test](https://github.com/google/googletest) — Testing framework
- [Google Benchmark](https://github.com/google/benchmark) — Microbenchmarking framework
- [Eigen](https://eigen.tuxfamily.org/) — Reference implementation for benchmark comparisons
