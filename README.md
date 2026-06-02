# TensorLib

TensorLib is a C++23 tensor computation library geared toward low-level ML infrastructure work and numerical computing research. It provides a small, fast core with explicit control over memory, shape handling, and numerical operations, plus optional utilities for CSV ingestion and simple data loading.

## Features
- Tensor creation helpers (zeros, ones, random initializers, scalars)
- Shape/rank management with row-major indexing
- Elementwise arithmetic with broadcasting
- Matrix multiplication and 2D transpose
- Common activations and scalers (sigmoid, ReLU, tanh, min/max/standard scalers)
- Lightweight autograd graph registry (node tracking)
- CSV parsing utilities and tensor conversion helpers

## Repository Layout
- `tensor/` — core tensor and ops implementation (`tensorlib` static library)
- `csvlib/` — CSV parsing utilities (`csvlib` static library)
- `dataLoader/` — CSV-to-tensor helpers (`dataLoader` static library)
- `apps/` — example applications
- `benchmarks/` — performance benchmarks
- `tests/` — unit tests

## Requirements
- C++23 toolchain
- CMake 3.20+
- Eigen3 (used by tests/benchmarks; fetched automatically if not installed)
- GoogleTest and Google Benchmark (fetched automatically via CMake)

## Build
```bash
cmake -S . -B build
cmake --build build
```

## Usage (Minimal Example)
```cpp
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>

int main() {
    auto a = Tensor::createOnes({2, 3});
    auto b = Tensor::createOnes({3, 2});
    auto c = TensorOps::matmul(a, b);
    return 0;
}
```

## Tests
```bash
ctest --test-dir build
```

## Benchmarks
```bash
./build/benchmarks/bench_tensor
```

## Applications
- `linearRegression` — simple linear regression over CSV data (see `apps/linearRegression.cpp`)
- `test` — placeholder app

## License
See [LICENSE](LICENSE).
