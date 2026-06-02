# TensorLib

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/C%2B%2B-23-blue)
![CMake](https://img.shields.io/badge/CMake-3.20%2B-brightgreen)

A minimal, dependency-free C++23 tensor computation library designed for low-level ML infrastructure development and numerical computing research.

## Why TensorLib?
TensorLib exists to provide a small, transparent tensor core for learning, experimentation, and infrastructure work where you want full control over memory layout and math operations without heavyweight dependencies.

## Key Features
- N-dimensional tensor core (up to rank 8) with shape/stride-aware indexing
- Factory helpers for zeros, ones, scalars, and randomized initialization (Normal/He/Xavier)
- Elementwise arithmetic with broadcasting, matrix multiplication, and 2D transpose
- Basic activation functions and scaling utilities (min/max, standard, etc.)
- Lightweight autograd node tracking (registry scaffolding)
- CSV parsing + data loader helpers to convert features into tensors
- Example app (linear regression) plus benchmarks and tests

## Tech Stack
- **Language:** C++23
- **Build:** CMake 3.20+
- **Testing:** GoogleTest
- **Benchmarking:** Google Benchmark
- **Optional math:** Eigen (used in tests/benchmarks)

## Getting Started
1. **Clone the repo**
2. **Configure & build**
3. **Run tests or an example app**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## Installation (Local Build)
```bash
# Release build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Debug build (enables sanitizers)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Usage
### Quick Example
```cpp
#include <tensorlib/tensor.h>
#include <tensorlib/ops.h>

using namespace TensorOps;

auto a = Tensor::createOnes({2, 2});
auto b = Tensor::createRandTensor({2, 2}, InitType::He);
auto c = matmul(a, b);
```

### Run the Example App
```bash
./build/apps/linearRegression
```
> Note: Update the CSV dataset path in `apps/linearRegression.cpp` to point at your local data.

### Run Benchmarks
```bash
./build/benchmarks/bench_tensor
```

## Folder Structure
```
tensor/       Core tensor implementation and ops
csvlib/       CSV parser utilities
dataLoader/   CSV-to-tensor helpers
apps/         Example applications
benchmarks/   Performance benchmarks
tests/        Unit tests
```

## Future Improvements
- Expand autograd to full backprop support
- Add more tensor ops (conv, reduce, slicing)
- Optional GPU backend
- Better dataset utilities and examples

## License
MIT — see [LICENSE](LICENSE).

## Contributing
Contributions are welcome. Open an issue or PR with your proposal.
