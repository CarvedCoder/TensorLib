## MiniTensor – A Lightweight C++20 Tensor Library

Made it so that i can use it for my other projects

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![Tests](https://img.shields.io/badge/tests-42%20✅-brightgreen.svg)](test_tensor.cpp)

Tiny, header-friendly tensor library with NumPy-compatible strides, zero dependencies at runtime, and 100 % test coverage via GoogleTest.

## Current Features

| Feature                   | Status | Notes                                                    |
| ------------------------- | ------ | -------------------------------------------------------- |
| **Zero-copy creation**    | ✅     | Takes `unique_ptr&lt;float[]&gt;` – no extra memcpy      |
| **Scalar → 8-D tensors**  | ✅     | `MAX_RANK = 8`, shape `{0}` = 0-D scalar                 |
| **NumPy strides**         | ✅     | Row-major, calculated exactly like NumPy                 |
| **Element-wise ops**      | ✅     | `+`, `-`, `*` (shape must match, no broadcast yet)       |
| **Bounds checking**       | ✅     | `operator()` throws `std::out_of_range`                  |
| **Factory helpers**       | ✅     | `createZeros`, `createOnes`, `createScalar`              |
| **Alignment**             | ✅     | All data 64-byte aligned for SIMD / cache lines          |
| **Modern CMake**          | ✅     | FetchContent pulls GoogleTest, sanitizers in Debug       |
| **Compiler warnings**     | ✅     | `-Wall -Wextra -Werror` (GCC/Clang) and `/W4 /WX` (MSVC) |
| **UB & AddressSanitizer** | ✅     | Auto-enabled in Debug builds                             |

---

## What the 42 Tests Prove

| Suite                  | Tests                      | Highlights                                             |
| ---------------------- | -------------------------- | ------------------------------------------------------ |
| **TensorTest (30)**    | Shape & stride correctness | 1-D … 8-D, shapes with ones, negative/overflow indices |
|                        | Element access             | Flat index vs. multi-index equivalence                 |
|                        | Memory safety              | 64-byte alignment, move semantics, unique ownership    |
|                        | Performance                | 1 000 000-element zero-init &lt; 50 ms                 |
| **TensorOpsTest (12)** | Scalar & 2-D ops           | `+`, `-`, `*` with exact element verification          |
|                        | Error handling             | Mismatched shapes throw `invalid_argument`             |
|                        | IEEE-754 edge cases        | `inf - inf = nan`, `nan + x = nan`                     |
|                        | Alias safety               | `t*t` does **not** mutate original tensor              |

All tests pass in **31 ms** on a cold run (Release build).

---

## Quick Start

```bash
git clone https://github.com/yourname/minitensor.git
cd minitensor
```

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

or

```bash
mkdir build && cd build
cmake ..
make
```

Run any program you want
