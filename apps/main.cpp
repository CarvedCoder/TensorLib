#include "tensorlib/ops/ops.h"
#include "tensorlib/tensor/tensor.h"
#include <iostream>
#include <tensorlib/tensor.h>
int main() {
    auto t1 = Tensor::createTensor({1, 2, 3, 4, 5, 6}, {2, 3});
    auto t2 = Tensor::createTensor({7, 8, 9, 10, 11, 12}, {3, 2});
    auto t3 = TensorOps::matmul(t1, t2);
    for (const auto elem : t3.view()) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}
