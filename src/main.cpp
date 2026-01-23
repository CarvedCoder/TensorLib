#include "tensorlib/ops/ops.h"
#include <iostream>
#include <tensorlib/tensor/tensor.h>
int main() {

    auto t1 = Tensor::createRandTensor({2, 2});
    auto t2 = Tensor::createRandTensor({2, 2}, InitType::He);
    auto result = TensorOps::canBroadcast(t1, t2);
    if (result)
        std::cout << "Broadcast possible\n";
    else
        std::cout << "Broadcast not possible\n";
    using namespace TensorOps;
    auto matmul_t = matmul(t1, t2);
    for (const auto v : matmul_t.view()) {
        std::cout << v << " ";
    }
}
