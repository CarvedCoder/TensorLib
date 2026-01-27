#include <iostream>
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>
#include <tensorlib/tensor_RNG.h>
int main() {
    TensorRNG::setSeed(42);
    auto t1 = Tensor::createRandTensor({30, 1, 20});
    auto t2 = Tensor::createRandTensor({30, 20}, InitType::He);
    std::cout << "Tensor 1 :- \n";
    for (auto v : t1.view()) {
        std::cout << v << " ";
    }
    std::cout << "\nTensor 2 :- \n";
    for (auto v : t2.view()) {
        std::cout << v << " ";
    }
    using namespace TensorOps;
    auto result1 = t1 + t2;
    auto result2 = t1 * t2;

    std::cout << "\nTensor shape before reshape\n";

    for (auto dim : t1.getShape()) {
        std::cout << dim << " ";
    }

    t1.reshape({30, 20});

    std::cout << "\n Tensor after reshape\n";
    for (auto dim : t1.getShape()) {
        std::cout << dim << " ";
    }

    auto t3 = Tensor::createRandTensor({30, 20, 40, 30}, InitType::Xavier);

    t3.reshape({600, 1200});
    std::cout << "\n";
    for (auto dim : t3.getShape()) {
        std::cout << dim << " ";
    }
}
