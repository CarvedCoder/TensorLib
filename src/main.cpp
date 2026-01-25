#include "tensorlib/ops.h"
#include <iostream>
#include <tensorlib/tensor.h>
int main() {

    auto t1 = Tensor::createRandTensor({3, 1, 2});
    auto t2 = Tensor::createRandTensor({3, 2}, InitType::He);
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
    std::cout << "\nresult data for addition:-\n";
    for (auto v : result1.view()) {
        std::cout << v << " ";
    }
    std::cout << "\nresult data for multiplication :-\n";
    for (auto v : result2.view()) {
        std::cout << v << " ";
    }
}
