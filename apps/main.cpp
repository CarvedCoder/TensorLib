#include <iostream>
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>
int main() {
    auto t1 = Tensor::createTensor({2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 9.0f}, {2, 3});
    auto t2 = Tensor::createTensor({3.0f, 5.0f, 4.0f, 5.0f, 6.0f, 10.0f}, {3, 3});
    auto t3 = TensorOps::matmul(t1, t2);
    for (const auto elem : t3.view()) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}
