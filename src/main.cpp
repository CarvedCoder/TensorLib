#include <iostream>
#include <tensor/tensor.h>
int main() {
    const auto t = Tensor::createRandTensor({3, 2}, He);
    size_t size = t.getTotalSize();
    for (size_t i = 0; i < size; i++) {
        std::cout << (t)(i) << " ";
    }
    return 0;
}
