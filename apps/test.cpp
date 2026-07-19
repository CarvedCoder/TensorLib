#include <print>
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>
int main(int argc, char* argv[]) {
    auto t1 = Tensor::createTensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4});
    auto softmax_t1 = TensorOps::softmax(t1);
    for (const auto elem : softmax_t1.view()) {
        std::print("{} ", elem);
    }
}
