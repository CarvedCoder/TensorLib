#include "tensorlib/autograd/autograd.h"
#include "tensorlib/tensor/tensor.h"
#include <print>
#include <tensorlib/ops.h>
int main(int argc, char* argv[]) {
    using namespace TensorOps;
    Tensor a = Tensor::createScalar(2.0f);
    a.getImpl()->m_require_grad = true;
    a.getImpl()->ensureGrad();
    Tensor b = Tensor::createScalar(3.0f);
    b.getImpl()->m_require_grad = true;
    b.getImpl()->ensureGrad();
    Tensor c = a + b;
    Tensor d = c + a;
    d.backward();
    std::println("{}", a.getGradPtr()[0]);
    std::println("{}", b.getGradPtr()[0]);

    return 0;
}
