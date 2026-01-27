#include <tensorlib/tensor_RNG.h>

std::mt19937 &TensorRNG::engine() {
    static std::mt19937 gen(5489u);
    return gen;
}

void TensorRNG::setSeed(uint32_t seed) { engine().seed(seed); }
