#include "tensorlib/ops/ops.h"
#include "tensorlib/tensor/tensor.h"
#include "tensorlib/tensor_RNG.h"
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>

void saveCSV(const Tensor& t, const std::string& filename) {
    if (!std::filesystem::exists("../csv/")) {
        std::filesystem::create_directory("../csv/");
    }
    auto shape = t.getShape();
    if (shape.size() != 2)
        throw std::runtime_error("Only supports 2D tensors");
    std::ofstream out(filename);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            out << t(i, j);
            if (j + 1 != shape[1])
                out << ",";
        }
        out << "\n";
    }
}
int main(int argc, char* argv[]) {
    TensorRNG::setSeed(42);
    constexpr size_t seq_len = 4;
    constexpr size_t d_model = 8;
    constexpr size_t d_k = d_model;

    Tensor W_q = Tensor::createRandTensor({d_model, d_k}, InitType::Xavier);
    Tensor W_k = Tensor::createRandTensor({d_model, d_k}, InitType::Xavier);
    Tensor W_v = Tensor::createRandTensor({d_model, d_k}, InitType::Xavier);
    Tensor X = Tensor::createRandTensor({seq_len, d_model}, InitType::Xavier);

    saveCSV(W_q, "../csv/Wq.csv");
    saveCSV(W_k, "../csv/Wk.csv");
    saveCSV(W_v, "../csv/Wv.csv");
    saveCSV(X, "../csv/X.csv");
    Tensor Q = TensorOps::matmul(X, W_q);
    Tensor K = TensorOps::matmul(X, W_k);
    Tensor V = TensorOps::matmul(X, W_v);

    Tensor scores = TensorOps::matmul(Q, TensorOps::transpose2D(K));
    auto scores_scaled = scores.getMutableDataPtr();
    for (size_t i = 0; i < scores.getTotalSize(); i++) {
        scores_scaled[i] /= std::sqrt(d_k);
    }

    Tensor final_scores = TensorOps::softmax(scores);

    Tensor output = TensorOps::matmul(final_scores, V);

    for (const auto elem : output.view()) {
        std::print("{} ", elem);
    }
    return 0;
}
