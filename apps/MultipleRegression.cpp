#include "tensorlib/tensor/tensor.h"
#include <csvlib/csv.h>
#include <csvlib/dataframe.h>
#include <dataLoader/loader.h>
#include <filesystem>
#include <print>
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>
#include <vector>
float derv_w(Tensor& X, Tensor& y, Tensor& y_hat) {
    float res = 0;
    uint n = X.getTotalSize();
    for (int i = 0; i < X.getTotalSize(); i++) {
        res += (y_hat(i) - y(i)) * X(i);
    }
    return res / n;
}

float derv_b(Tensor& X, Tensor& y, Tensor& y_hat) {
    float res = 0;
    uint n = X.getTotalSize();
    for (int i = 0; i < X.getTotalSize(); i++) {
        res += y_hat(i) - y(i);
    }
    return res / n;
}

void LinearRegression(Tensor& X, Tensor& y, Tensor& w, Tensor& b, Tensor& y_hat, float alpha,
                      uint epochs) {
    auto X_ptr = X.getDataPtr();
    auto y_ptr = y.getDataPtr();
    auto y_hat_ptr = y_hat.getMutableDataPtr();

    auto w_mut_ptr = w.getMutableDataPtr();
    auto b_mut_ptr = b.getMutableDataPtr();
    for (size_t i = 0; i < epochs; i++) {
        for (size_t j = 0; j < X.getTotalSize(); j++) {
            y_hat_ptr[j] = w(0) * X_ptr[j] + b(0);
        }
        float w_res = derv_w(X, y, y_hat);
        float b_res = derv_b(X, y, y_hat);
        w_mut_ptr[0] = w(0) - alpha * w_res;
        b_mut_ptr[0] = b(0) - alpha * b_res;

        if (i % 100 == 0) {
            std::println("Loss : {}", TensorOps::calcCost(y_hat, y, LossType::MSE));
        }
    }
}

int main(int argc, char* argv[]) {
    std::filesystem::path data_path = "../csv/Housing.csv";
    DataFrame df(CSVParser::readCSV(data_path));
    df.head();
    df.encodeBinary(
          {"mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"},
          "yes", "no")
        .oneHot("furnishingstatus");

    auto csv_data = df.toNumeric();
    std::vector<Tensor> cols;
    cols.reserve(csv_data.num_cols - 1);
    for (const auto& feature : csv_data.features) {
        if (feature != "price")
            cols.emplace_back(DataLoader::toTensor(csv_data, feature));
    }

    Tensor X = Tensor::concat(cols);
    Tensor y = DataLoader::toTensor(csv_data, "price");
    Tensor w = Tensor::createRandTensor({1});
    Tensor b = Tensor::createRandTensor({1});
    Tensor y_hat = Tensor::createZeros({X.getShape()});
    int count = 0;
    auto view_data = X.view(0, Axis::Col);
    for (size_t i = 0; i < view_data.extent(0); ++i) {
        std::print("{} ", view_data[i]);
        count++;
    }
}
