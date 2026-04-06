#include <csvlib/csv.h>
#include <dataLoader/loader.h>
#include <iostream>
#include <random>
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>
#include <tensorlib/tensor_RNG.h>
int main() {
    using namespace TensorOps;
    TensorRNG::setSeed(std::random_device{}());
    std::string DataPath = "../../Datasets/Salary_dataset.csv";
    CSVData csv_data = CSVParser::readCSV(DataPath);
    auto x = Data::toTensor(csv_data, "YearsExperience");
    auto y = Data::toTensor(csv_data, "Salary");
    float mean_x = 0, std_x = 0;
    for (size_t i = 0; i < x.getTotalSize(); i++)
        mean_x += x(i);
    mean_x /= x.getTotalSize();

    for (size_t i = 0; i < x.getTotalSize(); i++)
        std_x += (x(i) - mean_x) * (x(i) - mean_x);
    std_x = std::sqrt(std_x / x.getTotalSize());

    for (size_t i = 0; i < x.getTotalSize(); i++) {
        x.setDataElem(i, (x(i) - mean_x) / std_x);
    }

    float mean_y = 0, std_y = 0;

    for (size_t i = 0; i < y.getTotalSize(); i++)
        mean_y += y(i);
    mean_y /= y.getTotalSize();

    for (size_t i = 0; i < y.getTotalSize(); i++)
        std_y += (y(i) - mean_y) * (y(i) - mean_y);
    std_y = std::sqrt(std_y / y.getTotalSize());

    for (size_t i = 0; i < y.getTotalSize(); i++) {
        y.setDataElem(i, (y(i) - mean_y) / std_y);
    }
    const auto tensor_shape = csv_data.num_rows;
    auto w = Tensor::createRandTensor({1});
    auto b = Tensor::createRandTensor({1});
    auto y_pred = w * x + b;
    auto error = TensorOps::calcCost(y, y_pred, LossType::MSE);
    std::cout << "epoch 0 : " << error << "\n";

    float lr = 0.01f;

    for (int epoch = 0; epoch < 10000; epoch++) {

        auto y_pred = w * x + b;

        auto diff = y_pred - y;

        auto grad_w = diff * x;
        auto& grad_b = diff;

        float dw = 0.0f, db = 0.0f;

        auto gw = grad_w.getDataPtr();
        auto gb = grad_b.getDataPtr();

        size_t n = x.getTotalSize();

        for (size_t i = 0; i < n; i++) {
            dw += gw[i];
            db += gb[i];
        }

        dw /= n;
        db /= n;

        float w_val = w(0);
        float b_val = b(0);

        w_val -= lr * dw;
        b_val -= lr * db;

        w.setDataElem(0, w_val);
        b.setDataElem(0, b_val);

        if (epoch % 100 == 0) {
            auto loss = TensorOps::calcCost(y, y_pred, LossType::MSE);
            std::cout << "Epoch " << epoch << " Loss: " << loss << "\n";
        }
    }

    float x_input = 1.4f;

    float x_norm = (x_input - mean_x) / std_x;

    float w_val = w(0);
    float b_val = b(0);

    float y_pred_norm = w_val * x_norm + b_val;

    float y_pred_real = y_pred_norm * std_y + mean_y;

    float y_actual = 46206;

    std::cout << "Predicted: " << y_pred_real << "\n";
    std::cout << "Actual   : " << y_actual << "\n";
    std::cout << "Error    : " << std::abs(y_pred_real - y_actual) << "\n";
}
