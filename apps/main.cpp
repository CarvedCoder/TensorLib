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

    std::string DataPath = "../../Datasets/Salary_dataset_large.csv";
    CSVData csv_data = CSVParser::readCSV(DataPath);

    auto x = Data::toTensor(csv_data, "YearsExperience");
    auto y = Data::toTensor(csv_data, "Salary");

    auto norm_x = minMaxScaler(x);
    auto norm_y = minMaxScaler(y);

    auto w = Tensor::createRandTensor({1});
    auto b = Tensor::createRandTensor({1});

    float lr = 0.01f;
    size_t n = norm_x.getTotalSize();

    for (int epoch = 0; epoch < 10000; epoch++) {

        auto y_pred = w * norm_x + b;
        auto diff = y_pred - norm_y;

        auto grad_w = diff * norm_x;
        auto& grad_b = diff;

        float dw = 0.0f, db = 0.0f;

        auto gw = grad_w.getDataPtr();
        auto gb = grad_b.getDataPtr();

        for (size_t i = 0; i < n; i++) {
            dw += gw[i];
            db += gb[i];
        }

        // 🔥 ADD IT HERE
        dw = (2.0f / n) * dw;
        db = (2.0f / n) * db;

        // Update
        float w_val = w(0);
        float b_val = b(0);

        w_val -= lr * dw;
        b_val -= lr * db;

        w.setDataElem(0, w_val);
        b.setDataElem(0, b_val);

        if (epoch % 100 == 0) {
            auto loss = TensorOps::calcCost(norm_y, y_pred, LossType::MSE);
            std::cout << "Epoch " << epoch << " Loss: " << loss << "\n";
        }
    }
    auto y_minMax = y.getMinMax();
    float total_error = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float xi = norm_x(i);
        float yi = norm_y(i);

        float pred = w.getDataPtr()[0] * xi + b.getDataPtr()[0];

        float real_pred = pred * (y_minMax.max - y_minMax.min) + y_minMax.min;
        float real_y = y(i);

        total_error += std::abs(real_pred - real_y);
    }

    std::cout << "Avg Error: " << total_error / n << "\n";
}
