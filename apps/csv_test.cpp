#include "tensorlib/tensor/tensor.h"
#include <csvlib/csv.h>
#include <csvlib/dataframe.h>
#include <dataLoader/loader.h>
#include <filesystem>
#include <print>
#include <tensorlib/tensor.h>
int main() {
    std::filesystem::path data_path = "../csv/Housing.csv";
    DataFrame df(CSVParser::readCSV(data_path));
    df.head();
    df.encodeBinary(
          {"mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"},
          "yes", "no")
        .oneHot("furnishingstatus");
    df.head();
    CSVData csv_data = df.toNumeric();
    std::vector<Tensor> cols;
    cols.reserve(csv_data.num_cols - 1);
    for (const auto& feature : csv_data.features) {
        if (feature != "price")
            cols.emplace_back(DataLoader::toTensor(csv_data, feature));
    }

    Tensor X = Tensor::concat(cols);
    auto row_data = X.view(0, Axis::Row);
    auto col_data = X.view(0, Axis::Col);
    std::print("Row data : ");
    for (size_t i = 0; i < row_data.extent(0); ++i) {
        std::print("{} ", row_data[i]);
    }
    std::println("");
    std::print("Col data : ");
    for (size_t i = 0; i < col_data.extent(0); ++i) {
        std::print("{} ", col_data[i]);
    }
    std::println("");
}
