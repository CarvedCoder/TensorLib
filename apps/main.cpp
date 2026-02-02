#include <csvlib/csv.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <tensorlib/tensor.h>
int main() {
    CSVData data =
        CSVParser::readCSV("../../../Dataset/Plant_1_Generation_Data.csv", ',');
    for (const auto &feature : data.features) {
        std::cout << feature << "\n";
    }

    for (const auto &f : data.features) {
        std::cout << "[" << f << "] len=" << f.size() << "\n";
    }

    const float *col_data = CSVParser::getColumn(data, "TOTAL_YIELD");

    std::cout << std::fixed << std::setprecision(2);

    for (size_t r = 0; r < data.num_rows; ++r) {
        float v = *(col_data + r);
        if (std::isnan(v))
            std::cout << "NaN ";
        else
            std::cout << v << " ";
    }

    std::cout << "\n";
}
