#include <csvlib/csv.h>
#include <dataLoader/loader.h>
#include <tensorlib/tensor.h>
#include <vector>

namespace Data {
Tensor toTensor(const CSVData& csv_data, const std::string& feature) {
    const auto feature_data = CSVParser::getColumnData(csv_data, feature);
    auto resultTensor = Tensor::createTensor(feature_data, {feature_data.size()}, false);
    return resultTensor;
}
Tensor toTensor(const std::vector<float>& data) {
    auto resultTensor = Tensor::createTensor(data, {data.size()}, false);
    return resultTensor;
}
} // namespace Data
