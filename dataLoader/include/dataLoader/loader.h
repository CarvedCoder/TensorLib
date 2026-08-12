#ifndef LOADER_H
#define LOADER_H

#include <csvlib/csv.h>
#include <tensorlib/tensor.h>
#include <vector>

namespace DataLoader {
Tensor toTensor(const CSVData& csv_data, const std::string& feature);
Tensor toTensor(std::vector<float>& data);
} // namespace DataLoader

#endif // !LOADER_H
