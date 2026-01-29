#ifndef CSV_H
#define CSV_H

#include <string>
#include <unordered_map>
#include <vector>

namespace csvlib {
struct CSVData {
    std::vector<std::string> features;
    std::vector<std::vector<float>> feature_data;
    std::unordered_map<std::string, size_t> feature_index;
    size_t num_cols = 0;
};

CSVData readCSV(const std::string &path);
} // namespace csvlib

#endif // CSV_H
