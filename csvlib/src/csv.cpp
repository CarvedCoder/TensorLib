#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <csvlib/csv.h>
#include <fstream>
#include <iosfwd>
#include <limits>
#include <string>
#include <vector>
CSVData CSVParser::readCSV(const std::string &path, const char delim = ',') {
    CSVData parsed_data;
    std::ifstream file(path);
    std::string line;
    getFeatures(file, line, parsed_data.features, delim, parsed_data);
    parsed_data.num_rows = 1024;
    getFeaturesData(file, line, parsed_data.feature_data, parsed_data.num_rows,
                    parsed_data.num_cols, delim);
    parsed_data.feature_index.clear();
    parsed_data.feature_index.reserve(parsed_data.features.size());
    for (size_t i = 0; i < parsed_data.features.size(); i++) {
        parsed_data.feature_index.emplace(parsed_data.features[i], i);
    }
    return parsed_data;
}

const float *CSVParser::getColumn(const CSVData &csv,
                                  const std::string &feature) {
    size_t col = csv.feature_index.at(feature);
    return &csv.feature_data[col * csv.num_rows];
}

void CSVParser::getFeatures(std::ifstream &file, std::string &line,
                            std::vector<std::string> &header, const char delim,
                            CSVData &csv_data) {
    csv_data.num_cols = 0;
    std::getline(file, line);
    header.clear();
    size_t start = 0;
    size_t len = line.size();
    for (size_t i = 0; i <= len; i++) {
        if (i == len || line[i] == delim) {
            std::string feature = line.substr(start, i - start);
            while (!feature.empty() &&
                   (feature.back() == '\r' || feature.back() == '\n' ||
                    feature.back() == ' ' || feature.back() == '\t')) {
                feature.pop_back();
            }
            header.emplace_back(feature);
            start = i + 1;
            csv_data.num_cols++;
        }
    }
}

void CSVParser::getFeaturesData(std::ifstream &file, std::string &line,
                                std::vector<float> &data, size_t &num_rows,
                                size_t num_cols, const char delim) {
    size_t expected_rows = num_rows;
    size_t row = 0;
    data.resize(num_cols * num_rows);
    while (std::getline(file, line)) {
        size_t col = 0;
        size_t start = 0;
        size_t len = line.size();
        if (row == expected_rows) {
            expected_rows *= 2;
            data.resize(num_cols * expected_rows);
        }
        for (size_t i = 0; i <= len; i++) {
            float value = 0.0f;
            if (i == len || line[i] == delim) {
                if (i == start) {
                    value = std::numeric_limits<float>::quiet_NaN();
                } else {
                    std::from_chars((line.data() + start), (line.data() + i),
                                    value);
                }
                data[col * expected_rows + row] = value;
                start = i + 1;
                col++;
            }
        }

        while (col < num_cols) {
            data[col * expected_rows + row] =
                std::numeric_limits<float>::quiet_NaN();
            col++;
        }

        row++;
    }
    num_rows = row;
    data.resize(num_cols * expected_rows);
    if (num_rows != expected_rows) {
        for (size_t col = 0; col < num_cols; col++) {
            for (size_t row = 0; row < num_rows; row++) {
                data[col * num_rows + row] = data[col * expected_rows + row];
            }
        }
    }
}
