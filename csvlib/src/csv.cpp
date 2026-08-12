#include <csvlib/csv.h>
#include <fstream>
#include <stdexcept>
RawCSVData CSVParser::readCSV(const std::string& path, char delim) {
    RawCSVData raw;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open file " + path);
    }

    std::string line;
    if (!std::getline(file, line)) {
        return raw; // empty file
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    for (auto& field : splitLine(line, delim)) {
        raw.headers.emplace_back(trim(std::move(field)));
    }
    raw.num_cols = raw.headers.size();
    raw.columns.resize(raw.num_cols);

    size_t row = 0;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        auto fields = splitLine(line, delim);
        for (size_t col = 0; col < raw.num_cols; col++) {
            raw.columns[col].emplace_back(col < fields.size() ? std::move(fields[col])
                                                              : std::string{});
        }
        row++;
    }
    raw.num_rows = row;

    raw.column_index.reserve(raw.num_cols);
    for (size_t i = 0; i < raw.headers.size(); i++) {
        raw.column_index.emplace(raw.headers[i], i);
    }
    return raw;
}

std::span<const float> CSVParser::getColumnData(const CSVData& csv, const std::string& feature) {
    size_t col = csv.feature_index.at(feature);
    return std::span<const float>(&csv.feature_data[col * csv.num_rows], csv.num_rows);
}

std::vector<std::string> CSVParser::splitLine(std::string_view line, char delim) {
    std::vector<std::string> fields;
    size_t start = 0;
    size_t len = line.size();
    for (size_t i = 0; i <= len; i++) {
        if (i == len || line[i] == delim) {
            fields.emplace_back(line.substr(start, i - start));
            start = i + 1;
        }
    }
    return fields;
}

std::string CSVParser::trim(std::string field) {
    while (!field.empty() && (field.back() == '\r' || field.back() == '\n' || field.back() == ' ' ||
                              field.back() == '\t')) {
        field.pop_back();
    }
    return field;
}
