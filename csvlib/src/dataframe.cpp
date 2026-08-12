#include <csvlib/dataframe.h>
#include <limits>
#include <print>
#include <set>
#include <stdexcept>
#include <vector>
DataFrame::DataFrame(RawCSVData data) : data_(std::move(data)) {}

size_t DataFrame::indexOf(const std::string& column) const {
    auto it = data_.column_index.find(column);
    if (it == data_.column_index.end()) {
        throw std::out_of_range("no such column: " + column);
    }
    return it->second;
}

void DataFrame::rebuildIndex() {
    data_.column_index.clear();
    data_.column_index.reserve(data_.headers.size());
    for (size_t i = 0; i < data_.headers.size(); i++) {
        data_.column_index.emplace(data_.headers[i], i);
    }
}

const std::vector<std::string>& DataFrame::column(const std::string& name) const {
    return data_.columns[indexOf(name)];
}

DataFrame& DataFrame::drop(const std::string& column) {
    size_t idx = indexOf(column);
    data_.headers.erase(data_.headers.begin() + idx);
    data_.columns.erase(data_.columns.begin() + idx);
    data_.num_cols--;
    rebuildIndex();
    return *this;
}

DataFrame& DataFrame::mapColumn(const std::string& column,
                                const std::function<std::string(const std::string&)>& fn) {
    auto& col = data_.columns[indexOf(column)];
    for (auto& field : col) {
        field = fn(field);
    }
    return *this;
}

DataFrame& DataFrame::encodeBinary(const std::vector<std::string>& columns,
                                   const std::string& true_value, const std::string& false_value) {
    for (const auto& column : columns) {
        encodeBinary(column, true_value, false_value);
    }
    return *this;
}

DataFrame& DataFrame::encodeBinary(const std::string& column, const std::string& true_value,
                                   const std::string& false_value) {
    return mapColumn(column, [&true_value, &false_value, &column](const std::string& field) {
        if (field == true_value) {
            return std::string("1");
        }
        if (field == false_value) {
            return std::string("0");
        }
        throw std::invalid_argument("encodeBinary: unexpected value '" + field + "' in column '" +
                                    column + "'");
    });
}

DataFrame& DataFrame::oneHot(const std::string& column) {
    size_t idx = indexOf(column);
    std::vector<std::string> source = data_.columns[idx];

    std::set<std::string> unique_values(source.begin(), source.end());

    for (const auto& value : unique_values) {
        std::vector<std::string> new_col;
        new_col.reserve(source.size());
        for (const auto& field : source) {
            new_col.emplace_back(field == value ? "1" : "0");
        }
        data_.headers.emplace_back(column + "_" + value);
        data_.columns.emplace_back(std::move(new_col));
    }
    data_.num_cols += unique_values.size();

    drop(column);
    return *this;
}

CSVData DataFrame::toNumeric() const {
    CSVData csv;
    csv.features = data_.headers;
    csv.num_cols = data_.num_cols;
    csv.num_rows = data_.num_rows;
    csv.feature_data.resize(csv.num_cols * csv.num_rows);

    for (size_t col = 0; col < csv.num_cols; col++) {
        const auto& source = data_.columns[col];
        for (size_t row = 0; row < csv.num_rows; row++) {
            const std::string& field = source[row];
            float value;
            if (field.empty()) {
                value = std::numeric_limits<float>::quiet_NaN();
            } else {
                auto res = CSVParser::parseNum<float>(field);
                if (!res) {
                    throw std::invalid_argument("column '" + data_.headers[col] + "' row " +
                                                std::to_string(row) + ": not numeric ('" + field +
                                                "') — drop() or encode it before toNumeric()");
                }
                value = *res;
            }
            csv.feature_data[col * csv.num_rows + row] = value;
        }
    }

    csv.feature_index.reserve(csv.features.size());
    for (size_t i = 0; i < csv.features.size(); i++) {
        csv.feature_index.emplace(csv.features[i], i);
    }
    return csv;
}

void DataFrame::head(size_t n) const {
    size_t rows_to_show = std::min(n, data_.num_rows);

    std::vector<size_t> widths(data_.num_cols);
    for (size_t col = 0; col < data_.num_cols; col++) {
        size_t w = data_.headers[col].size();
        for (size_t row = 0; row < rows_to_show; row++) {
            w = std::max(w, data_.columns[col][row].size());
        }
        widths[col] = w;
    }

    for (size_t col = 0; col < data_.num_cols; col++) {
        std::print("{:<{}}  ", data_.headers[col], widths[col]);
    }
    std::println("");

    for (size_t col = 0; col < data_.num_cols; col++) {
        std::print("{:-<{}}  ", "", widths[col]);
    }
    std::println("");

    for (size_t row = 0; row < rows_to_show; row++) {
        for (size_t col = 0; col < data_.num_cols; col++) {
            std::print("{:<{}}  ", data_.columns[col][row], widths[col]);
        }
        std::println("");
    }

    if (data_.num_rows > rows_to_show) {
        std::println("[{} rows x {} columns, showing first {}]", data_.num_rows, data_.num_cols,
                     rows_to_show);
    } else {
        std::println("[{} rows x {} columns]", data_.num_rows, data_.num_cols);
    }
}
