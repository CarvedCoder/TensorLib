#ifndef CSV_H
#define CSV_H

#include <charconv>
#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

struct CSVData {
    std::vector<std::string> features;
    std::vector<float> feature_data;
    std::unordered_map<std::string, size_t> feature_index;
    size_t num_cols = 0;
    size_t num_rows = 0;
};

struct RawCSVData {
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> columns; // columns[col][row]
    std::unordered_map<std::string, size_t> column_index;
    size_t num_cols = 0;
    size_t num_rows = 0;
};

class CSVParser {
  public:
    template <typename T>
    [[nodiscard]] static inline std::expected<T, std::errc> parseNum(std::string_view field) {
        T value{};
        auto [ptr, err] = std::from_chars(field.data(), field.data() + field.size(), value);

        if (err != std::errc{}) {
            return std::unexpected(err);
        }
        if (ptr != field.data() + field.size()) {
            return std::unexpected(std::errc::invalid_argument);
        }
        return value;
    }

    static RawCSVData readCSV(const std::string& path, char delim = ',');

    static std::span<const float> getColumnData(const CSVData& csv, const std::string& feature);

  private:
    static std::vector<std::string> splitLine(std::string_view line, char delim);
    static std::string trim(std::string field);
};

#endif // !CSV_H
