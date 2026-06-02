#ifndef CSV_H
#define CSV_H

#include <charconv>
#include <cstddef>
#include <expected>
#include <fstream>
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

class CSVParser {
  public:
    template <typename T>
    [[nodiscard]] static inline std::expected<T, std::errc> parseNum(std::string_view line,
                                                                     size_t start, size_t i) {
        T value{};
        auto [ptr, err] = std::from_chars((line.data() + start), (line.data() + i), value);

        if (err != std::errc{}) {
            return std::unexpected(err);
        }

        if (ptr != line.data() + i) {
            return std::unexpected(std::errc::invalid_argument);
        }
        return value;
    }
    static CSVData readCSV(const std::string& path, const char delim = ',');

    static std::span<const float> getColumnData(const CSVData& csv, const std::string& feature);

  private:
    static void getFeatures(std::ifstream& file, std::string& line,
                            std::vector<std::string>& header, const char delim, CSVData& csv_data);

    static void getFeaturesData(std::ifstream& file, std::string& line, std::vector<float>& data,
                                size_t& num_rows, size_t num_cols, const char delim);
};

#endif // !CSV_H
