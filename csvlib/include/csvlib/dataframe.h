#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <csvlib/csv.h>
#include <functional>
#include <string>
#include <vector>

class DataFrame {
  public:
    explicit DataFrame(RawCSVData data);

    DataFrame& drop(const std::string& column);
    DataFrame& mapColumn(const std::string& column,
                         const std::function<std::string(const std::string&)>& fn);
    DataFrame& encodeBinary(const std::string& column, const std::string& true_value,
                            const std::string& false_value);
    DataFrame& encodeBinary(const std::vector<std::string>& columns, const std::string& true_value,
                            const std::string& false_value);
    DataFrame& oneHot(const std::string& column);

    [[nodiscard]] const std::vector<std::string>& column(const std::string& name) const;
    [[nodiscard]] const std::vector<std::string>& headers() const noexcept {
        return data_.headers;
    }
    [[nodiscard]] size_t numRows() const noexcept {
        return data_.num_rows;
    }
    [[nodiscard]] size_t numCols() const noexcept {
        return data_.num_cols;
    }

    void head(size_t n = 5) const;

    [[nodiscard]] CSVData toNumeric() const;

  private:
    RawCSVData data_;

    [[nodiscard]] size_t indexOf(const std::string& column) const;
    void rebuildIndex();
};

#endif // !DATAFRAME_H
