#ifndef CSV_H
#define CSV_H

#include <cstddef>
#include <fstream>
#include <string>
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
    static CSVData readCSV(const std::string &path, const char delim);

    static const float *getColumn(const CSVData &csv,
                                  const std::string &feature);

  private:
    static void getFeatures(std::ifstream &file, std::string &line,
                            std::vector<std::string> &header, const char delim,
                            CSVData &csv_data);

    static void getFeaturesData(std::ifstream &file, std::string &line,
                                std::vector<float> &data, size_t &num_rows,
                                size_t num_cols, const char delim);
};

#endif // !CSV_H
