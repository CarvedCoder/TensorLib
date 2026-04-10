#include <tensorlib/tensor.h>

class Scaler {
  private:
  public:
    Tensor minMaxScaler(const Tensor& t) {
        auto minMax = std::ranges::minmax(t.view());
        auto min = minMax.min;
        auto max = minMax.max;
        auto t_data = t.getDataPtr();
        const size_t t_size = t.getTotalSize();
        auto result = Tensor::createZeros(t.getShape());
        auto mutData_result = result.getMutableDataPtr();
        for (size_t i = 0; i < t_size; i++) {
            mutData_result[i] = (t_data[i] - min) / (max - min);
        }
        return result;
    }
};
