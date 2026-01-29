#include "tensorlib/ops/ops.h"
#include "tensorlib/tensor/tensor.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>

namespace TensorOps {

float sigmoid(const float input_data) {
    return 1 / (1 + std::exp(-input_data));
}

float relu(const float input_data) {
    return input_data > 0 ? input_data : 0.0f;
}

float leakyRelu(const float input_data) {
    return input_data > 0 ? input_data : 0.01f;
}

float m_tanh(const float input_data) { return std::tanh(input_data); }

bool sameShape(const std::span<const size_t> &t1_shape,
               const std::span<const size_t> &t2_shape) {
    return t1_shape.size() == t2_shape.size() &&
           std::equal(t1_shape.begin(), t1_shape.end(), t2_shape.begin());
}

BroadcastInfo computeBroadcast(const Tensor &t1, const Tensor &t2) {
    BroadcastInfo info{};
    auto shape_t1 = t1.getShape();
    auto shape_t2 = t2.getShape();
    size_t rank_t1 = shape_t1.size();
    size_t rank_t2 = shape_t2.size();
    size_t max_rank = std::max(rank_t1, rank_t2);
    size_t offset_t1 = max_rank - rank_t1;
    size_t offset_t2 = max_rank - rank_t2;
    info.b_totalSize = 1;
    for (size_t i = 0; i < max_rank; i++) {
        size_t dim_t1 = (i < offset_t1) ? 1 : shape_t1[i - (offset_t1)];
        size_t dim_t2 = (i < offset_t2) ? 1 : shape_t2[i - (offset_t2)];
        if (dim_t1 != dim_t2 && dim_t1 != 1 && dim_t2 != 1) {
            info.Possible = false;
            return info;
        }
        info.b_Shape[i] = std::max(dim_t1, dim_t2);
        info.b_totalSize *= info.b_Shape[i];
    }
    info.Possible = true;
    info.b_ShapeRank = max_rank;
    auto strides_t1 = t1.getStrides();
    auto strides_t2 = t2.getStrides();
    for (size_t i = 0; i < max_rank; i++) {
        if (i < offset_t1) {
            info.b_Stride_t1[i] = 0;
        } else {
            size_t a = i - offset_t1;
            info.b_Stride_t1[i] = shape_t1[a] == 1 ? 0 : strides_t1[a];
        }
        if (i < offset_t2) {
            info.b_Stride_t2[i] = 0;
        } else {
            size_t a = i - offset_t2;
            info.b_Stride_t2[i] = shape_t2[a] == 1 ? 0 : strides_t2[a];
        }
    }
    return info;
}

Tensor operator+(const Tensor &t1, const Tensor &t2) {
    return binaryKernel(t1, t2, std::plus<float>{});
}

Tensor operator-(const Tensor &t1, const Tensor &t2) {
    return binaryKernel(t1, t2, std::minus<float>{});
}

Tensor operator*(const Tensor &t1, const Tensor &t2) {
    return binaryKernel(t1, t2, std::multiplies<float>{});
}

Tensor operator/(const Tensor &t1, const Tensor &t2) {
    return binaryKernel(t1, t2, std::divides<float>{});
}

Tensor operator*(const Tensor &lhs, float rhs) {
    return lhs * Tensor::createScalar(rhs);
}

Tensor transpose2D(const Tensor &t) {
    const size_t rank = t.getRank();
    const auto shape = t.getShape();
    if (rank >= 3 || rank == 1)
        throw std::invalid_argument("Expected 2D rank tensor for 'transpose2D' "
                                    "but got >2D rank tensor");
    size_t rows = shape[0];
    size_t cols = shape[1];
    auto T = Tensor::createZeros({cols, rows});
    auto &strides_T = T.getStrides();
    auto mutData_T = T.getMutableDataPtr();
    auto data_t = t.getDataPtr();
    auto &strides_t = t.getStrides();
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            mutData_T[j * strides_T[0] + i * strides_T[1]] =
                data_t[i * strides_t[0] + j * strides_t[1]];
        }
    }
    return T;
}

float calcCost(const Tensor &t1, const Tensor &t2, const LossType mode) {
    if (!sameShape(t1.getShape(), t2.getShape()))
        throw std::invalid_argument("Tensor shape not same for calcCost");
    float result = 0;
    const size_t size = t1.getTotalSize();
    auto data_t1 = t1.getDataPtr();
    auto data_t2 = t2.getDataPtr();
    for (size_t i = 0; i < size; i++) {
        const float diff = data_t1[i] - data_t2[i];
        result += diff * diff;
    }
    return mode == LossType::SSE ? result : result / static_cast<float>(size);
}

Tensor matmul(const Tensor &t1, const Tensor &t2) {
    const auto r1 = t1.getRank();
    if (r1 != t2.getRank())
        throw std::invalid_argument("Rank is not the same for matmul");
    if (r1 > 2)
        throw std::invalid_argument("Rank input is greater than 2 in matmul");
    const auto s1 = t1.getShape();
    const auto s2 = t2.getShape();
    const size_t m = s1[0];
    const size_t K = s1[1];
    const size_t n = s2[1];
    auto &strides_t1 = t1.getStrides();
    auto data_t1 = t1.getDataPtr();
    auto data_t2 = t2.getDataPtr();
    auto &strides_t2 = t2.getStrides();
    if (K != s2[0])
        throw std::invalid_argument("Inner ranks in matmul aren't the same");
    auto result = Tensor::createZeros({m, n});
    auto &strides_result = result.getStrides();
    auto mutData_result = result.getMutableDataPtr();
    for (size_t i = 0; i < m; i++) {
        for (size_t k = 0; k < K; k++) {
            const float a_ik = data_t1[i * strides_t1[0] + k * strides_t1[1]];
            for (size_t j = 0; j < n; j++) {
                mutData_result[i * strides_result[0] + j * strides_result[1]] +=
                    a_ik * data_t2[k * strides_t2[0] + j * strides_t2[1]];
            }
        }
    }
    return result;
}
} // namespace TensorOps
