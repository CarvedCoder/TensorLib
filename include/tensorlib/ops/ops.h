#ifndef OPS_H
#define OPS_H
#include <array>
#include <tensorlib/tensor/tensor.h>
enum class LossType { SSE, MSE };
struct BroadcastInfo {
    bool Possible = false;
    size_t b_ShapeRank;
    size_t b_totalSize;
    std::array<size_t, MAX_RANK> b_Shape;
    std::array<size_t, MAX_RANK> b_Stride_t1;
    std::array<size_t, MAX_RANK> b_Stride_t2;
};
namespace TensorOps {

BroadcastInfo computeBroadcast(const Tensor &t1, const Tensor &t2);

template <typename Op>
Tensor binaryKernel(const Tensor &t1, const Tensor &t2, Op op) {
    auto info = computeBroadcast(t1, t2);
    if (!info.Possible)
        throw std::invalid_argument("Can't broadcast these two tensors");

    // collapseBroadcastDims(info);

    Tensor result = Tensor::createZeros(info.b_Shape);
    auto out = result.getMutableDataPtr();
    auto d1 = t1.getDataPtr();
    auto d2 = t2.getDataPtr();

    size_t n = info.b_totalSize;

    if (info.b_ShapeRank == 1) {
        size_t s1 = info.b_Stride_t1[0];
        size_t s2 = info.b_Stride_t2[0];
        for (size_t i = 0; i < n; ++i)
            out[i] = op(d1[i * s1], d2[i * s2]);
        return result;
    }

    std::array<size_t, MAX_RANK> idx{};
    for (size_t linear = 0; linear < n; ++linear) {
        size_t off1 = 0, off2 = 0;

        for (size_t d = 0; d < info.b_ShapeRank; ++d) {
            off1 += info.b_Stride_t1[d] * idx[d];
            off2 += info.b_Stride_t2[d] * idx[d];
        }

        out[linear] = op(d1[off1], d2[off2]);

        for (int d = static_cast<int>(info.b_ShapeRank) - 1; d >= 0; --d) {
            if (++idx[static_cast<size_t>(d)] <
                info.b_Shape[static_cast<size_t>(d)])
                break;
            idx[static_cast<size_t>(d)] = 0;
        }
    }

    return result;
}

Tensor operator+(const Tensor &t1, const Tensor &t2);
Tensor operator-(const Tensor &t1, const Tensor &t2);
Tensor operator*(const Tensor &t1, const Tensor &t2);
Tensor operator*(const Tensor &lhs, float rhs);
Tensor matmul(const Tensor &t1, const Tensor &t2);
Tensor transpose2D(const Tensor &t);
float calcCost(const Tensor &t1, const Tensor &t2,
               LossType mode = LossType::SSE);
float sigmoid(float input_data);
float relu(float input_data);
float leakyRelu(float input_data);
float m_tanh(float input_data);

bool sameShape(const std::span<const size_t> &t1_shape,
               const std::span<const size_t> &t2_shape);
} // namespace TensorOps
#endif // OPS_H
