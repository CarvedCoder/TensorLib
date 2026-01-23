#ifndef OPS_H
#define OPS_H

#include <tensorlib/tensor/tensor.h>
enum class LossType { SSE, MSE };
namespace TensorOps {
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
bool canBroadcast(const Tensor &t1, const Tensor &t2);
bool sameShape(const std::span<const size_t> &t1_shape,
               const std::span<const size_t> &t2_shape);
} // namespace TensorOps
#endif // OPS_H
