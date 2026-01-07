#ifndef OPS_H
#define OPS_H

#include <tensor/tensor.h>
enum LossType { SSE, MSE };
namespace TensorOps {
Tensor operator+(const Tensor &t1, const Tensor &t2);
Tensor operator-(const Tensor &t1, const Tensor &t2);
Tensor operator*(const Tensor &t1, const Tensor &t2);
Tensor operator*(const Tensor, size_t rhs);
Tensor matmul(const Tensor &t1, const Tensor &t2);
Tensor transpose2D(const Tensor &t);
float sigmoid(float input_data);
float relu(float input_data);
float leakyRelu(float input_data);
float m_tanh(float input_data);
float calcCost(const Tensor &t1, const Tensor &t2, LossType mode = SSE);
} // namespace TensorOps
#endif // OPS_H
