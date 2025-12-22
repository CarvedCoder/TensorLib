#ifndef OPS_H
#define OPS_H

#include "tensor.h"
enum LossType {SSE,MSE};
namespace TensorOps{

    Tensor::Tensorptr operator+(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2);
    Tensor::Tensorptr operator-(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2);
    Tensor::Tensorptr operator*(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2);
    Tensor::Tensorptr matmul(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2);
    Tensor::Tensorptr transpose2D(const std::shared_ptr<Tensor>&t);
    float sigmoid(float input_data);
    float relu(float input_data);
    float leakyRelu(float input_data);
    float m_tanh(float input_data);
    float calcCost(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2,LossType mode = SSE);
}
#endif // OPS_H
    
