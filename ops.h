#ifndef OPS_H
#define OPS_H

#include "tensor.h"

namespace TensorOps{

    Tensor::Tensorptr operator+(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2);
    Tensor::Tensorptr operator-(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2);
    Tensor::Tensorptr operator*(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2);
}
#endif // OPS_H
    
