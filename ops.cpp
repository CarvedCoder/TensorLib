#include "ops.h"
#include "tensor.h"

namespace TensorOps{
    Tensor::Tensorptr operator+(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2){
        if(t1->getShape() != t2->getShape()){
            throw std::invalid_argument("size of tensors don't match for add operator");
        }
        auto result = Tensor::createZeros(t1->getShape());
        if(t1->getShape().size() == 0){
            result->setDataElem(0, t1->getDataElem(0)+t2->getDataElem(0));
            return result;
        }
        
        for(size_t i = 0; i < t1->getTotalSize();i++){
            result->setDataElem(i,t1->getDataElem(i)+ t2->getDataElem(i));
        }
        return result;
    }
    
    Tensor::Tensorptr operator-(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2){
        if(t1->getShape() != t2->getShape()){
            throw std::invalid_argument("size of tensors don't match for sub operator");
        }
        auto result = Tensor::createZeros(t1->getShape());
        if(t1->getShape().size() == 0){
            result->setDataElem(0, t1->getDataElem(0) - t2->getDataElem(0));
            return result;
        }
        
        for(size_t i = 0; i < t1->getTotalSize();i++){
            result->setDataElem(i,t1->getDataElem(i) - t2->getDataElem(i));
        }
        return result;
    }
    Tensor::Tensorptr operator*(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2){
        if(t1->getShape() != t2->getShape()){
            throw std::invalid_argument("size of tensors don't match for mul operator");
        }
        auto result = Tensor::createZeros(t1->getShape());
        if(t1->getShape().size() == 0){
            result->setDataElem(0, t1->getDataElem(0) * t2->getDataElem(0));
            return result;
        }
        
        for(size_t i = 0; i < t1->getTotalSize();i++){
            result->setDataElem(i,t1->getDataElem(i) * t2->getDataElem(i));
        }
        return result;
    }
}
