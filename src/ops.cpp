#include "../include/ops.h"
#include <complex>
#include "../include/tensor.h"

namespace TensorOps{
    Tensor::Tensorptr operator+(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2){
        if(t1->getShape() != t2->getShape()){
            throw std::invalid_argument("shape of tensors don't match for add operator");
        }
        auto result = Tensor::createZeros(t1->getShape());
        if(t1->getShape().empty()){
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
            throw std::invalid_argument("shape of tensors don't match for sub operator");
        }
        auto result = Tensor::createZeros(t1->getShape());
        if(t1->getShape().empty()){
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
            throw std::invalid_argument("shape of tensors don't match for mul operator");
        }
        auto result = Tensor::createZeros(t1->getShape());
        if(t1->getShape().empty()){
            result->setDataElem(0, t1->getDataElem(0) * t2->getDataElem(0));
            return result;
        }
        
        for(size_t i = 0; i < t1->getTotalSize();i++){
            result->setDataElem(i,t1->getDataElem(i) * t2->getDataElem(i));
        }
        return result;
    }

    Tensor::Tensorptr operator*(const Tensor::Tensorptr & lhs, size_t rhs);

    Tensor::Tensorptr transpose2D(const std::shared_ptr<Tensor>&t) {
        const size_t rank = t->getRank();
        const auto shape = t->getShape();
        if (rank >= 3 || rank == 1) throw std::invalid_argument("Expected 2D rank tensor for 'transpose2D' but got >2D rank tensor");
        size_t rows =  shape[0];
        size_t cols = shape[1];
        auto T = Tensor::createZeros({cols,rows});
        for (size_t i = 0; i < rows;i++) {
            for (size_t j = 0;j < cols;j++) {
                (*T)(j,i) = (*t)(i,j);
            }
        }
        return T;
    }

    float sigmoid(const float input_data) {
        return 1/(1+std::exp(-input_data));
    }

    float relu(const float input_data) {
        return input_data > 0 ? input_data : 0.0f;
    }

    float leakyRelu(const float input_data) {
        return input_data > 0? input_data : 0.01f;
    }

    float m_tanh(const float input_data) {
        return std::tanh(input_data);
    }
    float calcCost(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2,const LossType mode) {
        if (t1->getShape()!= t2->getShape()) throw std::invalid_argument ("Tensor shape not same for calcCost");
        float result = 0;
        const size_t size = t1->getTotalSize();
        for (size_t i = 0; i < size;i++) {
            const float diff =(*t1)(i)-(*t2)(i);
            result += diff * diff;
        }
        return mode == SSE ? result : result/static_cast<float>(size);
    }

    Tensor::Tensorptr matmul(const std::shared_ptr<Tensor>&t1,const std::shared_ptr<Tensor>&t2) {
        const auto r1 = t1->getRank();
        const auto r2 = t2->getRank();
        if ( r1 != r2) throw std::invalid_argument("Rank is not the same for matmul");
        if (r1 > 2) throw std::invalid_argument("Rank input is greater than 2 in matmul");
        const auto s1 = t1->getShape();
        const auto s2 = t2->getShape();
        const size_t m = s1[0];
        const size_t K = s1[1];
        const size_t n = s2[1];
        if (K != s2[0]) throw std::invalid_argument("Inner ranks in matmul aren't the same");
        auto result = Tensor::createZeros({m,n});
        for (size_t i = 0; i < m;i++) {
            for (size_t k =0; k < K;k++) {
                const float a_ik = (*t1)(i,k);
                for (size_t j = 0; j < n;j++) {
                    (*result)(i,j) += a_ik * (*t2)(k,j);
                }
            }
        }
        return result;
    }
}
