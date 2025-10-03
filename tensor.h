#ifndef TENSOR_H
#define TENSOR_H
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

constexpr int MAX_RANK = 8;

class Tensor:public std::enable_shared_from_this<Tensor>{
private:
    std::unique_ptr<float[]> data; 
    std::array<size_t,MAX_RANK> shape;
    std::array<size_t,MAX_RANK> stride;
    size_t total_size = 0;
    // bool require_grad = false; 
    // std::unique_ptr<float[]>grad;
private:
    Tensor(std::unique_ptr<float[]>input_data,
           size_t size,
           const std::array<size_t,MAX_RANK>&shape,
           const std::array<size_t,MAX_RANK>&stride
           ):
           data(std::move(input_data)),shape(shape),stride(stride),total_size(size){}
public:
    using Tensorptr = std::shared_ptr<Tensor>;
    static Tensorptr CreateTensor(std::unique_ptr<float[]>input_data,
                                  size_t size,
                                  const std::array<size_t,MAX_RANK>& shape){
        size_t expected_size = 1;
        for (auto dim:shape) {
            if (dim == 0) break;
            expected_size *= dim;
        }
        if (expected_size != size || expected_size < 0){
            throw std::invalid_argument("shape and data size don't match or -ve int passed in shape");
        }
        auto stride = calculate_strides(shape); 
        auto tensor = Tensorptr(new Tensor(std::move(input_data),size,shape,stride));
        return tensor;
    }

    static Tensorptr CreateTensor(std::unique_ptr<float[]>input_data,
                                  size_t size,
                                  std::initializer_list<size_t> shape_list){
        std::array<size_t,MAX_RANK>shape{};
        size_t i = 0;
        for(auto s : shape_list){
            if(i >= MAX_RANK) break;
            shape[i++] = s;
        }
        return CreateTensor(std::move(input_data),size,shape);
    }
    
    static std::array<size_t,8> calculate_strides(const std::array<size_t,MAX_RANK>&shape){ 
        std::array<size_t,MAX_RANK> strides{};
        if (shape[0] == 0) return strides;
        size_t rank=0;
        for(size_t i = 0; i < MAX_RANK;i++){
            if(shape[i] == 0)break;
            rank++;
        }
        size_t stride_val = 1;
        for(int i = static_cast<int>(rank)-1;i>= 0;i--){
            strides[static_cast<size_t>(i)] = stride_val;
            stride_val *= shape[static_cast<size_t>(i)];
        }
    return strides;
    } 
    
    static Tensorptr createScalar(float data){
        auto arr = std::make_unique<float[]>(1);
        arr[0]=data;
        return CreateTensor(std::move(arr),1,{});
    }

    static Tensorptr createZeros(const std::array<size_t,MAX_RANK>&shape){
        if (shape[0]==0) {return createScalar(0.0f);}
        size_t total_size = 1;
        for(auto dim:shape){
            if(dim == 0) break;
            total_size *= dim;
        }
        auto arr = std::make_unique<float[]>(total_size);
        std::fill_n(arr.get(), total_size, 0.0f);
        return CreateTensor(std::move(arr),total_size,shape);
    }

    static Tensorptr createOnes(const std::array<size_t,MAX_RANK>&shape){
        if (shape[0]==0) {return createScalar(1.0f);}
        size_t total_size = 1;
        for (auto dim : shape){
            if (dim == 0) break;
            total_size*=dim;
        }
        auto arr = std::make_unique<float[]>(total_size);
        std::fill_n(arr.get(), total_size, 1.0f);
        return CreateTensor(std::move(arr),total_size,shape);
    } 

    const std::array<size_t,MAX_RANK>&getShape() {return shape;}
    size_t getTotalSize(){return total_size;}
    float getDataElem(size_t i){return data[i];}
    void setDataElem(size_t i,float val){data[i]=val;}
    const float* getData()const{return data.get();}
    float &operator()(size_t i){
        if(i >= total_size) throw std::out_of_range("index 0-D out of range");
        return data[i];
    }
    float &operator()(size_t i,size_t j){
        if(i >= shape[0] || j >= shape[1]) throw std::out_of_range("index 2-D out of range");
        return data[i*stride[0] + j*stride[1]];
    }
    float &operator()(size_t i,size_t j,size_t k){
        if(i >= shape[0] || j >= shape[1] || k >= shape[2]){
            throw std::out_of_range("index 3-D out of range");
        }
        return data[i*stride[0]+j*stride[1]+k*stride[2]];
    }
    // void zeroGrad(){std::fill_n(grad.get(),total_size,0.0f);}
};

#endif // TENSOR_H
