#ifndef TENSOR_H
#define TENSOR_H
#include <array>
#include <initializer_list>
#include <memory>
#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>
constexpr int MAX_RANK =8;
enum InitType {Normal,He,Xavier,XavierUniform,HeUniform};
class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    std::unique_ptr<float[]> data;
    std::array<size_t, MAX_RANK> shape;
    std::array<size_t, MAX_RANK> stride;
    size_t total_size = 0;
    bool require_grad = false;
    std::unique_ptr<float[]>grad;
private:
    Tensor(std::unique_ptr<float[]> input_data,
           size_t size,
           const std::array<size_t, MAX_RANK> &shape,
           const std::array<size_t, MAX_RANK> &stride
    );

public:
    using Tensorptr = std::shared_ptr<Tensor>;

    static Tensorptr createTensor(std::unique_ptr<float[]> input_data,
                                  size_t size,
                                  const std::array<size_t, MAX_RANK> &shape,
                                  bool require_grad = false);

    static Tensorptr CreateTensor(std::unique_ptr<float[]> input_data,
                                  size_t size,
                                  const std::vector<size_t>& shape_vec,
                                  bool require_grad = false);

    static Tensorptr CreateTensor(std::unique_ptr<float[]> input_data,
                                  size_t size,
                                  const std::initializer_list<size_t>& shape_list,
                                  bool require_grad = false);


    static std::array<size_t, MAX_RANK> calculateStrides(const std::array<size_t, MAX_RANK> &shape);
    static Tensorptr createScalar(float data);
    static Tensorptr createZeros(const std::initializer_list<size_t>&shape_list);
    static Tensorptr createZeros(const std::array<size_t,MAX_RANK>& shape);
    static Tensorptr createOnes(const std::initializer_list<size_t> &shape_list);
    static Tensorptr createOnes(const std::array<size_t,MAX_RANK>& shape);
    static Tensorptr createRandTensor(const std::initializer_list<size_t> &shape_list,InitType mode = Normal);
    static Tensorptr createRandTensor(const std::array<size_t,MAX_RANK>&shape,InitType mode = Normal);
    const std::array<size_t, MAX_RANK> &getShape() const;
    size_t getTotalSize() const;
    float getDataElem(size_t i) const;
    void setDataElem(size_t i, float val);
    const float *getData() const;
    float &operator()(size_t i);
    float &operator()(size_t i, size_t j);
    float &operator()(size_t i, size_t j, size_t k);
    size_t getRank() const;
    std::array<size_t,MAX_RANK> getStrides() const;
    void zeroGrad() const;
};

#endif // TENSOR_H
