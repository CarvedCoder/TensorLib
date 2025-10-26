#ifndef TENSOR_H
#define TENSOR_H
#include <algorithm>
#include <array>
#include <initializer_list>
#include <memory>
#include <vector>
constexpr int MAX_RANK = 8;

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    std::unique_ptr<float[]> data;
    std::array<size_t, MAX_RANK> shape;
    std::array<size_t, MAX_RANK> stride;
    size_t total_size = 0;
    // bool require_grad = false;
    // std::unique_ptr<float[]>grad;
private:
    Tensor(std::unique_ptr<float[]> input_data,
           size_t size,
           const std::array<size_t, MAX_RANK> &shape,
           const std::array<size_t, MAX_RANK> &stride
    );

public:
    using Tensorptr = std::shared_ptr<Tensor>;

    static Tensorptr CreateTensor(std::unique_ptr<float[]> input_data,
                                  size_t size,
                                  const std::array<size_t, MAX_RANK> &shape);

    static Tensorptr CreateTensor(std::unique_ptr<float[]> input_data,
                                  size_t size,
                                  const std::vector<size_t>& shape_vec);

    static Tensorptr CreateTensor(std::unique_ptr<float[]> input_data,
                                  size_t size,
                                  const std::initializer_list<size_t>& shape_list);


    static std::array<size_t, 8> calculate_strides(const std::array<size_t, MAX_RANK> &shape);

    static Tensorptr createScalar(float data);

    static Tensorptr createZeros(const std::initializer_list<size_t>&shape_list);
    static Tensorptr createZeros(const std::array<size_t,MAX_RANK>& shape);
    static Tensorptr createOnes(const std::initializer_list<size_t> &shape_list);
    static Tensorptr createOnes(const std::array<size_t,MAX_RANK>& shape);

    const std::array<size_t, MAX_RANK> &getShape() const;

    size_t getTotalSize() const;

    float getDataElem(size_t i) const;

    void setDataElem(size_t i, float val);

    const float *getData() const;

    float &operator()(size_t i);

    float &operator()(size_t i, size_t j);

    float &operator()(size_t i, size_t j, size_t k);

    size_t getRank() const;

    // void zeroGrad(){std::fill_n(grad.get(),total_size,0.0f);}
};

#endif // TENSOR_H
