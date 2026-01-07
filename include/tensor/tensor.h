#ifndef TENSOR_H
#define TENSOR_H
#include <array>
#include <initializer_list>
#include <memory>
#include <vector>
constexpr int MAX_RANK = 8;
enum InitType { Normal, He, Xavier, XavierUniform, HeUniform };
class Tensor : public std::enable_shared_from_this<Tensor> {
  private:
    std::unique_ptr<float[]> data;
    std::array<size_t, MAX_RANK> shape;
    std::array<size_t, MAX_RANK> stride;
    size_t total_size = 0;
    // bool require_grad = false;
    std::unique_ptr<float[]> grad;

  public:
    Tensor(std::unique_ptr<float[]> input_data, size_t size,
           const std::array<size_t, MAX_RANK> &shape,
           const std::array<size_t, MAX_RANK> &stride);

    // allows move only tensor
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    static Tensor createTensor(std::unique_ptr<float[]> input_data, size_t size,
                               const std::array<size_t, MAX_RANK> &shape,
                               bool require_grad = false);

    static Tensor CreateTensor(std::unique_ptr<float[]> input_data, size_t size,
                               const std::vector<size_t> &shape_vec,
                               bool require_grad = false);

    static Tensor CreateTensor(std::unique_ptr<float[]> input_data, size_t size,
                               const std::initializer_list<size_t> &shape_list,
                               bool require_grad = false);

    static std::array<size_t, MAX_RANK>
    calculateStrides(const std::array<size_t, MAX_RANK> &shape);
    static Tensor createScalar(float data);
    static Tensor createZeros(const std::initializer_list<size_t> &shape_list);
    static Tensor createZeros(const std::array<size_t, MAX_RANK> &shape);
    static Tensor createOnes(const std::initializer_list<size_t> &shape_list);
    static Tensor createOnes(const std::array<size_t, MAX_RANK> &shape);
    static Tensor
    createRandTensor(const std::initializer_list<size_t> &shape_list,
                     InitType mode = Normal);
    static Tensor createRandTensor(const std::array<size_t, MAX_RANK> &shape,
                                   InitType mode = Normal);
    const std::array<size_t, MAX_RANK> &getShape() const;
    size_t getTotalSize() const;
    void setDataElem(size_t i, float val);
    const float *getDataPtr() const;
    float *getMutableDataPtr() const;
    const float &operator()(size_t i) const;
    const float &operator()(size_t i, size_t j) const;
    const float &operator()(size_t i, size_t j, size_t k) const;
    size_t getRank() const;
    const std::array<size_t, MAX_RANK> getStrides() const;
    void zeroGrad() const;
};

#endif // TENSOR_H
