#ifndef TENSOR_H
#define TENSOR_H
#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <random>
#include <span>
#include <vector>

constexpr int MAX_RANK = 8;
enum class InitType { Normal, He, Xavier, XavierUniform, HeUniform };
class Tensor {
  private:
    std::unique_ptr<float[]> m_data;
    std::array<size_t, MAX_RANK> m_shape;
    std::array<size_t, MAX_RANK> m_stride;
    size_t m_total_size = 0;
    // bool require_grad = false;
    std::unique_ptr<float[]> m_grad;
    size_t m_rank;
    std::array<size_t, MAX_RANK> calculateStrides();
    size_t calculateRank();

  public:
    Tensor(std::unique_ptr<float[]> input_data, size_t size,
           const std::array<size_t, MAX_RANK> &shape);

    // allows move only tensor
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    static Tensor createTensor(std::unique_ptr<float[]> input_data, size_t size,
                               const std::array<size_t, MAX_RANK> &shape,
                               bool require_grad = false);

    static Tensor createTensor(std::unique_ptr<float[]> input_data, size_t size,
                               const std::vector<size_t> &shape_vec,
                               bool require_grad = false);

    static Tensor createTensor(std::unique_ptr<float[]> input_data, size_t size,
                               const std::initializer_list<size_t> &shape_list,
                               bool require_grad = false);

    static Tensor createScalar(float data);
    static Tensor createZeros(const std::initializer_list<size_t> &shape_list);
    static Tensor createZeros(const std::array<size_t, MAX_RANK> &shape);
    static Tensor createZeros(const std::span<const size_t> &span_shape);
    static Tensor createOnes(const std::initializer_list<size_t> &shape_list);
    static Tensor createOnes(const std::array<size_t, MAX_RANK> &shape);
    static Tensor
    createRandTensor(const std::initializer_list<size_t> &shape_list,
                     InitType mode = InitType::Normal);
    static Tensor createRandTensor(const std::array<size_t, MAX_RANK> &shape,
                                   InitType mode = InitType::Normal);
    const std::span<const size_t> getShape() const;
    const std::span<const float> view() const;
    size_t getTotalSize() const;
    size_t getRank() const;
    void setDataElem(size_t i, float val);
    const float *getDataPtr() const;
    float *getMutableDataPtr() const;
    const float &operator()(size_t i) const;
    const float &operator()(size_t i, size_t j) const;
    const float &operator()(size_t i, size_t j, size_t k) const;
    const std::span<const size_t> getStrides() const;
    void zeroGrad() const;
};

#endif // TENSOR_H
