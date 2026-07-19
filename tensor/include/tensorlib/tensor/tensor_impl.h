#ifndef TENSOR_IMPL_H

#include <array>
#include <cstddef>
#include <memory>
constexpr int MAX_RANK = 8;

struct Node;

struct TensorImpl {
    std::unique_ptr<float[]> m_data;
    std::array<size_t, MAX_RANK> m_shape{};
    std::array<size_t, MAX_RANK> m_strides{};
    size_t m_rank{};
    size_t m_total_size{};
    bool m_require_grad = false;
    std::unique_ptr<float[]> m_grad;
    bool m_is_leaf = true;
    std::shared_ptr<Node> m_grad_fn;
    std::array<size_t, MAX_RANK> calculateStrides();
    size_t calculateRank();
    explicit TensorImpl(std::unique_ptr<float[]> input_data, size_t size,
                        const std::array<size_t, MAX_RANK>& shape);
    void ensureGrad();
    void accumulatedGrad(const float* incoming, size_t size);
};

#endif // !TENSOR_IMPL_H
