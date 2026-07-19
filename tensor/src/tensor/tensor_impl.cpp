#include <tensorlib/tensor_impl.h>

TensorImpl::TensorImpl(std::unique_ptr<float[]> input_data, size_t size,
                       const std::array<size_t, MAX_RANK>& shape)
    : m_data(std::move(input_data)), m_shape(shape), m_total_size(size) {

    m_rank = calculateRank();
    m_strides = calculateStrides();
}

void TensorImpl::ensureGrad() {
    if (!m_grad) {
        m_grad = std::make_unique<float[]>(m_total_size);
        std::fill_n(m_grad.get(), m_total_size, 0.0f);
    }
}

void TensorImpl::accumulatedGrad(const float* incoming, size_t size) {
    ensureGrad();
    for (size_t i = 0; i < size; i++)
        m_grad[i] += incoming[i];
}
