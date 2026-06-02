#include <tensorlib/tensor_impl.h>

TensorImpl::TensorImpl(std::unique_ptr<float[]> input_data, size_t size,
                       const std::array<size_t, MAX_RANK>& shape)
    : m_data(std::move(input_data)), m_shape(shape), m_total_size(size) {

    m_rank = calculateRank();
    m_strides = calculateStrides();
}
