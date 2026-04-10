#include "tensorlib/tensor/tensor.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <tensorlib/tensor.h>
#include <tensorlib/tensor_RNG.h>
#include <utility>

Tensor::Tensor(std::unique_ptr<float[]> input_data, const size_t size,
               const std::array<size_t, MAX_RANK>& shape_in)
    : m_data(std::move(input_data)), m_shape(shape_in), m_total_size(size) {

    m_rank = calculateRank();
    m_stride = calculateStrides();
}

size_t Tensor::calculateRank() {
    size_t rank = 0;
    for (const auto dim : m_shape) {
        if (dim == 0)
            break;
        rank++;
    }
    return rank;
}
std::array<size_t, MAX_RANK> Tensor::calculateStrides() {
    std::array<size_t, MAX_RANK> strides{};
    if (m_rank == 0) {
        return strides;
    }
    strides[m_rank - 1] = 1;
    for (int i = static_cast<int>(m_rank) - 2; i >= 0; i--) {
        strides[static_cast<size_t>(i)] =
            strides[static_cast<size_t>(i) + 1] * m_shape[static_cast<size_t>(i) + 1];
    }
    return strides;
}

Tensor Tensor::createTensor(std::unique_ptr<float[]> input_data, std::span<const size_t> shape_data,
                            bool require_grad) {
    if (shape_data.size() > MAX_RANK) {
        throw std::invalid_argument("MAX_RANK allowed is 8");
    }
    std::array<size_t, MAX_RANK> shape_arr{};
    size_t total_size = 1;
    for (size_t i = 0; i < shape_data.size(); i++) {
        if (shape_data[i] == 0)
            throw std::invalid_argument("Shape data provided can't be zero");
        shape_arr[i] = shape_data[i];
        total_size *= shape_data[i];
    }
    if (require_grad) {
    }
    return Tensor(std::move(input_data), total_size, shape_arr);
}

Tensor Tensor::createTensor(std::span<const float> input_data, std::span<const size_t> shape_data,
                            bool require_grad) {
    if (shape_data.size() > MAX_RANK) {
        throw std::invalid_argument("MAX_RANK allowed is 8");
    }
    std::array<size_t, MAX_RANK> shape_arr{};
    size_t total_size = 1;
    for (size_t i = 0; i < shape_data.size(); i++) {
        if (shape_data[i] == 0)
            throw std::invalid_argument("Shape data provided can't be zero");
        shape_arr[i] = shape_data[i];
        total_size *= shape_data[i];
    }
    auto unique_arr = std::make_unique<float[]>(total_size);
    std::copy(input_data.begin(), input_data.end(), unique_arr.get());
    if (require_grad) {
    }
    return Tensor(std::move(unique_arr), total_size, shape_arr);
}

Tensor Tensor::createTensor(std::unique_ptr<float[]> input_data,
                            std::initializer_list<size_t> shape_data, bool require_grad) {
    return createTensor(std::move(input_data),
                        std::span<const size_t>(shape_data.begin(), shape_data.size()),
                        require_grad);
}

Tensor Tensor::createTensor(std::span<const float> input_data,
                            std::initializer_list<size_t> shape_data, bool require_grad) {
    return createTensor(input_data, std::span<const size_t>(shape_data.begin(), shape_data.size()),
                        require_grad);
}

Tensor Tensor::createTensor(std::initializer_list<float> input_data,
                            std::initializer_list<size_t> shape_data, bool require_grad) {
    return createTensor(std::span<const float>(input_data.begin(), input_data.size()),
                        std::span<const size_t>(shape_data.begin(), shape_data.size()),
                        require_grad);
}

Tensor Tensor::createScalar(const float data) {
    auto arr = std::make_unique<float[]>(1);
    arr[0] = data;
    return createTensor(std::move(arr), {1}, false);
}

Tensor Tensor::createOnes(std::initializer_list<size_t> shape_data) {
    std::array<size_t, MAX_RANK> shape{};
    size_t i = 0;
    if (shape_data.size() > MAX_RANK)
        throw std::invalid_argument("shape list provided is greater than 8");
    for (const auto dim : shape_data) {
        shape[i++] = dim;
    }
    return createOnes(shape);
}

Tensor Tensor::createOnes(const std::span<const size_t> shape_data) {
    bool is_scalar = true;
    for (const auto dim : shape_data) {
        if (dim != 0) {
            is_scalar = false;
            break;
        }
    }
    if (is_scalar)
        return createScalar(1.0f);
    size_t total_size = 1;
    for (const auto dim : shape_data) {
        if (dim == 0)
            break;
        total_size *= dim;
    }
    auto arr = std::make_unique<float[]>(total_size);
    std::fill_n(arr.get(), total_size, 1.0f);
    return createTensor(std::move(arr), shape_data, false);
}

Tensor Tensor::createZeros(const std::span<const size_t> shape_data) {
    bool is_scalar = true;
    for (const auto dim : shape_data) {
        if (dim != 0) {
            is_scalar = false;
            break;
        }
    }
    if (is_scalar)
        return createScalar(0.0f);
    size_t total_size = 1;
    for (const auto dim : shape_data) {
        if (dim == 0)
            break;
        total_size *= dim;
    }
    auto arr = std::make_unique<float[]>(total_size);
    std::fill_n(arr.get(), total_size, 0.0f);
    return createTensor(std::move(arr), shape_data, false);
}

Tensor Tensor::createZeros(const std::initializer_list<size_t> shape_data) {
    return createZeros(std::span<const size_t>(shape_data.begin(), shape_data.size()));
}

Tensor Tensor::createRandTensor(const std::initializer_list<size_t> shape_data,
                                const InitType mode) {
    return createRandTensor(std::span<const size_t>(shape_data.begin(), shape_data.size()), mode);
}

Tensor Tensor::createRandTensor(const std::span<const size_t> shape, const InitType mode) {
    size_t rank = 0;
    for (const auto dim : shape) {
        if (dim == 0)
            break;
        rank++;
    }
    size_t total_size = 1;
    for (const auto dim : shape) {
        if (dim == 0)
            break;
        total_size *= dim;
    }
    const size_t feature_out = total_size / shape[rank - 1];
    const size_t feature_in = total_size / shape[0];
    auto arr = std::make_unique<float[]>(total_size);

    std::mt19937& gen = TensorRNG::engine();

    switch (mode) {
    case InitType::He: {
        if (rank != 2)
            throw std::invalid_argument("He/Xavier initialization only supported for 2D tensors");
        const float limit = std::sqrt(2.0f / static_cast<float>(feature_in));
        std::normal_distribution<float> dist(0.0f, limit);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    } break;

    case InitType::Xavier: {
        if (rank != 2)
            throw std::invalid_argument("He/Xavier initialization only supported for 2D tensors");
        const float limit = std::sqrt(2.0f / static_cast<float>(feature_in + feature_out));
        std::normal_distribution<float> dist(0.0f, limit);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    } break;

    case InitType::HeUniform: {
        if (rank != 2)
            throw std::invalid_argument("He/Xavier initialization only supported for 2D tensors");
        const float limit = std::sqrt(6.0f / static_cast<float>(feature_in));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    } break;

    case InitType::XavierUniform: {
        if (rank != 2)
            throw std::invalid_argument("He/Xavier initialization only supported for 2D tensors");
        const float limit = std::sqrt(6.0f / static_cast<float>(feature_in + feature_out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    } break;

    default:
        std::normal_distribution<float> dist(0.00f, 0.01f);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    }
    return createTensor(std::move(arr), shape);
}

const std::span<const size_t> Tensor::getShape() const {
    return std::span<const size_t>(m_shape.data(), m_rank);
}

size_t Tensor::getRank() const {
    return m_rank;
}
size_t Tensor::getTotalSize() const {
    return m_total_size;
}

void Tensor::setDataElem(const size_t i, const float val) {
    m_data[i] = val;
}

float* Tensor::getMutableDataPtr() const {
    return m_data.get();
}

const float* Tensor::getDataPtr() const {
    return m_data.get();
}

const std::span<const float> Tensor::view() const {
    return std::span<const float>(m_data.get(), m_total_size);
}

const float& Tensor::operator()(const size_t i) const {
    if (i >= m_total_size)
        throw std::out_of_range("index 0-D out of range");
    return m_data[i];
}

std::ranges::minmax_result<float> Tensor::getMinMax() {
    auto t_view = std::span<const float>(m_data.get(), m_total_size);
    auto minMax = std::ranges::minmax(t_view);
    return minMax;
}

void Tensor::reshape(const std::array<size_t, MAX_RANK>& new_shape) {
    size_t total_elem = 1;
    for (const auto dim : new_shape) {
        if (dim == 0)
            break;
        total_elem *= dim;
    }
    if (total_elem == m_total_size) {
        m_shape = new_shape;
        m_rank = calculateRank();
        m_stride = calculateStrides();
    } else {
        throw std::invalid_argument("reshape doesn't contain all the elements");
    }
}

void Tensor::reshape(const std::initializer_list<size_t> new_shape_data) {
    std::array<size_t, MAX_RANK> new_shape{};
    size_t i = 0;
    for (const auto dim : new_shape_data) {
        new_shape[i++] = dim;
    }
    reshape(new_shape);
}

const float& Tensor::operator()(const size_t i, const size_t j) const {
    if (i >= m_shape[0] || j >= m_shape[1])
        throw std::out_of_range("index 2-D out of range");
    return m_data[i * m_stride[0] + j * m_stride[1]];
}

const float& Tensor::operator()(const size_t i, const size_t j, const size_t k) const {
    if (i >= m_shape[0] || j >= m_shape[1] || k >= m_shape[2]) {
        throw std::out_of_range("index 3-D out of range");
    }
    return m_data[i * m_stride[0] + j * m_stride[1] + k * m_stride[2]];
}

const std::span<const size_t> Tensor::getStrides() const {
    return std::span<const size_t>(m_stride.data(), m_rank);
}

void Tensor::zeroGrad() const {
    std::fill_n(m_grad.get(), m_total_size, 0.0f);
}
