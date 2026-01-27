#include <tensorlib/tensor.h>
#include <tensorlib/tensor_RNG.h>
Tensor::Tensor(std::unique_ptr<float[]> input_data, const size_t size,
               const std::array<size_t, MAX_RANK> &shape_in)
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
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i) + 1] *
                                          m_shape[static_cast<size_t>(i) + 1];
    }
    return strides;
}

Tensor Tensor::createTensor(std::unique_ptr<float[]> input_data,
                            const std::array<size_t, MAX_RANK> &shape,
                            bool require_grad) {
    size_t total_size = 1;
    for (const auto dim : shape) {
        if (dim == 0)
            break;
        total_size *= dim;
    }
    if (require_grad) {
    }
    return Tensor(std::move(input_data), total_size, shape);
}

Tensor Tensor::createTensor(std::unique_ptr<float[]> input_data,
                            const std::vector<size_t> &shape_vec,
                            bool require_grad) {
    std::array<size_t, MAX_RANK> shape{};
    if (shape_vec.size() > MAX_RANK)
        throw std::invalid_argument("Shape passed is greater than 8");
    size_t i = 0;
    for (const auto dim : shape_vec) {
        shape[i++] = dim;
    }
    if (require_grad) {
    }
    return createTensor(std::move(input_data), shape);
}

Tensor Tensor::createTensor(std::unique_ptr<float[]> input_data,
                            const std::initializer_list<size_t> &shape_list,
                            bool require_grad) {
    std::array<size_t, MAX_RANK> shape{};
    size_t i = 0;
    if (shape_list.size() > MAX_RANK)
        throw std::invalid_argument("Shape passed is greater than 8");
    for (const auto dim : shape_list) {
        shape[i++] = dim;
    }
    if (require_grad) {
    }
    return createTensor(std::move(input_data), shape);
}

Tensor Tensor::createScalar(const float data) {
    auto arr = std::make_unique<float[]>(1);
    arr[0] = data;
    return createTensor(std::move(arr), {});
}

Tensor Tensor::createOnes(const std::initializer_list<size_t> &shape_list) {
    std::array<size_t, MAX_RANK> shape{};
    size_t i = 0;
    if (shape_list.size() > MAX_RANK)
        throw std::invalid_argument("shape list provided is greater than 8");
    for (const auto dim : shape_list) {
        shape[i++] = dim;
    }
    return createOnes(shape);
}

Tensor Tensor::createOnes(const std::array<size_t, MAX_RANK> &shape) {
    bool is_scalar = true;
    for (const auto dim : shape) {
        if (dim != 0) {
            is_scalar = false;
            break;
        }
    }
    if (is_scalar)
        return createScalar(1.0f);
    size_t total_size = 1;
    for (const auto dim : shape) {
        if (dim == 0)
            break;
        total_size *= dim;
    }
    auto arr = std::make_unique<float[]>(total_size);
    std::fill_n(arr.get(), total_size, 1.0f);
    return createTensor(std::move(arr), shape);
}

Tensor Tensor::createZeros(const std::array<size_t, MAX_RANK> &shape) {
    bool is_scalar = true;
    for (const auto dim : shape) {
        if (dim != 0) {
            is_scalar = false;
            break;
        }
    }
    if (is_scalar)
        return createScalar(0.0f);
    size_t total_size = 1;
    for (const auto dim : shape) {
        if (dim == 0)
            break;
        total_size *= dim;
    }
    auto arr = std::make_unique<float[]>(total_size);
    std::fill_n(arr.get(), total_size, 0.0f);
    return createTensor(std::move(arr), shape);
}

Tensor Tensor::createZeros(const std::initializer_list<size_t> &shape_list) {
    std::array<size_t, MAX_RANK> shape{};
    size_t i = 0;
    if (shape_list.size() > MAX_RANK)
        throw std::invalid_argument("shape list provided is greater than 8");
    for (const auto dim : shape_list) {
        shape[i++] = dim;
    }
    return createZeros(shape);
}

Tensor Tensor::createZeros(const std::span<const size_t> &span_shape) {
    std::array<size_t, MAX_RANK> shape{};
    size_t i = 0;
    for (const auto dim : span_shape) {
        shape[i++] = dim;
    }
    return createZeros(shape);
}

Tensor Tensor::createRandTensor(const std::initializer_list<size_t> &shape_list,
                                const InitType mode) {
    std::array<size_t, MAX_RANK> shape{};
    size_t i = 0;
    for (const auto dim : shape_list) {
        shape[i++] = dim;
    }
    return createRandTensor(shape, mode);
}

Tensor Tensor::createRandTensor(const std::array<size_t, MAX_RANK> &shape,
                                const InitType mode) {
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

    std::mt19937 gen = TensorRNG::engine();

    switch (mode) {
    case InitType::He: {
        const float limit = std::sqrt(2.0f / static_cast<float>(feature_in));
        std::normal_distribution<float> dist(0.0f, limit);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    } break;

    case InitType::Xavier: {
        const float limit =
            std::sqrt(2.0f / static_cast<float>(feature_in + feature_out));
        std::normal_distribution<float> dist(0.0f, limit);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    } break;

    case InitType::HeUniform: {
        const float limit = std::sqrt(6.0f / static_cast<float>(feature_in));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    } break;

    case InitType::XavierUniform: {
        const float limit =
            std::sqrt(6.0f / static_cast<float>(feature_in + feature_out));
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

size_t Tensor::getRank() const { return m_rank; }
size_t Tensor::getTotalSize() const { return m_total_size; }

void Tensor::setDataElem(const size_t i, const float val) { m_data[i] = val; }

float *Tensor::getMutableDataPtr() const { return m_data.get(); }

const float *Tensor::getDataPtr() const { return m_data.get(); }

const std::span<const float> Tensor::view() const {
    return std::span<const float>(m_data.get(), m_total_size);
}

const float &Tensor::operator()(const size_t i) const {
    if (i >= m_total_size)
        throw std::out_of_range("index 0-D out of range");
    return m_data[i];
}

void Tensor::reshape(const std::array<size_t, MAX_RANK> &new_shape) {
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

void Tensor::reshape(const std::initializer_list<size_t> &new_shape_list) {
    std::array<size_t, MAX_RANK> new_shape{};
    size_t i = 0;
    for (const auto dim : new_shape_list) {
        new_shape[i++] = dim;
    }
    reshape(new_shape);
}

const float &Tensor::operator()(const size_t i, const size_t j) const {
    if (i >= m_shape[0] || j >= m_shape[1])
        throw std::out_of_range("index 2-D out of range");
    return m_data[i * m_stride[0] + j * m_stride[1]];
}

const float &Tensor::operator()(const size_t i, const size_t j,
                                const size_t k) const {
    if (i >= m_shape[0] || j >= m_shape[1] || k >= m_shape[2]) {
        throw std::out_of_range("index 3-D out of range");
    }
    return m_data[i * m_stride[0] + j * m_stride[1] + k * m_stride[2]];
}

const std::span<const size_t> Tensor::getStrides() const {
    return std::span<const size_t>(m_stride.data(), m_rank);
}

void Tensor::zeroGrad() const { std::fill_n(m_grad.get(), m_total_size, 0.0f); }
