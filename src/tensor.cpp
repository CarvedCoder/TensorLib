#include "../include/tensor.h"
#include <iostream>
#include <random>
#include <stdexcept>

Tensor::Tensor(std::unique_ptr<float[]> input_data, const size_t size,
               const std::array<size_t, MAX_RANK> &shape_in,
               const std::array<size_t, MAX_RANK> &stride_in)
    : data(std::move(input_data)), shape(shape_in), stride(stride_in),
      total_size(size) {}

Tensor Tensor::createTensor(std::unique_ptr<float[]> input_data,
                            const size_t size,
                            const std::array<size_t, MAX_RANK> &shape,
                            bool require_grad) {
    size_t expected_size = 1;
    for (const auto dim : shape) {
        if (dim == 0)
            break;
        expected_size *= dim;
    }
    if (expected_size != size) {
        throw std::invalid_argument(
            "shape and data size don't match or -ve size_t "
            "passed in shape while creating a new tensor");
    }
    const size_t f_size = expected_size;
    const auto stride = calculateStrides(shape);
    if (require_grad) {
    }
    return Tensor(std::move(input_data), f_size, shape, stride);
}

Tensor Tensor::CreateTensor(std::unique_ptr<float[]> input_data,
                            const size_t size,
                            const std::vector<size_t> &shape_vec,
                            bool require_grad) {
    std::array<size_t, MAX_RANK> shape{};
    if (shape_vec.size() > MAX_RANK)
        std::cerr << "shape vec provided is greater than 8";
    size_t i = 0;
    for (const auto dim : shape_vec) {
        shape[i++] = dim;
    }
    if (require_grad) {
    }
    return createTensor(std::move(input_data), size, shape);
}

Tensor Tensor::CreateTensor(std::unique_ptr<float[]> input_data,
                            const size_t size,
                            const std::initializer_list<size_t> &shape_list,
                            bool require_grad) {
    std::array<size_t, MAX_RANK> shape{};
    size_t i = 0;
    if (shape_list.size() > MAX_RANK)
        std::cerr << "shape list provided is greater than 8";
    for (const auto dim : shape_list) {
        shape[i++] = dim;
    }
    if (require_grad) {
    }
    return createTensor(std::move(input_data), size, shape);
}

std::array<size_t, MAX_RANK>
Tensor::calculateStrides(const std::array<size_t, MAX_RANK> &shape) {
    std::array<size_t, MAX_RANK> strides{};
    size_t rank = 0;
    for (size_t i = 0; i < MAX_RANK; i++) {
        if (shape[i] == 0)
            break;
        rank++;
    }
    if (rank == 0) {
        strides[0] = 1;
        return strides;
    }
    size_t stride_val = 1;
    for (auto i = static_cast<size_t>(rank); i-- > 0;) {
        strides[static_cast<size_t>(i)] = stride_val;
        stride_val *= shape[static_cast<size_t>(i)];
    }
    return strides;
}

Tensor Tensor::createScalar(const float data) {
    auto arr = std::make_unique<float[]>(1);
    arr[0] = data;
    return CreateTensor(std::move(arr), 1, {});
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
    return createTensor(std::move(arr), total_size, shape);
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
    size_t total_size = 1, rank = 0;
    for (const auto dim : shape) {
        if (dim == 0)
            break;
        total_size *= dim;
        rank++;
    }
    const size_t feature_out = shape[0];
    const size_t feature_in = shape[rank - 1];
    auto arr = std::make_unique<float[]>(total_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    switch (mode) {
    case He: {
        const float limit = std::sqrt(2.0f / static_cast<float>(feature_in));
        std::normal_distribution<float> dist(0.0f, limit);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    } break;

    case Xavier: {
        const float limit =
            std::sqrt(2.0f / static_cast<float>(feature_in + feature_out));
        std::normal_distribution<float> dist(0.0f, limit);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    } break;

    case HeUniform: {
        const float limit = std::sqrt(6.0f / static_cast<float>(feature_in));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (size_t i = 0; i < total_size; i++) {
            arr[i] = dist(gen);
        }
    } break;

    case XavierUniform: {
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
    return createTensor(std::move(arr), total_size, shape);
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
        return createScalar(1.0f);
    size_t total_size = 1;
    for (const auto dim : shape) {
        if (dim == 0)
            break;
        total_size *= dim;
    }
    auto arr = std::make_unique<float[]>(total_size);
    std::fill_n(arr.get(), total_size, 0.0f);
    return createTensor(std::move(arr), total_size, shape);
}

const std::array<size_t, MAX_RANK> &Tensor::getShape() const { return shape; }

size_t Tensor::getTotalSize() const { return total_size; }

void Tensor::setDataElem(const size_t i, const float val) { data[i] = val; }

float *Tensor::getMutableDataPtr() const { return data.get(); }

const float *Tensor::getDataPtr() const { return data.get(); }

const float &Tensor::operator()(const size_t i) const {
    if (i >= total_size)
        throw std::out_of_range("index 0-D out of range");
    return data[i];
}

const float &Tensor::operator()(const size_t i, const size_t j) const {
    if (i >= shape[0] || j >= shape[1])
        throw std::out_of_range("index 2-D out of range");
    return data[i * stride[0] + j * stride[1]];
}

const float &Tensor::operator()(const size_t i, const size_t j,
                                const size_t k) const {
    if (i >= shape[0] || j >= shape[1] || k >= shape[2]) {
        throw std::out_of_range("index 3-D out of range");
    }
    return data[i * stride[0] + j * stride[1] + k * stride[2]];
}

size_t Tensor::getRank() const {
    size_t rank = 0;
    for (size_t i = 0; i < MAX_RANK; i++) {
        if (shape[i] == 0)
            break;
        rank++;
    }
    return rank;
}

const std::array<size_t, MAX_RANK> Tensor::getStrides() const { return stride; }

void Tensor::zeroGrad() const { std::fill_n(grad.get(), total_size, 0.0f); }
