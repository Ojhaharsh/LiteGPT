#include "tensor.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>
#include <random>

Tensor::Tensor() 
    : data_(nullptr), total_elements_(0) {}

Tensor::Tensor(const std::vector<int>& shape)
    : shape_(shape) {
    total_elements_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    if (total_elements_ == 0) {
        throw std::invalid_argument("Tensor shape cannot be empty");
    }
    data_ = std::shared_ptr<float>(new float[total_elements_]);
    compute_strides();
    zero();
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<int>& strides)
    : shape_(shape), strides_(strides) {
    total_elements_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    if (total_elements_ == 0) {
        throw std::invalid_argument("Tensor shape cannot be empty");
    }
    data_ = std::shared_ptr<float>(new float[total_elements_]);
    zero();
}

Tensor::~Tensor() {}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    
    strides_[shape_.size() - 1] = 1;
    for (int i = shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::compute_index(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index dimensions do not match tensor dimensions");
    }
    
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides_[i];
    }
    return index;
}

void Tensor::validate_indices(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::out_of_range("Number of indices does not match tensor dimensions");
    }
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
    }
}

float& Tensor::operator[](const std::vector<int>& indices) {
    return data_.get()[compute_index(indices)];
}

const float& Tensor::operator[](const std::vector<int>& indices) const {
    return data_.get()[compute_index(indices)];
}

float& Tensor::at(const std::vector<int>& indices) {
    validate_indices(indices);
    return operator[](indices);
}

const float& Tensor::at(const std::vector<int>& indices) const {
    validate_indices(indices);
    return operator[](indices);
}

float Tensor::get(const std::vector<int>& indices) const {
    return at(indices);
}

void Tensor::set(const std::vector<int>& indices, float value) {
    at(indices) = value;
}

void Tensor::fill(float value) {
    std::fill(data_.get(), data_.get() + total_elements_, value);
}

void Tensor::random() {
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dis(0.0f, 0.1f);
    
    for (size_t i = 0; i < total_elements_; ++i) {
        data_.get()[i] = dis(gen);
    }
}

void Tensor::mul(float scalar) {
    for (size_t i = 0; i < total_elements_; ++i) {
        data_.get()[i] *= scalar;
    }
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (new_total != total_elements_) {
        throw std::invalid_argument("Cannot reshape tensor: total elements mismatch");
    }
    
    shape_ = new_shape;
    compute_strides();
}

void Tensor::view(const std::vector<int>& new_shape) {
    // Same as reshape for now (true view with strides would be more complex)
    reshape(new_shape);
}

void Tensor::print() const {
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    if (total_elements_ == 0) return;
    
    // Print first 10 elements
    std::cout << "Data (first 10 elements): [";
    for (size_t i = 0; i < std::min(size_t(10), total_elements_); ++i) {
        std::cout << data_.get()[i];
        if (i < std::min(size_t(10), total_elements_) - 1) std::cout << ", ";
    }
    if (total_elements_ > 10) std::cout << " ...";
    std::cout << "]" << std::endl;
}
