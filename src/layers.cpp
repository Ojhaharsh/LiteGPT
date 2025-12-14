#include "layers.h"
#include "matmul.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

// Define M_PI for MSVC
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============= Activation Functions =============

void gelu(Tensor& tensor) {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    float* data = tensor.data();
    size_t size = tensor.total_elements();
    
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    const float c = 0.044715f;
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        float x = data[i];
        float x3 = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + c * x3);
        float tanh_val = std::tanh(tanh_arg);
        data[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

void softmax(Tensor& tensor) {
    // Softmax along last dimension with numerical stability
    
    if (tensor.ndim() < 1) return;
    
    auto shape = tensor.shape();
    int last_dim = shape.back();
    int num_rows = 1;
    for (int i = 0; i < (int)shape.size() - 1; ++i) {
        num_rows *= shape[i];
    }
    
    float* data = tensor.data();
    int row_size = tensor.total_elements() / num_rows;
    
    #pragma omp parallel for
    for (int row = 0; row < num_rows; ++row) {
        float* row_ptr = data + row * row_size;
        
        // Find max for numerical stability
        float max_val = row_ptr[0];
        for (int j = 0; j < last_dim; ++j) {
            max_val = std::max(max_val, row_ptr[j]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < last_dim; ++j) {
            row_ptr[j] = std::exp(row_ptr[j] - max_val);
            sum += row_ptr[j];
        }
        
        // Normalize
        for (int j = 0; j < last_dim; ++j) {
            row_ptr[j] /= (sum + 1e-9f);
        }
    }
}

// ============= Layer Implementations =============

LayerNorm::LayerNorm(int normalized_shape, float eps)
    : normalized_shape_(normalized_shape), eps_(eps) {}

Tensor LayerNorm::forward(const Tensor& input) {
    return forward(input, const_cast<Tensor&>(input), const_cast<Tensor&>(input));
}

Tensor LayerNorm::forward(const Tensor& input, Tensor& weight, Tensor& bias) {
    // LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
    
    Tensor output = input;
    auto shape = output.shape();
    
    if (shape.empty()) return output;
    
    int feature_dim = shape.back();
    int num_rows = 1;
    for (int i = 0; i < (int)shape.size() - 1; ++i) {
        num_rows *= shape[i];
    }
    
    float* data = output.data();
    float* w_data = weight.data();
    float* b_data = bias.data();
    
    int row_size = output.total_elements() / num_rows;
    
    #pragma omp parallel for
    for (int row = 0; row < num_rows; ++row) {
        float* row_ptr = data + row * row_size;
        
        // Compute mean
        float mean = 0.0f;
        for (int j = 0; j < feature_dim; ++j) {
            mean += row_ptr[j];
        }
        mean /= feature_dim;
        
        // Compute variance
        float var = 0.0f;
        for (int j = 0; j < feature_dim; ++j) {
            float diff = row_ptr[j] - mean;
            var += diff * diff;
        }
        var /= feature_dim;
        
        // Normalize and scale
        float std_dev = std::sqrt(var + eps_);
        for (int j = 0; j < feature_dim; ++j) {
            float normalized = (row_ptr[j] - mean) / std_dev;
            row_ptr[j] = normalized * w_data[j] + b_data[j];
        }
    }
    
    return output;
}

Linear::Linear(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features) {}

Tensor Linear::forward(const Tensor& input, Tensor& weight, Tensor& bias) {
    // output = input @ weight.T + bias
    
    auto in_shape = input.shape();
    std::vector<int> out_shape = in_shape;
    out_shape.back() = out_features_;
    
    Tensor output(out_shape);
    
    int batch_size = 1;
    for (int i = 0; i < (int)in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }
    
    Tensor input_2d = input;
    input_2d.reshape({batch_size, in_features_});
    
    Tensor output_2d({batch_size, out_features_});
    matmul_bias(input_2d, weight, bias, output_2d);
    
    return output;
}

MultiHeadAttention::MultiHeadAttention(int hidden_size, int num_heads)
    : hidden_size_(hidden_size), num_heads_(num_heads) {
    
    if (hidden_size % num_heads != 0) {
        throw std::invalid_argument("hidden_size must be divisible by num_heads");
    }
    
    head_dim_ = hidden_size / num_heads;
    scale_ = 1.0f / std::sqrt((float)head_dim_);
}

Tensor MultiHeadAttention::forward(
    const Tensor& input,
    Tensor& q_weight, Tensor& q_bias,
    Tensor& k_weight, Tensor& k_bias,
    Tensor& v_weight, Tensor& v_bias,
    Tensor& out_weight, Tensor& out_bias,
    bool use_cache) {
    
    // Placeholder: returns input unchanged for now
    // Full implementation in Phase 3
    return input;
}

Tensor MultiHeadAttention::forward_with_cache(
    const Tensor& input,
    Tensor& q_weight, Tensor& q_bias,
    Tensor& k_weight, Tensor& k_bias,
    Tensor& v_weight, Tensor& v_bias,
    Tensor& out_weight, Tensor& out_bias,
    Tensor& cached_k, Tensor& cached_v) {
    
    // Placeholder: returns input unchanged for now
    return input;
}

