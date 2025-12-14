#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"
#include <cmath>

// ============= Activation Functions =============

// GELU (Gaussian Error Linear Unit)
// Used in GPT-2 and modern transformers
void gelu(Tensor& tensor);
void gelu_backward(Tensor& grad);

// Softmax along last dimension
void softmax(Tensor& tensor);

// ============= Layer Implementations =============

// Layer Normalization
// Normalizes across the feature dimension
class LayerNorm {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5);
    
    // Forward pass: output = (input - mean) / sqrt(var + eps) * weight + bias
    Tensor forward(const Tensor& input);
    Tensor forward(const Tensor& input, Tensor& weight, Tensor& bias);
    
private:
    int normalized_shape_;
    float eps_;
};

// Linear Layer (Dense/Fully Connected)
// Implements: output = input @ weight.T + bias
class Linear {
public:
    Linear(int in_features, int out_features);
    
    // Forward pass with loaded weights
    Tensor forward(const Tensor& input, Tensor& weight, Tensor& bias);
    
private:
    int in_features_;
    int out_features_;
};

// Multi-Head Attention
class MultiHeadAttention {
public:
    MultiHeadAttention(int hidden_size, int num_heads);
    
    // Forward pass
    // input shape: (batch, seq_len, hidden_size)
    // returns: (batch, seq_len, hidden_size)
    Tensor forward(
        const Tensor& input,
        Tensor& q_weight, Tensor& q_bias,
        Tensor& k_weight, Tensor& k_bias,
        Tensor& v_weight, Tensor& v_bias,
        Tensor& out_weight, Tensor& out_bias,
        bool use_cache = false
    );
    
    // With KV cache for generation
    Tensor forward_with_cache(
        const Tensor& input,
        Tensor& q_weight, Tensor& q_bias,
        Tensor& k_weight, Tensor& k_bias,
        Tensor& v_weight, Tensor& v_bias,
        Tensor& out_weight, Tensor& out_bias,
        Tensor& cached_k, Tensor& cached_v
    );
    
private:
    int hidden_size_;
    int num_heads_;
    int head_dim_;
    float scale_;
};

#endif // LAYERS_H
