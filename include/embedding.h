#pragma once

#include "tensor.h"
#include <vector>

class Embedding {
public:
    Embedding(int vocab_size, int embedding_dim);
    ~Embedding() = default;

    // Forward pass: token_ids -> embeddings
    Tensor forward(const std::vector<int>& token_ids, const Tensor& weight);

    // Get embedding for single token
    Tensor get_embedding(int token_id, const Tensor& weight);

    int vocab_size() const { return vocab_size_; }
    int embedding_dim() const { return embedding_dim_; }

private:
    int vocab_size_;
    int embedding_dim_;
};
