#include "embedding.h"
#include <stdexcept>

Embedding::Embedding(int vocab_size, int embedding_dim)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim) {}

Tensor Embedding::forward(const std::vector<int>& token_ids, const Tensor& weight) {
    // weight shape: [vocab_size, embedding_dim]
    // output shape: [1, seq_len, embedding_dim]
    
    int seq_len = token_ids.size();
    Tensor output({1, seq_len, embedding_dim_});
    
    // For each token, get its embedding from the weight matrix
    for (int i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        
        if (token_id < 0 || token_id >= vocab_size_) {
            throw std::out_of_range("Token ID out of vocabulary range");
        }
        
        // Copy row from weight matrix to output
        const float* weight_data = weight.data();
        float* output_data = output.data();
        
        // Get source pointer (token_id-th row of weight matrix)
        const float* src = weight_data + token_id * embedding_dim_;
        
        // Get destination pointer (i-th token in output)
        float* dst = output_data + i * embedding_dim_;
        
        // Copy embedding
        for (int j = 0; j < embedding_dim_; ++j) {
            dst[j] = src[j];
        }
    }
    
    return output;
}

Tensor Embedding::get_embedding(int token_id, const Tensor& weight) {
    // Get single embedding vector
    // weight shape: [vocab_size, embedding_dim]
    // output shape: [embedding_dim]
    
    if (token_id < 0 || token_id >= vocab_size_) {
        throw std::out_of_range("Token ID out of vocabulary range");
    }
    
    Tensor output({embedding_dim_});
    const float* weight_data = weight.data();
    float* output_data = output.data();
    
    // Get source pointer
    const float* src = weight_data + token_id * embedding_dim_;
    
    // Copy embedding
    for (int j = 0; j < embedding_dim_; ++j) {
        output_data[j] = src[j];
    }
    
    return output;
}
