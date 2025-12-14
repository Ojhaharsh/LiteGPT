#pragma once

#include "tensor.h"
#include "model.h"
#include "embedding.h"
#include <vector>

class ForwardPass {
public:
    ForwardPass();
    ~ForwardPass() = default;

    // Single step: token_ids -> next logits
    Tensor forward_step(const std::vector<int>& token_ids, Model& model);

    // Full sequence forward pass
    Tensor forward(const std::vector<int>& token_ids, Model& model);

    // Get next token logits (vocab probabilities)
    std::vector<float> get_logits(const std::vector<int>& token_ids, Model& model);

private:
    Embedding embedding_{50257, 768};  // GPT-2 vocab and hidden size

    // Helper functions for transformer block computation
    Tensor embed_tokens(const std::vector<int>& token_ids, Tensor& embed_weight);
    Tensor apply_position_embeddings(const Tensor& token_embeddings, Tensor& pos_embed_weight);
    Tensor transformer_block(const Tensor& hidden_state, int block_idx, Model& model);
    Tensor attention_layer(const Tensor& hidden_state, int layer_idx, Model& model);
    Tensor mlp_layer(const Tensor& hidden_state, int layer_idx, Model& model);
};
