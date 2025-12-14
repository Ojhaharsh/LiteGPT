#include "forward_pass.h"
#include "layers.h"
#include "matmul.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <limits>

ForwardPass::ForwardPass() {}

Tensor ForwardPass::embed_tokens(const std::vector<int>& token_ids, Tensor& embed_weight) {
    // Token embedding: [1, seq_len, 768]
    return embedding_.forward(token_ids, embed_weight);
}

Tensor ForwardPass::apply_position_embeddings(const Tensor& token_embeddings, Tensor& pos_embed_weight) {
    // Add position embeddings: token_embed + pos_embed
    // pos_embed_weight shape: [1024, 768]
    // token_embeddings shape: [1, seq_len, 768]
    
    Tensor output = token_embeddings;
    int seq_len = token_embeddings.shape()[1];
    int hidden_size = 768;
    
    float* output_data = output.data();
    const float* pos_data = pos_embed_weight.data();
    
    // Add position embeddings
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            int out_idx = i * hidden_size + j;
            int pos_idx = i * hidden_size + j;
            output_data[out_idx] += pos_data[pos_idx];
        }
    }
    
    return output;
}

Tensor ForwardPass::attention_layer(const Tensor& hidden_state, int layer_idx, Model& model) {
    // Full attention implementation
    // std::cout << "  [DEBUG] Attention layer " << layer_idx << " start" << std::endl;
    auto shape = hidden_state.shape();
    int batch_size = shape[0];
    int seq_len = shape[1];
    int hidden_size = shape[2];
    
    // Reshape to 2D for matmul
    Tensor hidden_2d({batch_size * seq_len, hidden_size});
    std::copy(hidden_state.data(), hidden_state.data() + hidden_state.total_elements(), 
              hidden_2d.data());
    
    // Get attention weights
    char layer_str[100];
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.attn.c_attn.weight", layer_idx);
    auto c_attn_weight_it = model.weights.find(layer_str);
    
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.attn.c_attn.bias", layer_idx);
    auto c_attn_bias_it = model.weights.find(layer_str);
    
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.attn.c_proj.weight", layer_idx);
    auto c_proj_weight_it = model.weights.find(layer_str);
    
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.attn.c_proj.bias", layer_idx);
    auto c_proj_bias_it = model.weights.find(layer_str);
    
    if (c_attn_weight_it == model.weights.end() || c_proj_weight_it == model.weights.end()) {
        std::cerr << "Attention weights not found for layer " << layer_idx << std::endl;
        return hidden_state;
    }
    
    Tensor& c_attn_weight = c_attn_weight_it->second;
    Tensor& c_attn_bias = c_attn_bias_it->second;
    Tensor& c_proj_weight = c_proj_weight_it->second;
    Tensor& c_proj_bias = c_proj_bias_it->second;
    
    // Project to QKV
    Tensor qkv({batch_size * seq_len, 3 * hidden_size});
    matmul_bias(hidden_2d, c_attn_weight, c_attn_bias, qkv);
    // std::cout << "  [DEBUG] QKV projection done" << std::endl;
    
    // Split QKV
    Tensor q({batch_size * seq_len, hidden_size});
    Tensor k({batch_size * seq_len, hidden_size});
    Tensor v({batch_size * seq_len, hidden_size});
    
    for (int i = 0; i < batch_size * seq_len; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            q.data()[i * hidden_size + j] = qkv.data()[i * 3 * hidden_size + j];
            k.data()[i * hidden_size + j] = qkv.data()[i * 3 * hidden_size + hidden_size + j];
            v.data()[i * hidden_size + j] = qkv.data()[i * 3 * hidden_size + 2 * hidden_size + j];
        }
    }
    // std::cout << "  [DEBUG] QKV split done" << std::endl;
    // printf("  [DEBUG] QKV split done\n");
    
    // Multi-head attention
    int num_heads = model.num_heads;
    int head_dim = hidden_size / num_heads;
    Tensor attn_out({batch_size * seq_len, hidden_size});
    // std::cout << "  [DEBUG] Starting multi-head attention with " << num_heads << " heads" << std::endl;
    for (int head = 0; head < num_heads; ++head) {
        // std::cout << "  [DEBUG] Processing head " << head << std::endl;
        int offset = head * head_dim;
        
        // Extract head
        Tensor q_head({batch_size * seq_len, head_dim});
        Tensor k_head({batch_size * seq_len, head_dim});
        Tensor v_head({batch_size * seq_len, head_dim});
        // Use std::copy for extraction
        for (int i = 0; i < batch_size * seq_len; ++i) {
            std::copy(q.data() + i * hidden_size + offset, q.data() + i * hidden_size + offset + head_dim, q_head.data() + i * head_dim);
            std::copy(k.data() + i * hidden_size + offset, k.data() + i * hidden_size + offset + head_dim, k_head.data() + i * head_dim);
            std::copy(v.data() + i * hidden_size + offset, v.data() + i * hidden_size + offset + head_dim, v_head.data() + i * head_dim);
        }
        // std::cout << "  [DEBUG] Head extraction done for head " << head << std::endl;
        // printf("  [DEBUG] Head extraction done\n");
        
        // Attention scores: Q_head @ K_head^T / sqrt(d_k)
        Tensor k_head_t = transpose_2d(k_head);
        Tensor scores({batch_size * seq_len, batch_size * seq_len});
        matmul(q_head, k_head_t, scores);
        scores.mul(1.0f / std::sqrt((float)head_dim));
        // std::cout << "  [DEBUG] Attention scores computed for head " << head << std::endl;
        
        // Causal mask
        for (int i = 0; i < batch_size * seq_len; ++i) {
            for (int j = i + 1; j < batch_size * seq_len; ++j) {
                scores.data()[i * batch_size * seq_len + j] = -std::numeric_limits<float>::infinity();
            }
        }
        
        // Softmax
        softmax(scores);
        // std::cout << "  [DEBUG] Softmax applied for head " << head << std::endl;
        
        // Attention output: scores @ V_head
        Tensor attn_head({batch_size * seq_len, head_dim});
        matmul(scores, v_head, attn_head);
        // std::cout << "  [DEBUG] Attention output computed for head " << head << std::endl;
        // printf("  [DEBUG] Attention output computed\n");
        
        // Store in attn_out
        for (int i = 0; i < batch_size * seq_len; ++i) {
            for (int j = 0; j < head_dim; ++j) {
                attn_out.data()[i * hidden_size + offset + j] = attn_head.data()[i * head_dim + j];
            }
        }
        // std::cout << "  [DEBUG] Head " << head << " stored in attn_out" << std::endl;
    }
    // std::cout << "  [DEBUG] Loop ended" << std::endl;
    // std::cout << "  [DEBUG] All heads processed, starting projection" << std::endl;
    
    // Project output
    Tensor out_2d({batch_size * seq_len, hidden_size});
    // std::cout << "  [DEBUG] Created out_2d tensor" << std::endl;
    matmul_bias(attn_out, c_proj_weight, c_proj_bias, out_2d);
    // std::cout << "  [DEBUG] Projection done" << std::endl;
    
    // Reshape back to 3D
    Tensor out_3d({batch_size, seq_len, hidden_size});
    std::copy(out_2d.data(), out_2d.data() + out_2d.total_elements(), out_3d.data());
    
    // printf("  [DEBUG] Attention layer %d end\n", layer_idx);
    return out_3d;
}

Tensor ForwardPass::mlp_layer(const Tensor& hidden_state, int layer_idx, Model& model) {
    // MLP: hidden -> 3072 -> hidden
    // Pattern: Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)
    
    auto shape = hidden_state.shape();
    int batch_size = shape[0];
    int seq_len = shape[1];
    int hidden_size = 768;
    int intermediate_size = 3072;
    
    // Reshape to 2D for matmul
    Tensor hidden_2d({batch_size * seq_len, hidden_size});
    std::copy(hidden_state.data(), hidden_state.data() + hidden_state.total_elements(), 
              hidden_2d.data());
    
    // Get layer weights
    char layer_str[100];
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.mlp.c_fc.weight", layer_idx);
    auto fc_weight_it = model.weights.find(layer_str);
    
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.mlp.c_fc.bias", layer_idx);
    auto fc_bias_it = model.weights.find(layer_str);
    
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.mlp.c_proj.weight", layer_idx);
    auto proj_weight_it = model.weights.find(layer_str);
    
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.mlp.c_proj.bias", layer_idx);
    auto proj_bias_it = model.weights.find(layer_str);
    
    if (fc_weight_it == model.weights.end() || fc_bias_it == model.weights.end() ||
        proj_weight_it == model.weights.end() || proj_bias_it == model.weights.end()) {
        // Weights not found, return input unchanged
        return hidden_state;
    }
    
    // FC layer: hidden -> intermediate
    Tensor fc_output({batch_size * seq_len, intermediate_size});
    matmul_bias(hidden_2d, fc_weight_it->second, fc_bias_it->second, fc_output);
    
    // Apply GELU
    gelu(fc_output);
    
    // Projection layer: intermediate -> hidden
    Tensor output_2d({batch_size * seq_len, hidden_size});
    matmul_bias(fc_output, proj_weight_it->second, proj_bias_it->second, output_2d);
    
    // Reshape back to 3D
    Tensor output(shape);
    std::copy(output_2d.data(), output_2d.data() + output_2d.total_elements(), 
              output.data());
    
    return output;
}

Tensor ForwardPass::transformer_block(const Tensor& hidden_state, int block_idx, Model& model) {
    // Single transformer block:
    // 1. LayerNorm + Attention + Residual
    // 2. LayerNorm + MLP + Residual
    
    Tensor x = hidden_state;
    
    // Attention sub-block
    char layer_str[100];
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.ln_1.weight", block_idx);
    auto ln1_weight_it = model.weights.find(layer_str);
    
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.ln_1.bias", block_idx);
    auto ln1_bias_it = model.weights.find(layer_str);
    
    if (ln1_weight_it != model.weights.end() && ln1_bias_it != model.weights.end()) {
        LayerNorm ln(768);
        Tensor ln_out = ln.forward(x, ln1_weight_it->second, ln1_bias_it->second);
        Tensor attn_out = attention_layer(ln_out, block_idx, model);
        
        // Add residual
        // #pragma omp parallel for
        for (int i = 0; i < (int)x.total_elements(); ++i) {
            x.data()[i] = x.data()[i] + attn_out.data()[i];
        }
    }
    
    // MLP sub-block
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.ln_2.weight", block_idx);
    auto ln2_weight_it = model.weights.find(layer_str);
    
    snprintf(layer_str, sizeof(layer_str), "transformer.h.%d.ln_2.bias", block_idx);
    auto ln2_bias_it = model.weights.find(layer_str);
    
    if (ln2_weight_it != model.weights.end() && ln2_bias_it != model.weights.end()) {
        LayerNorm ln(768);
        Tensor ln_out = ln.forward(x, ln2_weight_it->second, ln2_bias_it->second);
        Tensor mlp_out = mlp_layer(ln_out, block_idx, model);
        
        // Add residual
        // #pragma omp parallel for
        for (int i = 0; i < (int)x.total_elements(); ++i) {
            x.data()[i] = x.data()[i] + mlp_out.data()[i];
        }
    }
    
    return x;
}

Tensor ForwardPass::forward_step(const std::vector<int>& token_ids, Model& model) {
    // Single forward pass step
    // std::cout << "  [DEBUG] Starting forward pass for " << token_ids.size() << " tokens..." << std::endl;
    
    // 1. Embed tokens
    auto embed_weight_it = model.weights.find("transformer.wte.weight");
    if (embed_weight_it == model.weights.end()) {
        throw std::runtime_error("Embedding weights not found");
    }
    
    Tensor hidden = embed_tokens(token_ids, embed_weight_it->second);
    // std::cout << "  [DEBUG] Token embedding done" << std::endl;
    
    // 2. Add position embeddings
    auto pos_weight_it = model.weights.find("transformer.wpe.weight");
    if (pos_weight_it != model.weights.end()) {
        hidden = apply_position_embeddings(hidden, pos_weight_it->second);
    }
    // std::cout << "  [DEBUG] Position embeddings done" << std::endl;
    
    // 3. Pass through all transformer blocks (12 layers)
    for (int i = 0; i < model.num_layers; ++i) {
        // std::cout << "  [DEBUG] Starting Layer " << i << "..." << std::endl;
        hidden = transformer_block(hidden, i, model);
        // std::cout << "  [DEBUG] Finished Layer " << i << std::endl;
    }
    
    // std::cout << "  [DEBUG] Forward pass complete, calculating logits..." << std::endl;
    
    // 4. Apply final layer norm
    auto ln_f_weight_it = model.weights.find("ln_f.weight");
    auto ln_f_bias_it = model.weights.find("ln_f.bias");
    
    if (ln_f_weight_it != model.weights.end() && ln_f_bias_it != model.weights.end()) {
        LayerNorm ln(768);
        hidden = ln.forward(hidden, ln_f_weight_it->second, ln_f_bias_it->second);
    }
    
    return hidden;
}

Tensor ForwardPass::forward(const std::vector<int>& token_ids, Model& model) {
    return forward_step(token_ids, model);
}

std::vector<float> ForwardPass::get_logits(const std::vector<int>& token_ids, Model& model) {
    // Get hidden state from forward pass
    // printf("  [DEBUG] Starting forward pass for %zu tokens...\n", token_ids.size());
    Tensor hidden = forward_step(token_ids, model);
    
    // Project to vocabulary size: hidden [1, seq_len, 768] -> logits [1, seq_len, 50257]
    // We only care about the last token: hidden[0, -1, :] -> logits [50257]
    
    int seq_len = hidden.shape()[1];
    int hidden_size = 768;
    int vocab_size = 50257;
    
    std::vector<float> logits(vocab_size, 0.0f);
    
    // Simple projection: x @ W where W is embedding.weight transposed
    // In GPT-2, the output projection uses the embedding weights transposed
    auto embed_weight_it = model.weights.find("transformer.wte.weight");
    if (embed_weight_it == model.weights.end()) {
        // Return uniform distribution if weights not found
        std::fill(logits.begin(), logits.end(), 1.0f / vocab_size);
        return logits;
    }
    
    // Get last token hidden state
    const float* hidden_data = hidden.data();
    const float* embed_data = embed_weight_it->second.data();
    
    // Last token is at position [seq_len - 1]
    const float* last_hidden = hidden_data + (seq_len - 1) * hidden_size;
    
    // Compute logits = last_hidden @ embedding.weight^T
    // #pragma omp parallel for
    for (int i = 0; i < vocab_size; ++i) {
        float logit = 0.0f;
        const float* embed_row = embed_data + i * hidden_size;
        
        for (int j = 0; j < hidden_size; ++j) {
            logit += last_hidden[j] * embed_row[j];
        }
        
        logits[i] = logit;
    }
    
    // Check for nan
    for (float l : logits) {
        if (std::isnan(l)) {
            std::fill(logits.begin(), logits.end(), 1.0f / vocab_size);
            break;
        }
    }
    
    return logits;
}
