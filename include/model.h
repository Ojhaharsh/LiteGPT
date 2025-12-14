#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include <string>
#include <map>
#include <memory>

class Model {
public:
    Model();
    ~Model();
    
    // Load model weights from binary file (real GPT-2 weights)
    bool load_weights(const std::string& weights_file, const std::string& config_file, bool verbose = false);
    
    // Load config from JSON
    bool load_config(const std::string& config_file);
    
    // Get tensor by name
    Tensor* get_tensor(const std::string& name);
    const Tensor* get_tensor(const std::string& name) const;
    
    // Model dimensions
    int vocab_size;
    int block_size;           // Context length (max sequence length)
    int embedding_dim;        // Hidden size
    int num_heads;            // Number of attention heads
    int num_layers;           // Number of transformer layers
    int intermediate_size;    // MLP hidden dimension
    
    // All weights stored in a map for easy access
    // Keys match GPT-2 weight names from HuggingFace
    std::map<std::string, Tensor> weights;
    
    // Scales for quantized weights
    std::map<std::string, float> scales;
    
    // Quick access to common layers
    Tensor* token_embedding;      // transformer.wte.weight
    Tensor* position_embedding;   // transformer.wpe.weight
    Tensor* final_ln_weight;      // transformer.ln_f.weight
    Tensor* final_ln_bias;        // transformer.ln_f.bias
    Tensor* lm_head;              // lm_head.weight
    
    bool is_loaded() const { return loaded_; }
    bool is_quantized() const { return quantized_; }
    
private:
    bool loaded_ = false;
    bool quantized_ = false;
    
    // Helper to parse tensor name and extract layer info
    struct TensorInfo {
        int layer_idx;
        std::string component;  // "attn", "mlp", "ln", etc.
        std::string param_type; // "weight", "bias"
    };
};

#endif // MODEL_H
