#include "model.h"
#include <fstream>
#include <iostream>
#include <cstring>

// Simple JSON parser for config (minimal)
#include <sstream>

Model::Model() 
    : vocab_size(50257), block_size(1024), embedding_dim(768), 
      num_heads(12), num_layers(12), intermediate_size(3072),
      token_embedding(nullptr), position_embedding(nullptr),
      final_ln_weight(nullptr), final_ln_bias(nullptr), lm_head(nullptr) {
}

Model::~Model() {}

bool Model::load_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << config_file << std::endl;
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    file.close();
    
    // Simple JSON parsing (looking for key values)
    auto find_int = [&content](const std::string& key) -> int {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return -1;
        
        pos = content.find(":", pos);
        if (pos == std::string::npos) return -1;
        
        // Skip whitespace and find number
        pos = content.find_first_not_of(" \t\n\r", pos + 1);
        if (pos == std::string::npos) return -1;
        
        size_t end = content.find_first_of(",}", pos);
        if (end == std::string::npos) end = content.length();
        
        std::string num_str = content.substr(pos, end - pos);
        try {
            return std::stoi(num_str);
        } catch (...) {
            return -1;
        }
    };
    
    int vocab_sz = find_int("vocab_size");
    int hidden_sz = find_int("hidden_size");
    int num_layers_cfg = find_int("num_hidden_layers");
    int num_heads_cfg = find_int("num_attention_heads");
    int max_pos = find_int("max_position_embeddings");
    int inter_sz = find_int("intermediate_size");
    
    if (vocab_sz > 0) vocab_size = vocab_sz;
    if (hidden_sz > 0) embedding_dim = hidden_sz;
    if (num_layers_cfg > 0) num_layers = num_layers_cfg;
    if (num_heads_cfg > 0) num_heads = num_heads_cfg;
    if (max_pos > 0) block_size = max_pos;
    if (inter_sz > 0) intermediate_size = inter_sz;
    
    std::cout << "Loaded config:" << std::endl;
    std::cout << "  Vocab size: " << vocab_size << std::endl;
    std::cout << "  Hidden size: " << embedding_dim << std::endl;
    std::cout << "  Num layers: " << num_layers << std::endl;
    std::cout << "  Num heads: " << num_heads << std::endl;
    std::cout << "  Max position: " << block_size << std::endl;
    std::cout << "  Intermediate size: " << intermediate_size << std::endl;
    
    return true;
}

bool Model::load_weights(const std::string& weights_file, const std::string& config_file, bool verbose) {
    // Load config first
    if (!load_config(config_file)) {
        std::cerr << "Warning: Could not load config, using defaults" << std::endl;
    }
    
    // Open weights file
    std::ifstream file(weights_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open weights file: " << weights_file << std::endl;
        return false;
    }
    
    if (verbose) {
        std::cout << "\nLoading weights from: " << weights_file << std::endl;
    }
    
    // Read header: number of tensors
    uint32_t num_tensors;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(uint32_t));
    
    if (verbose) {
        std::cout << "Loading " << num_tensors << " tensors..." << std::endl;
    }
    
    std::map<std::string, std::vector<int8_t>> quant_packed;
    std::map<std::string, std::vector<int>> quant_shapes;
    int loaded_count = 0;
    while (file.good() && loaded_count < (int)num_tensors) {
        // Read tensor name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        if (!file.good()) break;
        
        std::string tensor_name(name_len, ' ');
        file.read(&tensor_name[0], name_len);
        
        // Read shape
        uint32_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));
        std::vector<int> shape(num_dims);
        for (uint32_t i = 0; i < num_dims; ++i) {
            uint32_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
            shape[i] = static_cast<int>(dim);
        }
        
        // Read data size
        uint32_t num_elements;
        file.read(reinterpret_cast<char*>(&num_elements), sizeof(uint32_t));
        
        if (tensor_name.find("_scale") != std::string::npos) {
            // Scale tensor
            Tensor tensor(shape);
            file.read(reinterpret_cast<char*>(tensor.data()), num_elements * sizeof(float));
            scales[tensor_name] = tensor.data()[0];
        } else if (quantized_) {
            // Quantized tensor
            std::vector<int8_t> data(num_elements);
            file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(int8_t));
            quant_packed[tensor_name] = data;
            quant_shapes[tensor_name] = shape;
        } else {
            // Normal float tensor
            Tensor tensor(shape);
            file.read(reinterpret_cast<char*>(tensor.data()), num_elements * sizeof(float));
            weights[tensor_name] = tensor;
            
            // Set quick-access pointers for common tensors
            if (tensor_name == "transformer.wte.weight") {
                token_embedding = &weights[tensor_name];
            }
            else if (tensor_name == "transformer.wpe.weight") {
                position_embedding = &weights[tensor_name];
            }
            else if (tensor_name == "transformer.ln_f.weight") {
                final_ln_weight = &weights[tensor_name];
            }
            else if (tensor_name == "transformer.ln_f.bias") {
                final_ln_bias = &weights[tensor_name];
            }
            else if (tensor_name == "lm_head.weight") {
                lm_head = &weights[tensor_name];
            }
        }
        
        if (verbose && loaded_count % 5 == 0) {
            std::cout << "  ✓ " << tensor_name << " shape=" << shape[0];
            if (shape.size() > 1) std::cout << "x" << shape[1];
            std::cout << std::endl;
        }
        
        loaded_count++;
    }
    
    // Dequantize quantized tensors
    for (auto& p : quant_packed) {
        std::string name = p.first;
        auto& packed = p.second;
        auto& shape = quant_shapes[name];
        std::string scale_name = name + "_scale";
        float scale = scales[scale_name];
        Tensor tensor(shape);
        std::vector<int8_t> unpacked;
        for (int8_t p : packed) {
            int8_t low = p & 0xF;
            int8_t high = (p >> 4) & 0xF;
            if (low > 7) low -= 16;
            if (high > 7) high -= 16;
            unpacked.push_back(low);
            unpacked.push_back(high);
        }
        size_t num_elements = 1;
        for (int s : shape) num_elements *= s;
        for (size_t i = 0; i < num_elements; ++i) {
            tensor.data()[i] = static_cast<float>(unpacked[i]) / scale;
        }
        weights[name] = tensor;
        // Set quick-access pointers for common tensors
        if (name == "transformer.wte.weight") {
            token_embedding = &weights[name];
        }
        else if (name == "transformer.wpe.weight") {
            position_embedding = &weights[name];
        }
        else if (name == "transformer.ln_f.weight") {
            final_ln_weight = &weights[name];
        }
        else if (name == "transformer.ln_f.bias") {
            final_ln_bias = &weights[name];
        }
        else if (name == "lm_head.weight") {
            lm_head = &weights[name];
        }
    }
    
    file.close();
    
    if (loaded_count == 0) {
        std::cerr << "Error: No tensors loaded from file" << std::endl;
        return false;
    }
    
    loaded_ = true;
    if (verbose) {
        std::cout << "\n✓ Successfully loaded " << loaded_count << " weight tensors!" << std::endl;
    }
    return true;
}

Tensor* Model::get_tensor(const std::string& name) {
    auto it = weights.find(name);
    if (it != weights.end()) {
        return &it->second;
    }
    return nullptr;
}

const Tensor* Model::get_tensor(const std::string& name) const {
    auto it = weights.find(name);
    if (it != weights.end()) {
        return &it->second;
    }
    return nullptr;
}
