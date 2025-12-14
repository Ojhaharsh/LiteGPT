#pragma once

#include "model.h"
#include "tokenizer.h"
#include "forward_pass.h"
#include "tensor.h"
#include <vector>
#include <string>
#include <random>

class LLMInferenceEngine {
public:
    LLMInferenceEngine();
    ~LLMInferenceEngine() = default;

    // Initialize model and tokenizer
    bool initialize(const std::string& weights_file, 
                   const std::string& config_file,
                   const std::string& vocab_file,
                   const std::string& merges_file = "");

    // Single forward pass
    Tensor forward(const std::vector<int>& input_ids);

    // Generate text
    std::string generate(const std::string& prompt, int max_tokens = 50, float temperature = 1.0f);

    // Get probabilities for next token
    std::vector<float> get_logits(const std::vector<int>& input_ids);

    // Settings
    void set_temperature(float temp) { temperature_ = temp; }
    void set_top_k(int k) { top_k_ = k; }
    void set_top_p(float p) { top_p_ = p; }

private:
    Model model_;
    Tokenizer tokenizer_;
    ForwardPass forward_pass_;
    float temperature_ = 1.0f;
    int top_k_ = 0;  // 0 = disabled
    float top_p_ = 1.0f;  // 1.0 = disabled

    // Sampling methods
    int sample_greedy(const std::vector<float>& logits);
    int sample_top_k(const std::vector<float>& logits, int k);
    int sample_top_p(const std::vector<float>& logits, float p);

    // RNG
    std::mt19937 rng_{std::random_device{}()};
};
