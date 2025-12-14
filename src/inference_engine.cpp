#include "inference_engine.h"
#include "matmul.h"
#include "layers.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

LLMInferenceEngine::LLMInferenceEngine() {
    top_k_ = 40;
    top_p_ = 0.9f;
}

bool LLMInferenceEngine::initialize(const std::string& weights_file,
                                   const std::string& config_file,
                                   const std::string& vocab_file,
                                   const std::string& merges_file) {
    // Load model weights
    if (!model_.load_weights(weights_file, config_file, true)) {
        std::cerr << "Failed to load model weights" << std::endl;
        return false;
    }

    // Load tokenizer
    if (!tokenizer_.load_vocab(vocab_file)) {
        std::cerr << "Failed to load tokenizer vocab" << std::endl;
        return false;
    }

    if (!merges_file.empty() && !tokenizer_.load_merges(merges_file)) {
        std::cerr << "Warning: Failed to load BPE merges" << std::endl;
    }

    return true;
}

Tensor LLMInferenceEngine::forward(const std::vector<int>& input_ids) {
    // Use actual forward pass
    return forward_pass_.forward(input_ids, model_);
}

std::vector<float> LLMInferenceEngine::get_logits(const std::vector<int>& input_ids) {
    // Get actual logits from forward pass
    return forward_pass_.get_logits(input_ids, model_);
}

int LLMInferenceEngine::sample_greedy(const std::vector<float>& logits) {
    // Find index with maximum logit
    int max_idx = 0;
    float max_val = logits[0];

    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    return max_idx;
}

int LLMInferenceEngine::sample_top_k(const std::vector<float>& logits, int k) {
    // Keep only top-k logits, set others to -inf
    std::vector<std::pair<float, int>> sorted_logits;

    for (size_t i = 0; i < logits.size(); ++i) {
        sorted_logits.push_back({logits[i], i});
    }

    std::sort(sorted_logits.rbegin(), sorted_logits.rend());

    // Create filtered logits
    std::vector<float> filtered(logits.size(), -1e9f);
    for (int i = 0; i < k && i < (int)sorted_logits.size(); ++i) {
        filtered[sorted_logits[i].second] = sorted_logits[i].first;
    }

    // Apply softmax and sample
    float max_val = *std::max_element(filtered.begin(), filtered.end());
    std::vector<float> probs(filtered.size());
    float sum = 0.0f;

    for (size_t i = 0; i < filtered.size(); ++i) {
        probs[i] = std::exp(filtered[i] - max_val);
        sum += probs[i];
    }

    for (auto& p : probs) p /= sum;

    // Sample from distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float rand_val = dist(rng_);
    float cumsum = 0.0f;

    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (rand_val < cumsum) {
            return i;
        }
    }

    return 0;
}

int LLMInferenceEngine::sample_top_p(const std::vector<float>& logits, float p) {
    // Nucleus sampling (top-p)
    std::vector<std::pair<float, int>> sorted_logits;

    for (size_t i = 0; i < logits.size(); ++i) {
        sorted_logits.push_back({logits[i], i});
    }

    std::sort(sorted_logits.rbegin(), sorted_logits.rend());

    // Apply softmax
    float max_val = sorted_logits[0].first;
    std::vector<float> probs(logits.size(), 0.0f);
    float sum = 0.0f;

    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }

    for (auto& pr : probs) pr /= sum;

    // Cumulative sum and filter by top-p
    float cumsum = 0.0f;
    std::vector<float> filtered_probs(logits.size(), 0.0f);

    for (const auto& [logit, idx] : sorted_logits) {
        cumsum += probs[idx];
        filtered_probs[idx] = probs[idx];

        if (cumsum > p) break;
    }

    // Renormalize
    float filtered_sum = 0.0f;
    for (auto pr : filtered_probs) filtered_sum += pr;

    if (filtered_sum > 0) {
        for (auto& pr : filtered_probs) pr /= filtered_sum;
    }

    // Sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float rand_val = dist(rng_);
    float cs = 0.0f;

    for (size_t i = 0; i < filtered_probs.size(); ++i) {
        cs += filtered_probs[i];
        if (rand_val < cs) {
            return i;
        }
    }

    return 0;
}

std::string LLMInferenceEngine::generate(const std::string& prompt, 
                                        int max_tokens,
                                        float temperature) {
    // Encode prompt
    std::vector<int> input_ids = tokenizer_.encode(prompt);

    std::string generated_text = prompt;

    // Generate tokens one by one
    for (int i = 0; i < max_tokens; ++i) {
        // Get logits for next token
        std::vector<float> logits;
        try {
            logits = get_logits(input_ids);
        } catch (const std::exception& e) {
            std::cout << "Exception in get_logits: " << e.what() << std::endl;
            return generated_text;
        }

        // Apply temperature
        if (temperature != 1.0f) {
            for (auto& logit : logits) {
                logit /= temperature;
            }
        }

        // Sample next token
        int next_token;
        if (top_k_ > 0) {
            next_token = sample_top_k(logits, top_k_);
        } else if (top_p_ < 1.0f) {
            next_token = sample_top_p(logits, top_p_);
        } else {
            next_token = sample_greedy(logits);
        }

        // Check for end-of-sequence token
        // if (next_token == 50256) {  // GPT-2 EOS token
        //     break;
        // }

        // Decode and append
        std::string token_str = tokenizer_.decode_single(next_token);
        generated_text += token_str;

        // Add to input for next iteration
        input_ids.push_back(next_token);

        // Limit context to model's max length
        if (input_ids.size() > 1024) {
            input_ids.erase(input_ids.begin());
        }
    }

    return generated_text;
}
