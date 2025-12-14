#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

Tokenizer::Tokenizer() {}

bool Tokenizer::load_vocab(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        return false;
    }

    encoder_.clear();
    decoder_.clear();

    std::string line;
    int line_num = 0;

    // Simple JSON parser for vocab file
    // Format: { "token": id, ... }
    while (std::getline(file, line)) {
        line_num++;

        // Skip opening/closing braces
        if (line.find("{") != std::string::npos || line.find("}") != std::string::npos) {
            continue;
        }

        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty() || line[0] == '/') continue;

        // Parse "token": id,
        size_t quote1 = line.find('"');
        size_t quote2 = line.find('"', quote1 + 1);
        size_t colon = line.find(':', quote2);

        if (quote1 != std::string::npos && quote2 != std::string::npos && colon != std::string::npos) {
            std::string token = line.substr(quote1 + 1, quote2 - quote1 - 1);
            std::string id_str = line.substr(colon + 1);

            // Remove comma
            if (id_str.back() == ',') {
                id_str.pop_back();
            }

            // Remove whitespace
            id_str.erase(0, id_str.find_first_not_of(" \t"));
            id_str.erase(id_str.find_last_not_of(" \t") + 1);

            try {
                int token_id = std::stoi(id_str);
                encoder_[token] = token_id;
                decoder_[token_id] = token;
            } catch (...) {
                continue;
            }
        }
    }

    vocab_size_ = encoder_.size();
    return vocab_size_ > 0;
}

bool Tokenizer::load_merges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    int rank = 0;

    bpe_merges_.clear();
    bpe_rank_.clear();

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty()) continue;

        // Split by space
        std::istringstream iss(line);
        std::string first, second;
        if (iss >> first >> second) {
            bpe_merges_.push_back({first, second});
            bpe_rank_[{first, second}] = rank++;
        }
    }

    loaded_ = true;
    return true;
}

std::vector<std::string> Tokenizer::split_into_bytes(const std::string& text) {
    std::vector<std::string> result;

    for (unsigned char c : text) {
        // Convert each byte to a UTF-8 safe representation
        result.push_back(std::string(1, c));
    }

    // Add end-of-word token
    if (!result.empty()) {
        result.back() += "</w>";
    }

    return result;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> token_ids;

    // Simple approach: encode each byte/character
    // Full BPE would recursively merge pairs, but this is simplified

    for (unsigned char c : text) {
        // Try to find single character in vocab
        std::string ch(1, c);
        auto it = encoder_.find(ch);
        if (it != encoder_.end()) {
            token_ids.push_back(it->second);
        } else {
            // Fallback to unknown token
            auto unk = encoder_.find("<|endoftext|>");
            if (unk != encoder_.end()) {
                token_ids.push_back(unk->second);
            }
        }
    }

    return token_ids;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string result;

    for (int token_id : tokens) {
        auto it = decoder_.find(token_id);
        if (it != decoder_.end()) {
            result += it->second;
        }
    }

    // Clean up special tokens
    size_t pos = 0;
    while ((pos = result.find("</w>", pos)) != std::string::npos) {
        result.replace(pos, 4, " ");
        pos += 1;
    }

    return result;
}

std::string Tokenizer::decode_single(int token_id) {
    auto it = decoder_.find(token_id);
    if (it != decoder_.end()) {
        return it->second;
    }
    return "[UNK]";
}
