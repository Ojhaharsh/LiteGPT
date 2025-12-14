#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer() = default;

    // Load tokenizer from vocab and merges files
    bool load_vocab(const std::string& vocab_file);
    bool load_merges(const std::string& merges_file);

    // Tokenization
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
    std::string decode_single(int token_id);

    // Utility
    int get_vocab_size() const { return vocab_size_; }
    bool is_loaded() const { return loaded_; }

private:
    // BPE encoding
    std::vector<int> bpe_encode(const std::string& text);
    std::string bytes_to_unicode_decode(const std::vector<int>& bytes);

    // State
    std::unordered_map<std::string, int> encoder_;  // token -> id
    std::map<int, std::string> decoder_;            // id -> token
    std::vector<std::pair<std::string, std::string>> bpe_merges_;
    std::map<std::pair<std::string, std::string>, int> bpe_rank_;
    int vocab_size_ = 0;
    bool loaded_ = false;

    // Helper functions
    std::vector<std::string> split_into_bytes(const std::string& text);
    std::pair<std::string, std::string> get_stats(const std::vector<std::string>& vocab);
    std::vector<std::string> merge_vocab(std::vector<std::string> vocab, 
                                         const std::pair<std::string, std::string>& pair);
};
