#include "tensorlib/tokenizer/tokenizer.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <tensorlib/tokenizer/impl/bpe.h>
#include <unordered_map>
#include <utility>
#include <vector>
namespace tensorlib::tokenizer::impl::bpe {

PairFrequency countPairs(const std::vector<uint16_t>& bytes) {
    PairFrequency result;
    result.freq.reserve(bytes.size());
    result.order.reserve(bytes.size());
    for (const auto [first, second] : bytes | std::ranges::views::adjacent<2>) {
        uint32_t key = static_cast<uint32_t>((static_cast<uint64_t>(first) << 16) | second);
        auto [it, inserted] = result.freq.try_emplace(key, 0);

        if (inserted)
            result.order.emplace_back(key);
        ++it->second;
    }

    return result;
}
uint32_t getMostFrequentPair(const PairFrequency& pf) {
    uint32_t best = pf.order.front();
    auto best_freq = pf.freq.at(best);
    for (uint32_t key : pf.order) {
        auto f = pf.freq.at(key);
        if (f > best_freq) {
            best = key;
            best_freq = f;
        }
    }
    return best;
}

std::pair<uint16_t, uint16_t> decodePair(const uint32_t& val) {
    return {static_cast<uint32_t>(val >> 16), static_cast<uint16_t>(val)};
}

void merge(std::vector<uint16_t>& tokens_list, const std::pair<uint16_t, uint16_t>& decodedPair,
           uint16_t index) {
    const auto [first, second] = decodedPair;
    size_t read = 0, write = 0;
    while (read < tokens_list.size()) {
        if (read + 1 < tokens_list.size() && tokens_list[read] == first &&
            tokens_list[read + 1] == second) {
            tokens_list[write++] = index;
            read += 2;
        } else {
            tokens_list[write++] = tokens_list[read++];
        }
    }
    tokens_list.resize(write);
}

std::vector<std::vector<uint8_t>> createVocab(uint16_t vocab_size) {
    std::vector<std::vector<uint8_t>> vocab(vocab_size);
    for (size_t i = 0; i < vocab_size; i++) {
        vocab[i].push_back(static_cast<uint8_t>(i));
    }
    return vocab;
}

std::vector<MergeRule> createMegreRules(uint16_t num_merges) {
    std::vector<MergeRule> merge_rules;
    merge_rules.reserve(num_merges);
    return merge_rules;
}

void buildVocab(std::vector<std::vector<uint8_t>>& vocab, const MergeRule& merge_rule) {
    const auto [first, second] = merge_rule.merge_tokens;
    auto& token = vocab[merge_rule.result];
    token.reserve(vocab[first].size() + vocab[second].size());
    token.insert(token.end(), vocab[first].begin(), vocab[first].end());
    token.insert(token.end(), vocab[second].begin(), vocab[second].end());
}

} // namespace tensorlib::tokenizer::impl::bpe
