#pragma once
#include "tensorlib/tokenizer/tokenizer.h"
#include <cstdint>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>
namespace tensorlib::tokenizer::impl::bpe {

struct PairFrequency {
    std::unordered_map<uint32_t, int> freq;
    std::vector<uint32_t> order;
};
PairFrequency countPairs(const std::vector<uint16_t>& bytes);
uint32_t getMostFrequentPair(const PairFrequency& pf);
std::pair<uint16_t, uint16_t> decodePair(const uint32_t& val);
void merge(std::vector<uint16_t>& tokens_list, const std::pair<uint16_t, uint16_t>& decodedPair,
           uint16_t index);
std::vector<MergeRule> createMegreRules(uint16_t num_merges);
std::vector<std::vector<uint8_t>> createVocab(uint16_t vocab_size);
void buildVocab(std::vector<std::vector<uint8_t>>& vocab, const MergeRule& merge_rule);

} // namespace tensorlib::tokenizer::impl::bpe
