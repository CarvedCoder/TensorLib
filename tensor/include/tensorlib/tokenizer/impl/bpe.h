#pragma once
#include <cstdint>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>
namespace tensorlib::tokenizer::impl::bpe {

struct PairFrequency {
    std::unordered_map<uint64_t, int> freq;
    std::vector<uint64_t> order;
};
PairFrequency countPairs(const std::vector<uint32_t>& bytes);
uint64_t getMostFrequentPair(const PairFrequency& pf);
std::pair<uint32_t, uint32_t> decodePair(const uint64_t& val);
std::vector<uint32_t> merge(const std::vector<uint32_t>& tokens_list,
                            const std::pair<uint32_t, uint32_t>& decodedPair, uint32_t index);
} // namespace tensorlib::tokenizer::impl::bpe
