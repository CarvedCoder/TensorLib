#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <tensorlib/tokenizer/impl/bpe.h>
#include <unordered_map>
#include <utility>
#include <vector>
namespace tensorlib::tokenizer::impl::bpe {

PairFrequency countPairs(const std::vector<uint32_t>& bytes) {
    PairFrequency result;
    result.freq.reserve(bytes.size());
    result.order.reserve(bytes.size());
    for (const auto [first, second] : bytes | std::ranges::views::adjacent<2>) {
        uint64_t key = static_cast<uint64_t>((static_cast<uint64_t>(first) << 32) | second);
        auto [it, inserted] = result.freq.try_emplace(key, 0);

        if (inserted)
            result.order.emplace_back(key);
        ++it->second;
    }

    return result;
}
uint64_t getMostFrequentPair(const PairFrequency& pf) {
    uint64_t best = pf.order.front();
    auto best_freq = pf.freq.at(best);
    for (uint64_t key : pf.order) {
        auto f = pf.freq.at(key);
        if (f > best_freq) {
            best = key;
            best_freq = f;
        }
    }
    return best;
}

std::pair<uint32_t, uint32_t> decodePair(const uint64_t& val) {
    return {static_cast<uint32_t>(val >> 32), static_cast<uint32_t>(val)};
}

std::vector<uint32_t> merge(const std::vector<uint32_t>& tokens_list,
                            const std::pair<uint32_t, uint32_t>& decodedPair, uint32_t index) {
    uint32_t first = decodedPair.first;
    uint32_t second = decodedPair.second;
    std::vector<uint32_t> new_tokens;
    new_tokens.reserve(tokens_list.size());
    size_t i = 0;
    while (i < tokens_list.size()) {
        if (i + 1 < tokens_list.size() && tokens_list[i] == first && tokens_list[i + 1] == second) {
            new_tokens.push_back(index);
            i += 2;
        } else {
            new_tokens.push_back(tokens_list[i]);
            i++;
        }
    }
    return new_tokens;
}

} // namespace tensorlib::tokenizer::impl::bpe
