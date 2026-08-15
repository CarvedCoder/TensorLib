#include "tensorlib/tokenizer/tokenizer.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <print>
#include <tensorlib/tokenizer/impl/bpe.h>
#include <tensorlib/tokenizer/trainer.h>
#include <vector>
namespace tensorlib::tokenizer {

Tokenizer BPETrainer::train(const std::filesystem::path& data_path, uint16_t vocab_size) {
    uint16_t num_merges = vocab_size - 256;

    auto vocab = impl::bpe::createVocab(vocab_size);
    auto merge_rules = impl::bpe::createMegreRules(num_merges);

    std::string contents = impl::utf8::readFile(data_path);
    std::vector<uint16_t> tokens = impl::utf8::encodeToBytes(contents);

    using Clock = std::chrono::system_clock;

    auto start = Clock::now();

    for (size_t i = 0; i < num_merges; i++) {
        auto index = static_cast<uint16_t>(256 + i);
        auto pairs = impl::bpe::countPairs(tokens);

        auto frequentPair = getMostFrequentPair(pairs);
        auto decoded_pair = impl::bpe::decodePair(frequentPair);

        MergeRule merge_rule = {.merge_tokens = decoded_pair, .result = index};
        merge_rules.push_back(merge_rule);
        impl::bpe::buildVocab(vocab, merge_rule);

        impl::bpe::merge(tokens, decoded_pair, index);
    }

    auto end = Clock::now();
    std::println("Completed training final size is {}", tokens.size());
    std::println("Total time taken is {} ", std::chrono::duration<double>(end - start).count());
    return Tokenizer(tokens, merge_rules, vocab, vocab_size);
}
} // namespace tensorlib::tokenizer
