#include "tensorlib/tokenizer/impl/utf8.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <print>
#include <tensorlib/tokenizer/impl/bpe.h>
#include <tensorlib/tokenizer/trainer.h>
#include <vector>
namespace tensorlib::tokenizer {

Tokenizer BPETrainer::train(const std::filesystem::path& data_path, uint32_t vocab_size) {
    uint32_t num_merges = vocab_size - 256;
    std::string contents = impl::utf8::readFile(data_path);
    std::vector<uint32_t> tokens = impl::utf8::encodeToBytes(contents);
    using Clock = std::chrono::system_clock;

    auto start = Clock::now();
    for (size_t i = 0; i < num_merges; i++) {
        auto t1 = Clock::now();
        auto pairs = impl::bpe::countPairs(tokens);
        auto t2 = Clock::now();

        auto frequentPair = getMostFrequentPair(pairs);
        auto t3 = Clock::now();

        tokens = impl::bpe::merge(tokens, impl::bpe::decodePair(frequentPair),
                                  static_cast<uint32_t>(256 + i));
        auto t4 = Clock::now();

        std::println("count = {:.6f}", std::chrono::duration<double>(t2 - t1).count());

        std::println("max = {:.6f}", std::chrono::duration<double>(t3 - t2).count());

        std::println("merge = {:.6f}", std::chrono::duration<double>(t4 - t3).count());
    }
    auto end = Clock::now();
    std::println("Completed training final size is {}", tokens.size());
    std::println("Total time taken is {} ", std::chrono::duration<double>(end - start).count());
    return Tokenizer(tokens);
}
} // namespace tensorlib::tokenizer
