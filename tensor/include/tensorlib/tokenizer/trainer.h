#pragma once
#include <cstdint>
#include <filesystem>
#include <tensorlib/tokenizer/tokenizer.h>

namespace tensorlib::tokenizer {
class BPETrainer {
  public:
    static Tokenizer train(const std::filesystem::path& data_path, uint32_t vocab_size);
};
} // namespace tensorlib::tokenizer
