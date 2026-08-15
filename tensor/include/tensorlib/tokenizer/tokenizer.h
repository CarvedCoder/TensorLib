#pragma once
#include <cstdint>
#include <string>
#include <tensorlib/tokenizer/impl/utf8.h>
#include <vector>

namespace tensorlib::tokenizer {

constexpr std::string_view MAGIC = "TLTK";
constexpr uint32_t FORMAT_VERSION = 1;
constexpr uint16_t INITIAL_VOCAB_SIZE = 256;

struct MergeRule {
    std::pair<uint16_t, uint16_t> merge_tokens;
    uint16_t result;
};

class Tokenizer {
  private:
    std::vector<uint16_t> m_trained_tokens;
    std::vector<MergeRule> m_merge_rules;
    std::vector<std::vector<uint8_t>> m_vocab;
    uint16_t m_vocab_size;

  public:
    Tokenizer() = default;

    Tokenizer(const std::vector<uint16_t>& trained_tokens,
              const std::vector<MergeRule>& merge_rules,
              const std::vector<std::vector<uint8_t>>& m_vocab, const uint16_t m_vocab_size);
    std::vector<uint16_t> encode(std::string_view text) const;
    std::string decode(std::span<const uint16_t> ids) const;
    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);
};
} // namespace tensorlib::tokenizer
