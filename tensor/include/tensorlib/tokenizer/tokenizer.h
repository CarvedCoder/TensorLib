#pragma once
#include <tensorlib/tokenizer/impl/utf8.h>
#include <vector>

namespace tensorlib::tokenizer {
class Tokenizer {
  private:
    std::vector<uint32_t> m_trained_tokens;

  public:
    Tokenizer() = default;

    Tokenizer(const std::vector<uint32_t>& trained_tokens);
};
} // namespace tensorlib::tokenizer
