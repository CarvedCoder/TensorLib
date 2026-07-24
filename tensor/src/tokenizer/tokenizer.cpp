#include <cstdint>
#include <tensorlib/tokenizer/tokenizer.h>
#include <vector>

namespace tensorlib::tokenizer {
Tokenizer::Tokenizer(const std::vector<uint32_t>& trained_tokens)
    : m_trained_tokens(trained_tokens) {}
} // namespace tensorlib::tokenizer
