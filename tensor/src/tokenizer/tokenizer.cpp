#include <cstdint>
#include <fstream>
#include <span>
#include <string>
#include <tensorlib/tokenizer/impl/bpe.h>
#include <tensorlib/tokenizer/tokenizer.h>
#include <tensorlib/tokenizer/trainer.h>
#include <vector>

namespace tensorlib::tokenizer {
Tokenizer::Tokenizer(const std::vector<uint16_t>& trained_tokens,
                     const std::vector<MergeRule>& merge_rules,
                     const std::vector<std::vector<uint8_t>>& vocab, const uint16_t vocab_size)
    : m_trained_tokens(trained_tokens), m_merge_rules(merge_rules), m_vocab(vocab),
      m_vocab_size(vocab_size) {}

const std::vector<uint8_t>& Tokenizer::vocab(uint16_t id) const {
    return m_vocab[id];
}

std::string Tokenizer::decode(std::span<const uint16_t> ids) const {
    std::string output;
    for (uint16_t id : ids) {
        const auto& bytes = m_vocab[id];
        output.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    return output;
}

std::vector<uint16_t> Tokenizer::encode(std::string_view text) const {
    auto tokens = impl::utf8::encodeToBytes(text);

    for (const auto& rule : m_merge_rules) {
        impl::bpe::merge(tokens, rule.merge_tokens, rule.result);
    }

    return tokens;
}

namespace {

template <typename T> void writeValue(std::ofstream& file, const T& value) {
    file.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));

    if (!file) {
        throw std::runtime_error("Failed while writing tokenizer file");
    }
}

template <typename T> void readValue(std::ifstream& file, T& value) {
    file.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));

    if (!file) {
        throw std::runtime_error("Corrupt or truncated tokenizer file");
    }
}

} // namespace

void Tokenizer::save(const std::filesystem::path& path) const {

    if (const auto parent = path.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open tokenizer file for writing: " + path.string());
    }

    file.write(MAGIC.data(), static_cast<std::streamsize>(MAGIC.size()));
    writeValue(file, FORMAT_VERSION);

    const uint32_t vocab_size = static_cast<uint32_t>(m_vocab.size());
    const uint32_t merge_count = static_cast<uint32_t>(m_merge_rules.size());

    writeValue(file, vocab_size);
    writeValue(file, merge_count);

    for (const MergeRule& rule : m_merge_rules) {
        writeValue(file, rule.merge_tokens.first);
        writeValue(file, rule.merge_tokens.second);
        writeValue(file, rule.result);
    }
}

void Tokenizer::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open tokenizer file for reading");
    }

    char magic[4]{};

    file.read(magic, static_cast<std::streamsize>(sizeof(magic)));

    if (!file || std::string_view(magic, 4) != MAGIC) {
        throw std::runtime_error("Invalid tokenizer file");
    }

    uint32_t version = 0;
    uint32_t vocab_size = 0;
    uint32_t merge_count = 0;

    readValue(file, version);
    readValue(file, vocab_size);
    readValue(file, merge_count);

    if (version != FORMAT_VERSION) {
        throw std::runtime_error("Unsupported tokenizer format version");
    }

    if (vocab_size < INITIAL_VOCAB_SIZE) {
        throw std::runtime_error("Invalid tokenizer vocabulary size");
    }

    if (merge_count != vocab_size - INITIAL_VOCAB_SIZE) {
        throw std::runtime_error("Invalid merge count");
    }

    m_merge_rules.clear();
    m_merge_rules.reserve(merge_count);

    for (uint32_t i = 0; i < merge_count; ++i) {
        MergeRule rule{};

        readValue(file, rule.merge_tokens.first);
        readValue(file, rule.merge_tokens.second);
        readValue(file, rule.result);

        // Because byte-level BPE starts at token 256 so the expected result of the ith merge is 256
        // + i.
        const uint16_t expected_result = static_cast<uint16_t>(INITIAL_VOCAB_SIZE + i);

        if (rule.result != expected_result) {
            throw std::runtime_error("Invalid merge rule ordering");
        }

        m_merge_rules.push_back(rule);
    }

    m_vocab.clear();
    m_vocab.resize(vocab_size);

    for (uint16_t i = 0; i < INITIAL_VOCAB_SIZE; ++i) {
        m_vocab[i].push_back(static_cast<uint8_t>(i));
    }

    for (const MergeRule& rule : m_merge_rules) {
        const uint16_t first = rule.merge_tokens.first;
        const uint16_t second = rule.merge_tokens.second;

        if (first >= m_vocab.size() || second >= m_vocab.size() || rule.result >= m_vocab.size()) {
            throw std::runtime_error("Merge rule contains an invalid token ID");
        }

        const auto& first_bytes = m_vocab[first];
        const auto& second_bytes = m_vocab[second];

        auto& result_bytes = m_vocab[rule.result];

        result_bytes.reserve(first_bytes.size() + second_bytes.size());

        result_bytes.insert(result_bytes.end(), first_bytes.begin(), first_bytes.end());
        result_bytes.insert(result_bytes.end(), second_bytes.begin(), second_bytes.end());
    }
}

} // namespace tensorlib::tokenizer
