#ifndef EMBEDDING_H
#define EMBEDDING_H
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
namespace Embedding {

struct TokenStorage {
    size_t size{};
    std::span<std::string> tokens{};
};

TokenStorage tokenize(const std::string_view sentence);

} // namespace Embedding

#endif // !EMBEDDING_H
