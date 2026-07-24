#include <cstdint>
#include <string_view>
#include <tensorlib/tokenizer/impl/utf8.h>
#include <vector>

namespace tensorlib::tokenizer::impl::utf8 {
std::vector<uint32_t> encodeToBytes(std::string_view str) {
    std::vector<uint32_t> bytes(str.begin(), str.end());
    return bytes;
}

std::string readFile(const std::filesystem::path& data_path) {
    std::ifstream file(data_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open the file");
    }
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::string contents(static_cast<size_t>(size), '\0');
    file.read(contents.data(), size);
    return contents;
}
} // namespace tensorlib::tokenizer::impl::utf8
