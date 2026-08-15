#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <sys/types.h>
#include <vector>
namespace tensorlib::tokenizer::impl::utf8 {
std::vector<uint16_t> encodeToBytes(std::string_view str);
std::string readFile(const std::filesystem::path& data_path);

} // namespace tensorlib::tokenizer::impl::utf8
