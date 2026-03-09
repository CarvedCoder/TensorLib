add_library(dataLoader STATIC
    src/loader.cpp
)

target_link_libraries(dataLoader PUBLIC tensorlib csvlib)

target_include_directories(dataLoader
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(dataLoader PUBLIC cxx_std_20)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
target_compile_options(csvlib PRIVATE -Wall -Wextra)
endif()

