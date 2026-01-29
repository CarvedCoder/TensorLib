add_library(csvlib STATIC
    src/csv.cpp
)

target_include_directories(csvlib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(csvlib PUBLIC cxx_std_20)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
target_compile_options(csvlib PRIVATE -Wall -Wextra)
endif()
