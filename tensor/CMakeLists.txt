add_library(tensor_lib STATIC
        src/ops/ops.cpp
        src/tensor/tensor.cpp
        src/tensor/tensor_RNG.cpp
        src/data/loader.cpp
)

target_include_directories(tensor_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(tensor_lib PUBLIC cxx_std_20)

target_link_libraries(tensor_lib PUBLIC csvlib)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tensor_lib PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow
        -Wconversion -Wsign-conversion -Wreorder
        -Werror
    )
elseif(MSVC)
    target_compile_options(tensor_lib PRIVATE /W4 /WX)
endif()
