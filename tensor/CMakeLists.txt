add_library(tensorlib STATIC
        src/ops/ops.cpp
        src/ops/embedding.cpp
        src/tensor/tensor.cpp
        src/tensor/tensor_RNG.cpp
        src/tensor/tensor_impl.cpp
        src/autograd/autograd.cpp
)

target_include_directories(tensorlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(tensorlib PUBLIC cxx_std_23)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tensorlib PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow
        -Wconversion -Wsign-conversion -Wreorder
        -Werror
    )
elseif(MSVC)
    target_compile_options(tensorlib PRIVATE /W4 /WX)
endif()
