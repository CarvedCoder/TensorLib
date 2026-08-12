
add_executable(test
    test.cpp
)

add_executable(SingleAttentionHead
    SingleAttentionHead.cpp
)

add_executable( MultipleRegression
    MultipleRegression.cpp
)

add_executable(csv_test
    csv_test.cpp
)

add_executable( SingleLinearRegression
    SingleLinearRegression.cpp
)

target_link_libraries(SingleLinearRegression PRIVATE tensorlib csvlib dataLoader)

target_link_libraries(MultipleRegression PRIVATE tensorlib csvlib dataLoader)

target_link_libraries(csv_test PRIVATE tensorlib csvlib dataLoader)

target_link_libraries(test PRIVATE tensorlib csvlib dataLoader)

target_link_libraries(SingleAttentionHead PRIVATE tensorlib)

