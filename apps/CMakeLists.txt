add_executable(linearRegression
    linearRegression.cpp
)

add_executable(test
    test.cpp
)


target_link_libraries(linearRegression PRIVATE tensorlib csvlib dataLoader)

target_link_libraries(test PRIVATE tensorlib csvlib dataLoader)

