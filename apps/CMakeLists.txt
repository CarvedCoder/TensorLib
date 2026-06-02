add_executable(main
    main.cpp
)

add_executable(test
    test.cpp
)


target_link_libraries(main PRIVATE tensorlib csvlib dataLoader)

target_link_libraries(test PRIVATE tensorlib csvlib dataLoader)

