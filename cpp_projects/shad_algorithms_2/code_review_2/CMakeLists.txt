cmake_minimum_required(VERSION 3.12)
project(Code_review_2)

set(CMAKE_CXX_STANDARD 17)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror -Wsign-compare")

set(CMAKE_CXX_FLAGS_RELEASE " -O2")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -fsanitize=address,undefined,leak -fno-sanitize-recover=all"
        CACHE STRING "Compiler flags in asan build"
        FORCE)

add_executable(main main.cpp)
add_executable(main_short main_short.cpp)
