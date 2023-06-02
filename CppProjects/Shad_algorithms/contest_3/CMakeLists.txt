cmake_minimum_required(VERSION 3.12)
project(contest_3)

set(CMAKE_CXX_STANDARD 17)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror -Wsign-compare")

set(CMAKE_CXX_FLAGS_RELEASE " -O2")

set(CMAKE_CXX_FLAGS_ASAN "-g -fsanitize=address,undefined,leak -fno-sanitize-recover=all"
        CACHE STRING "Compiler flags in asan build"
        FORCE)

add_executable(fixed_set_test fixed_set_test.cpp fixed_set.h)
add_executable(B B.cpp)
add_executable(C C.cpp)
add_executable(D D.cpp)
add_executable(E E.cpp)
