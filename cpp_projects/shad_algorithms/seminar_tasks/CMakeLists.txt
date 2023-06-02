cmake_minimum_required(VERSION 3.12)
project(Seminar_tasks)

set(CMAKE_CXX_STANDARD 17)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror -Wsign-compare")

set(CMAKE_CXX_FLAGS_RELEASE " -O2")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -fsanitize=address,undefined,leak -fno-sanitize-recover=all"
        CACHE STRING "Compiler flags in asan build"
        FORCE)

add_executable(cc contest.cpp graph.h)
add_executable(A A.cpp)
add_executable(B B.cpp)
add_executable(C C.cpp)
add_executable(D D.cpp)
add_executable(E E.cpp)
add_executable(F F.cpp)
add_executable(G G.cpp)
add_executable(H H.cpp)
add_executable(I I.cpp)
add_executable(J J.cpp)
