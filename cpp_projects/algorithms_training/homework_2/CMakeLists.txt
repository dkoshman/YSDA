cmake_minimum_required(VERSION 3.20)
project(Homework_2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_ASAN "-g -fsanitize=address,undefined,leak -fno-sanitize-recover=all")

add_executable(Homework_2 main.cpp)
add_executable(A A.cpp)
add_executable(B B.cpp)
add_executable(C C.cpp)
add_executable(D D.cpp)
add_executable(E E.cpp)
