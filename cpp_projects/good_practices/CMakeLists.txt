cmake_minimum_required(VERSION 3.20)
project(GoodPractices)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_OSX_ARCHITECTURES "arm64")

add_executable(GoodPractices main.cpp)
add_executable(profiler profiler.cpp)
add_executable(test test.cpp)
add_executable(sort sort.cpp)
