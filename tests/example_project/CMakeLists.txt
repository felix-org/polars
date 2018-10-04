cmake_minimum_required(VERSION 3.7.1)

project(Artic C CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(Polars)

add_executable(artic_app main.cpp)
target_link_libraries(artic_app polars_cpp)