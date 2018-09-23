cmake_minimum_required(VERSION 3.7.1)

project(Polars C CXX)

set(CMAKE_CXX_STANDARD 14)

# Dependencies
set(DEPENDENCIES_DIR "dependencies")

# Include other CMakeLists.txt
include(src/cpp/polars/CMakeLists.txt)
include(tests/test_cpp/polars/CMakeLists.txt)
