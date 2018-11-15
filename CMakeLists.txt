cmake_minimum_required(VERSION 3.7.1)

project(Polars C CXX)

set(CMAKE_CXX_STANDARD 14)

option(WITH_TESTS "Build polars_cpp_test target" ON)
option(BUILD_WITH_CONAN "Resolve dependencies using conan" OFF)

if(NOT BUILD_WITH_CONAN)
  # Dependencies
  include(dependencies/CMakeLists.txt)
endif()

# Include other CMakeLists.txt
include(src/cpp/polars/CMakeLists.txt)

if(WITH_TESTS)
  include(tests/test_cpp/polars/CMakeLists.txt)
endif()
