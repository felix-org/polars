add_subdirectory(dependencies/googletest)

set(TEST_CPP_SOURCE_DIR "tests/test_cpp/polars")

add_executable(
        cpp_test
        ${TEST_CPP_SOURCE_DIR}/Expect.h
        ${TEST_CPP_SOURCE_DIR}/test_numc.cpp
        ${TEST_CPP_SOURCE_DIR}/TestSeries.cpp
        ${TEST_CPP_SOURCE_DIR}/TestSeriesMask.cpp
)

target_link_libraries(cpp_test polars_cpp gtest_main)
