//
// Created by Calvin Giles on 23/03/2018.
//
#include "gtest/gtest.h"

#include "../../../src/cpp/zimmer/TimeSeriesMask.h"
#include "../../../src/cpp/zimmer/TimeSeries.h"


namespace TimeSeriesMaskTests {
    TEST(TimeSeriesMask, equal) {
        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                zimmer::TimeSeriesMask(),
                zimmer::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeriesMask' match";

        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                zimmer::TimeSeriesMask({1, 2}, {0, 1}),
                zimmer::TimeSeriesMask({1, 2}, {0, 1})
        ) << "Expect " << "simple TimeSeriesMask match";
    }

    TEST(TimeSeriesMask, to_time_series) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries(),
                zimmer::TimeSeriesMask().to_time_series()
        ) << "Expect " << "empty TimeSeriesMask' match";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2}, {0, 1}),
                zimmer::TimeSeriesMask({1, 2}, {0, 1}).to_time_series()
        ) << "Expect " << "simple TimeSeriesMask match";
    }

    TEST(TimeSeriesMask, operator__or) {
        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                zimmer::TimeSeriesMask(),
                zimmer::TimeSeriesMask() | zimmer::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeriesMask' match";

        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {0, 1, 1, 1}),
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {0, 0, 1, 1}) |
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {0, 1, 0, 1})
        ) << "Expect " << "logical OR";

    }

    TEST(TimeSeriesMask, operator__and) {
        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                zimmer::TimeSeriesMask(),
                zimmer::TimeSeriesMask() & zimmer::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeriesMask' match";

        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {0, 0, 0, 1}),
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {0, 0, 1, 1}) &
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {0, 1, 0, 1})
        ) << "Expect " << "logical OR";

    }

    TEST(TimeSeriesMask, operator__not) {
        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                zimmer::TimeSeriesMask(),
                !zimmer::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeriesMask' match";

        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                zimmer::TimeSeriesMask({1, 2}, {0, 1}),
                !zimmer::TimeSeriesMask({1, 2}, {1, 0})
        ) << "Expect " << "logical NOT";

    }
}  // TimeSeriesMaskTests
