//
// Created by Calvin Giles on 23/03/2018.
//
#include "gtest/gtest.h"

#include "polars/TimeSeriesMask.h"
#include "polars/TimeSeries.h"


namespace TimeSeriesMaskTests {
    TEST(TimeSeriesMask, equal) {
        EXPECT_PRED2(
                polars::TimeSeriesMask::equal,
                polars::TimeSeriesMask(),
                polars::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeriesMask' match";

        EXPECT_PRED2(
                polars::TimeSeriesMask::equal,
                polars::TimeSeriesMask({1, 2}, {0, 1}),
                polars::TimeSeriesMask({1, 2}, {0, 1})
        ) << "Expect " << "simple TimeSeriesMask match";
    }

    TEST(TimeSeriesMask, to_time_series) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries(),
                polars::TimeSeriesMask().to_time_series()
        ) << "Expect " << "empty TimeSeriesMask' match";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2}, {0, 1}),
                polars::TimeSeriesMask({1, 2}, {0, 1}).to_time_series()
        ) << "Expect " << "simple TimeSeriesMask match";
    }

    TEST(TimeSeriesMask, operator__or) {
        EXPECT_PRED2(
                polars::TimeSeriesMask::equal,
                polars::TimeSeriesMask(),
                polars::TimeSeriesMask() | polars::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeriesMask' match";

        EXPECT_PRED2(
                polars::TimeSeriesMask::equal,
                polars::TimeSeriesMask({1, 2, 3, 4}, {0, 1, 1, 1}),
                polars::TimeSeriesMask({1, 2, 3, 4}, {0, 0, 1, 1}) |
                polars::TimeSeriesMask({1, 2, 3, 4}, {0, 1, 0, 1})
        ) << "Expect " << "logical OR";

    }

    TEST(TimeSeriesMask, operator__and) {
        EXPECT_PRED2(
                polars::TimeSeriesMask::equal,
                polars::TimeSeriesMask(),
                polars::TimeSeriesMask() & polars::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeriesMask' match";

        EXPECT_PRED2(
                polars::TimeSeriesMask::equal,
                polars::TimeSeriesMask({1, 2, 3, 4}, {0, 0, 0, 1}),
                polars::TimeSeriesMask({1, 2, 3, 4}, {0, 0, 1, 1}) &
                polars::TimeSeriesMask({1, 2, 3, 4}, {0, 1, 0, 1})
        ) << "Expect " << "logical OR";

    }

    TEST(TimeSeriesMask, operator__not) {
        EXPECT_PRED2(
                polars::TimeSeriesMask::equal,
                polars::TimeSeriesMask(),
                !polars::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeriesMask' match";

        EXPECT_PRED2(
                polars::TimeSeriesMask::equal,
                polars::TimeSeriesMask({1, 2}, {0, 1}),
                !polars::TimeSeriesMask({1, 2}, {1, 0})
        ) << "Expect " << "logical NOT";

    }
}  // TimeSeriesMaskTests
