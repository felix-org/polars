//
// Created by Calvin Giles on 22/03/2018.
//
#include "gtest/gtest.h"

#include "../../../src/cpp/zimmer/TimeSeries.h"


namespace TimeSeriesTests {
    TEST(TimeSeries, EQTest) {
        EXPECT_PRED2(equal, TimeSeries(), TimeSeries()) << "Expect " << "empty TimeSeries' match";

        EXPECT_PRED2(equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})),
                     TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})))
                            << "Expect " << "simple timeseries match";

        EXPECT_PRED2(not_equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})),
                     TimeSeries(arma::vec({1, 2}), arma::vec({1, 2})))
                            << "Expect " << "timestamps match does not imply TimeSeries match";

        EXPECT_PRED2(not_equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})),
                     TimeSeries(arma::vec({3, 4}), arma::vec({3, 4})))
                            << "Expect " << "values match does not imply TimeSeries match";

        EXPECT_PRED2(not_equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})),
                     TimeSeries(arma::vec({3, 4}), arma::vec({1, 2})))
                            << "Expect swapping timestamps and values results in no match" << "";
    }
} // namespace TimeSeriesTests
