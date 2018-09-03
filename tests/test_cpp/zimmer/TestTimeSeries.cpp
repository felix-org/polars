//
// Created by Calvin Giles on 22/03/2018.
//
#include "gtest/gtest.h"

#include "zimmer/TimeSeries.h"
#include "zimmer/TimeSeriesMask.h"
#include "zimmer/look_ahead.h"
#include "zimmer/numc.h"


namespace TimeSeriesTests {
    TEST(TimeSeries, equal) {
        EXPECT_PRED2(TimeSeries::equal, TimeSeries(), TimeSeries()) << "Expect " << "empty TimeSeries' match";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})),
                     TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})));

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1}), arma::vec({NAN})),
                     TimeSeries(arma::vec({1}), arma::vec({NAN})))
                            << "Expect " << "simple timeseries with NAN match";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({}), arma::vec({})),
                     TimeSeries(arma::vec({}), arma::vec({})))
                            << "Expect " << "empty timeseries match";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2, 3, 4, 5, 6}), arma::vec({4, NAN, 5, NAN, NAN, 6})),
                     TimeSeries(arma::vec({1, 2, 3, 4, 5, 6}), arma::vec({4, NAN, 5, NAN, NAN, 6})))
                            << "Expect " << "longer timeseries with NANs match";
    }

    TEST(TimeSeries, not_equal) {
        EXPECT_PRED2(TimeSeries::not_equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})),
                     TimeSeries(arma::vec({1, 2}), arma::vec({1, 2})))
                            << "Expect " << "timestamps match does not imply TimeSeries match";

        EXPECT_PRED2(TimeSeries::not_equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})),
                     TimeSeries(arma::vec({3, 4}), arma::vec({3, 4})))
                            << "Expect " << "values match does not imply TimeSeries match";

        EXPECT_PRED2(TimeSeries::not_equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})),
                     TimeSeries(arma::vec({3, 4}), arma::vec({1, 2})))
                            << "Expect " << "swapping timestamps and values results in no match" << "";
    }

    TEST(TimeSeries, where) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2}, {3, 4}).where(zimmer::TimeSeriesMask({1, 2}, {0, 1}), 17),
                TimeSeries({1, 2}, {17, 4})
        ) << "Expect " << "simple where()  to select correctly";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2}, {3, 4}).where(zimmer::TimeSeriesMask({1, 2}, {0, 1}), NAN),
                TimeSeries({1, 2}, {NAN, 4})
        ) << "Expect " << ".where(..., NAN) to not set everything to NAN";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2}, {3, 4}).where(zimmer::TimeSeriesMask({1, 2}, {0, 1}),
                                                 zimmer::LookAheadInterface::DetailedState::unknown),
                TimeSeries({1, 2}, {zimmer::LookAheadInterface::DetailedState::unknown, 4})
        ) << "Expect " << ".where(..., enum_value) and correctly pass the enum through";
    }

    TEST(TimeSeries, DiffTest) {
        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})).diff(),
                     TimeSeries(arma::vec({1, 2}), arma::vec({NAN, 1})))
                            << "Expect " << "simple diff() fixture result to be correct" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2}), arma::vec({4, 3})).diff(),
                     TimeSeries(arma::vec({1, 2}), arma::vec({NAN, -1})))
                            << "Expect " << "simple diff() fixture result to be correct" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1}), arma::vec({3})).diff(),
                     TimeSeries(arma::vec({1}), arma::vec({NAN})))
                            << "Expect " << "simple diff() fixture result to be correct" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({}), arma::vec({})).diff(),
                     TimeSeries(arma::vec({}), arma::vec({})))
                            << "Expect " << "exmpty timeseries diff() fixture result to be correct" << "";
    }

    TEST(TimeSeries, abs) {
        EXPECT_PRED2(TimeSeries::equal,
                     TimeSeries(arma::vec({1, 2, 3, 4, 5, 6}), arma::vec({0, 3, 4, -2, 1.5, NAN})).abs(),
                     TimeSeries(arma::vec({1, 2, 3, 4, 5, 6}), arma::vec({0, 3, 4, 2, 1.5, NAN})))
                            << "Expect " << "negative values to be positive and rest to remain the same.";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({}), arma::vec({})).pow(2),
                     TimeSeries(arma::vec({}), arma::vec({})))
                            << "Expect " << "empty TimeSeries .abs() to return empty TimeSeries";
    }

    TEST(TimeSeries, PowTest) {
        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})).pow(2),
                     TimeSeries(arma::vec({1, 2}), arma::vec({9, 16})))
                            << "Expect " << "simple pow() fixture result to be correct" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})).pow(3),
                     TimeSeries(arma::vec({1, 2}), arma::vec({27, 64})))
                            << "Expect " << "simple pow() fixture result to be correct" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1}), arma::vec({9})).pow(0.5),
                     TimeSeries(arma::vec({1}), arma::vec({3})))
                            << "Expect " << "simple pow() fixture result to be correct" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({}), arma::vec({})).pow(2),
                     TimeSeries(arma::vec({}), arma::vec({})))
                            << "Expect " << "empty timeseries pow() fixture result to be correct" << "";
    }

    TEST(TimeSeries, clipTest) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries(arma::vec({}), arma::vec({})).clip(0,1),
                TimeSeries(arma::vec({}), arma::vec({}))
        )  << "Expect " << "empty TimeSeries";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries(arma::vec({1,2,3,4}), arma::vec({1,2,3,4})).clip(2,3),
                TimeSeries(arma::vec({1,2,3,4}), arma::vec({2,2,3,3}))
        ) << "Expect " << "timeseries clipped to 2-3";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries(arma::vec({1,2,3,4}), arma::vec({1,2,3,4})).clip(0,1),
                TimeSeries(arma::vec({1,2,3,4}), arma::vec({1,1,1,1}))
        ) << "Expect " << "timeseries clipped to 2-3";
    }

    TEST(TimeSeries, MeanTest) {
        EXPECT_EQ(TimeSeries(arma::vec({1, 2}), arma::vec({3, 4})).mean(), 3.5)
                            << "Expect " << "simple mean() fixture result to be correct" << "";

        ASSERT_TRUE(isnan(TimeSeries(arma::vec({}), arma::vec({})).mean()))
                                    << "Expect " << "empty series mean() should be NAN" << "";

        EXPECT_EQ(TimeSeries(arma::vec({1, 2, 3}), arma::vec({3, NAN, 4})).mean(), 3.5)
                            << "Expect " << "simple mean() fixture result with NAN to be correct, ignoring NANs" << "";


    }

    TEST(TimeSeries, RollingQuantileTest) {
        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({}), arma::vec({})),
                     TimeSeries(arma::vec({}), arma::vec({})).rolling(3, zimmer::Quantile(0.5)))
                            << "Expect " << "rolling() test 1 for empty series" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2}), arma::vec({NAN, NAN})),
                     TimeSeries(arma::vec({1, 2}), arma::vec({1, 1})).rolling(3, zimmer::Quantile(0.5)))
                            << "Expect " << "series of size 2, rolling window of 3, should be NANS" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2, 3}), arma::vec({NAN, 1, NAN})),
                     TimeSeries(arma::vec({1, 2, 3}), arma::vec({1, 1, 1})).rolling(3, zimmer::Quantile(0.5)))
                            << "Expect " << "series of size 3, rolling window of 3" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2, 3}), arma::vec({NAN, 5, NAN})),
                     TimeSeries(arma::vec({1, 2, 3}), arma::vec({10, 1, 5})).rolling(3, zimmer::Quantile(0.5)))
                            << "Expect " << "series of size 3, rolling window of 3" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2, 3, 4}), arma::vec({NAN, NAN, 5.5, NAN})),
                     TimeSeries(arma::vec({1, 2, 3, 4}), arma::vec({10, 1, 5, 6})).rolling(4, zimmer::Quantile(0.5)))
                            << "Expect " << "series of size 4, rolling window of 4" << "";

        EXPECT_PRED2(TimeSeries::almost_equal, TimeSeries(arma::vec({1, 2, 3, 4, 5}), arma::vec({NAN, NAN, 3.4, NAN, NAN})),
                     TimeSeries(arma::vec({1, 2, 3, 4, 5}), arma::vec({10, 1, 5, 4, 7})).rolling(5, zimmer::Quantile(0.2)))
                            << "Expect " << "series of size 5, rolling window of 5, quantile 0.2" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2, 3, 4, 5}), arma::vec({NAN, 5.5, NAN, NAN, 5.5})),
                     TimeSeries(arma::vec({1, 2, 3, 4, 5}), arma::vec({10, 1, NAN, 4, 7})).rolling(2, zimmer::Quantile(0.5)))
                            << "Expect " << "NAN test case passes" << "";


        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2, 3}), arma::vec({5.5, 5, 3})),
                     TimeSeries(arma::vec({1, 2, 3}), arma::vec({10, 1, 5})).rolling(3, zimmer::Quantile(0.5), 2, true))
                            << "Expect " << "rolling() test 1 for minperiods" << "";

        EXPECT_PRED2(TimeSeries::equal, TimeSeries(arma::vec({1, 2, 3}), arma::vec({NAN, 7.5, NAN})),
                     TimeSeries(arma::vec({1, 2, 3}), arma::vec({10, NAN, 5})).rolling(3, zimmer::Quantile(0.5), 2, true))
                            << "Expect " << "rolling() test 2 for minperiods, with nans" << "";

    }


    TEST(TimeSeries, rolling_sum) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries().rolling(5, zimmer::Sum()),
                TimeSeries()
        ) << "Expect " << "empty TimeSeries returns empty TimeSeries";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN}).rolling(1, zimmer::Sum()),
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN})
        ) << "Expect " << "with a window of 1 the timeseries is returned as is";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN}).rolling(3, zimmer::Sum()),
                TimeSeries({1, 2, 3, 4, 5}, {NAN,  6.5, 4.5, NAN, NAN})
        ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should be the sum, not NAN";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN}).rolling(3, zimmer::Sum(), 2),
                TimeSeries({1, 2, 3, 4, 5}, {3,  6.5, 4.5, 2.5, NAN})
        ) << "Expect " << "with window=3, min_periods=2 the edge values should be NAN and the rest should sum the windows";
    }


    TEST(TimeSeries, rolling_count) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries().rolling(5, zimmer::Count()),
                TimeSeries()
        ) << "Expect " << "empty TimeSeries returns empty TimeSeries";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1, 2, 3.5, -1, NAN}).rolling(1, zimmer::Count()),
                TimeSeries({1, 2, 3, 4, 5}, {1, 1, 1, 1, NAN})
        ) << "Expect " << "with a window of 1 the timeseries is returned as is";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1, 2, 3.5, -1, NAN}).rolling(3, zimmer::Count()),
                TimeSeries({1, 2, 3, 4, 5}, {NAN, 3, 3, NAN, NAN})
        ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should be the count, not NAN";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1, 2, 3.5, -1, NAN}).rolling(3, zimmer::Count(), 2),
                TimeSeries({1, 2, 3, 4, 5}, {2, 3, 3, 2, NAN})
        ) << "Expect " << "with window=3, min_periods=2 then any windows with at least 2 values should be non-NAN";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1, 2, 3.5, -1, NAN}).rolling(3, zimmer::Count(), 1),
                TimeSeries({1, 2, 3, 4, 5}, {2, 3, 3, 2, 1})
        ) << "Expect " << "with window=3, min_periods=1 the edge values should be NAN and the rest should count the windows";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1, 2, 3.5, NAN, NAN}).rolling(3, zimmer::Count(0), 1),
                TimeSeries({1, 2, 3, 4, 5}, {2, 3, 2, 1, 0})
        ) << "Expect " << "with window=3, min_periods=1 and a default of 0, all windows should have a count";

        // Symmetric = True
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}).rolling(3, zimmer::Count(), 1, true, true),
                TimeSeries({1, 2, 3, 4, 5}, {1, 3, 3, 3, 1})
        ) << "Expect " << "with window of 3 the timeseries expects smaller windows on the edges so counts are 1";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(7, zimmer::Count(), 1, true, true),
                TimeSeries({1, 2, 3, 4, 5, 6}, {1, 3, 5, 5, 3, 1})
        ) << "Expect " << "with window of 7 the timeseries expects smaller and smaller counts along the edges";
    }

    TEST(TimeSeries, operator__add) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries() + 1,
                TimeSeries()
        ) << "Expect " << "empty TimeSeries stays empty";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2}, {3, 4}) + 1,
                TimeSeries({1, 2}, {4, 5})
        ) << "Expect " << "adding 1 increases the values, not the timestamps";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3}, {3, 4, arma::datum::nan}) + 1,
                TimeSeries({1, 2, 3}, {4, 5, arma::datum::nan})
        ) << "Expect " << "adding 1 increases the values, not the timestamps, and ignores nan";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4}, {3, 4, 2, arma::datum::nan}) +
                TimeSeries({1, 2, 3, 4}, {3, -5, arma::datum::nan, arma::datum::nan}),
                TimeSeries({1, 2, 3, 4}, {6, -1, arma::datum::nan, arma::datum::nan})
        ) << "Expect " << "adding positive and negative values works and nan results in nan";
    }


    TEST(TimeSeries, operator__subtract) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries() - 1,
                TimeSeries()
        ) << "Expect " << "empty TimeSeries stays empty";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2}, {3, 4}) - 1,
                TimeSeries({1, 2}, {2, 3})
        ) << "Expect " << "adding 1 increases the values, not the timestamps";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3}, {3, 4, arma::datum::nan}) - 1,
                TimeSeries({1, 2, 3}, {2, 3, arma::datum::nan})
        ) << "Expect " << "adding 1 increases the values, not the timestamps, and ignores nan";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4}, {-3, 4, 2, arma::datum::nan}) -
                TimeSeries({1, 2, 3, 4}, {3, -5, arma::datum::nan, arma::datum::nan}),
                TimeSeries({1, 2, 3, 4}, {-6, 9, arma::datum::nan, arma::datum::nan})
        ) << "Expect " << "subtracting positive and negative values works and nan results in nan";
    }


    TEST(TimeSeries, operator__multiply) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries() * 2,
                TimeSeries()
        ) << "Expect " << "empty TimeSeries stays empty";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2}, {3, 4}) * 2,
                TimeSeries({1, 2}, {6, 8})
        ) << "Expect " << "multiplying by 2 changes the values, not the timestamps";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3}, {3, 4, arma::datum::nan}) * 2,
                TimeSeries({1, 2, 3}, {6, 8, arma::datum::nan})
        ) << "Expect " << "multiplying by 2 changes the values, not the timestamps, and ignores nan";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4}, {-3, 4, 2, arma::datum::nan}) *
                TimeSeries({1, 2, 3, 4}, {3, -5, arma::datum::nan, arma::datum::nan}),
                TimeSeries({1, 2, 3, 4}, {-9, -20, arma::datum::nan, arma::datum::nan})
        ) << "Expect " << "multiplying positive and negative values works and nan results in nan";
    }


    TEST(TimeSeries, operator__eq) {
        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries() == TimeSeries(),
                zimmer::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeries stays empty";

        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries({1, 2, 3, 4}, {0, 1, 2, NAN}) ==
                TimeSeries({1, 2, 3, 4}, {0, 1, 3, NAN}),
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {1, 1, 0, 0})
        ) << "Expect " << "matching should match, but NAN != NAN for the elementwise operator";

        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries() == 1,
                zimmer::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeries stays empty";

        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries({1, 2, 3, 4, 5}, {0, 1.99, 2, 1 + 1, NAN}) == 2,
                zimmer::TimeSeriesMask({1, 2, 3, 4, 5}, {0, 0, 1, 1, 0})
        ) << "Expect " << "matching should match, but NAN != NAN for the elementwise operator";
    }


    TEST(TimeSeries, operator__ne) {
        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries() != 1,
                zimmer::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeries stays empty";

        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries({1, 2, 3, 4, 5}, {0, 1.99, 2, 1 + 1, NAN}) != 2,
                zimmer::TimeSeriesMask({1, 2, 3, 4, 5}, {1, 1, 0, 0, 0})
        ) << "Expect " << "matching should match, but NAN != NAN for the elementwise operator";
    }

    TEST(TimeSeries, operator__gt) {
        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries() > TimeSeries(),
                zimmer::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeries stays empty";


        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries({1, 2, 3, 4}, {0, -1, 3, NAN}) >
                TimeSeries({1, 2, 3, 4}, {0, -2, 2, NAN}),
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {0, 1, 1, 0})
        ) << "Expect " << "> should work as per pair, including NAN != NAN";

        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries({1, 2, 3, 4}, {0, -1, 3, NAN}) >= 0,
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {1, 0, 1, 0})
        ) << "Expect " << ">= should work per item, including NAN != NAN";
    }

    TEST(TimeSeries, operator__lt) {
        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries() < TimeSeries(),
                zimmer::TimeSeriesMask()
        ) << "Expect " << "empty TimeSeries stays empty";


        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries({1, 2, 3, 4}, {0, -2, 2, NAN}) <
                TimeSeries({1, 2, 3, 4}, {0, -1, 3, NAN}),
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {0, 1, 1, 0})
        ) << "Expect " << "> should work as per pair, including NAN != NAN";

        EXPECT_PRED2(
                zimmer::TimeSeriesMask::equal,
                TimeSeries({1, 2, 3, 4}, {0, -1, 3, NAN}) <= 0,
                zimmer::TimeSeriesMask({1, 2, 3, 4}, {1, 1, 0, 0})
        ) << "Expect " << "<= should work per item, including NAN != NAN";
    }


    TEST(TimeSeries, apply){
        EXPECT_PRED2(
             TimeSeries::equal,
             TimeSeries({1, 2, 3}, {1., 2., 3.}),
             TimeSeries({1, 2, 3}, {1., 2., 3.}).apply(abs)
        ) << "Expect " << " should remain ideantical";

        EXPECT_PRED2(
             TimeSeries::equal,
             TimeSeries({1, 2, 3}, {1., 2., 3.}),
             TimeSeries({1, 2, 3}, {-1., -2., -3.}).apply(abs)
        ) << "Expect " << " should flip signs";

        EXPECT_PRED2(
             TimeSeries::equal,
             TimeSeries({1, 2, 3}, {2.7182818284590451, 7.3890560989306504, 20.085536923187668}),
             TimeSeries({1, 2, 3}, {1., 2., 3.}).apply(exp)
        ) << "Expect " << " should apply exponential";

        EXPECT_PRED2(
             TimeSeries::equal,
             TimeSeries({1, 2, 3, 4}, {0, 0.19, 1.9, 1.9}).apply(exp),
             TimeSeries({1, 2, 3, 4}, {1., 1.2092495976572515, 6.6858944422792685, 6.6858944422792685})
        ) << "Expect " << " should apply exponential";

    }

    TEST(TimeSeries, rolling_sum_triangle) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries().rolling(5, zimmer::Sum(),  0, true, false, zimmer::WindowProcessor::WindowType::triang),
                TimeSeries()
        ) << "Expect " << "empty TimeSeries returns empty TimeSeries";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN}).rolling(1, zimmer::Sum(), 0, true, false, zimmer::WindowProcessor::WindowType::triang),
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN})
        ) << "Expect " << "with a window of 1 the timeseries is returned as is";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3}, {1, 2, 3}).rolling(2, zimmer::Sum(), 0, true, false, zimmer::WindowProcessor::WindowType::triang),
                TimeSeries({1, 2, 3}, {NAN, 1.5, 2.5})
        ) << "Expect " << "with a window of 2";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN}).rolling(3, zimmer::Sum(), 0, true, false, zimmer::WindowProcessor::WindowType::triang),
                TimeSeries({1, 2, 3, 4, 5}, {NAN,  4.25, 4.0, NAN, NAN})
        ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should be the sum, not NAN";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN}).rolling(3, zimmer::Sum(), 2, true, false, zimmer::WindowProcessor::WindowType::triang),
                TimeSeries({1, 2, 3, 4, 5}, {2, 4.25, 4.0, 0.75, NAN})
        ) << "Expect " << "with window=3, min_periods=2 the edge values should be NAN and the rest should sum the windows";
    }

    TEST(TimeSeries, rolling_mean_triangle) {
        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries().rolling(5, zimmer::Mean(),  0, true, false, zimmer::WindowProcessor::WindowType::triang),
                TimeSeries()
        ) << "Expect " << "empty TimeSeries returns empty TimeSeries";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN}).rolling(1, zimmer::Mean(), 0, true, false, zimmer::WindowProcessor::WindowType::triang),
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN})
        ) << "Expect " << "with a window of 1 the timeseries is returned as is";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3}, {1, 2, 3}).rolling(3, zimmer::Mean(), 0, true, false, zimmer::WindowProcessor::WindowType::triang),
                TimeSeries({1, 2, 3}, {NAN, 2, NAN})
        ) << "Expect " << "with a window of 3 the timeseries of length 3 returns just central value";

        EXPECT_PRED2(
                TimeSeries::equal,
                TimeSeries({1, 2, 3, 4, 5}, {1,  2, 3.5, -1, NAN}).rolling(3, zimmer::Mean(), 0, true, false, zimmer::WindowProcessor::WindowType::triang),
                TimeSeries({1, 2, 3, 4, 5}, {NAN,  2.125, 2.0, NAN, NAN})
        ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should give weighted mean, not NAN";

        EXPECT_PRED2(
                TimeSeries::almost_equal,
                TimeSeries({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(5, zimmer::Mean(), 1, true, false, zimmer::WindowProcessor::WindowType::triang),
                TimeSeries({1, 2, 3, 4}, {1.66666667, 2.25, 2.75, 3.33333333})
        ) << "Expect " << "no NANs because min periods is 1.";
    }
} // namespace TimeSeriesTests
