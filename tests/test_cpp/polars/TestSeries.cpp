//
// Created by Calvin Giles on 22/03/2018.
//
#include "gtest/gtest.h"

#include "polars/Series.h"
#include "polars/SeriesMask.h"
#include "polars/look_ahead.h"
#include "polars/numc.h"


namespace SeriesTests {
    TEST(Series, equal) {
        EXPECT_PRED2(Series::equal, Series(), Series()) << "Expect " << "empty Series' match";

        EXPECT_PRED2(Series::equal, Series(arma::vec({3, 4}), arma::vec({1, 2})),
                     Series(arma::vec({3, 4}), arma::vec({1, 2})));

        EXPECT_PRED2(Series::equal, Series(arma::vec({NAN}), arma::vec({1})),
                     Series(arma::vec({NAN}), arma::vec({1})))
                            << "Expect " << "simple indices with NAN match";

        EXPECT_PRED2(Series::equal, Series(arma::vec({}), arma::vec({})),
                     Series(arma::vec({}), arma::vec({})))
                            << "Expect " << "empty indices match";

        EXPECT_PRED2(Series::equal, Series(arma::vec({4, NAN, 5, NAN, NAN, 6}), arma::vec({1, 2, 3, 4, 5, 6})),
                     Series(arma::vec({4, NAN, 5, NAN, NAN, 6}), arma::vec({1, 2, 3, 4, 5, 6})))
                            << "Expect " << "longer indices with NANs match";
    }

    TEST(Series, not_equal) {
        EXPECT_PRED2(Series::not_equal, Series(arma::vec({3, 4}), arma::vec({1, 2})),
                     Series(arma::vec({1, 2}), arma::vec({1, 2})))
                            << "Expect " << "index match does not imply Series match";

        EXPECT_PRED2(Series::not_equal, Series(arma::vec({3, 4}), arma::vec({1, 2})),
                     Series(arma::vec({3, 4}), arma::vec({3, 4})))
                            << "Expect " << "values match does not imply Series match";

        EXPECT_PRED2(Series::not_equal, Series(arma::vec({3, 4}), arma::vec({1, 2})),
                     Series(arma::vec({1, 2}), arma::vec({3, 4})))
                            << "Expect " << "swapping index and values results in no match" << "";
    }

    TEST(Series, where) {
        EXPECT_PRED2(
                Series::equal,
                Series({3, 4}, {1, 2}).where(polars::SeriesMask({0, 1}, {1, 2}), 17),
                Series({17, 4}, {1, 2})
        ) << "Expect " << "simple where()  to select correctly";

        EXPECT_PRED2(
                Series::equal,
                Series({3, 4}, {1, 2}).where(polars::SeriesMask({0, 1}, {1, 2}), NAN),
                Series({NAN, 4}, {1, 2})
        ) << "Expect " << ".where(..., NAN) to not set everything to NAN";

        EXPECT_PRED2(
                Series::equal,
                Series({3, 4}, {1, 2}).where(polars::SeriesMask({0, 1}, {1, 2}),
                                                 polars::LookAheadInterface::DetailedState::unknown),
                Series({polars::LookAheadInterface::DetailedState::unknown, 4}, {1, 2})
        ) << "Expect " << ".where(..., enum_value) and correctly pass the enum through";
    }

    TEST(Series, DiffTest) {
        EXPECT_PRED2(Series::equal, Series(arma::vec({3, 4}), arma::vec({1, 2})).diff(),
                     Series(arma::vec({NAN, 1}), arma::vec({1, 2})))
                            << "Expect " << "simple diff() fixture result to be correct" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({4, 3}), arma::vec({1, 2})).diff(),
                     Series(arma::vec({NAN, -1}), arma::vec({1, 2})))
                            << "Expect " << "simple diff() fixture result to be correct" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({3}), arma::vec({1})).diff(),
                     Series(arma::vec({NAN}), arma::vec({1})))
                            << "Expect " << "simple diff() fixture result to be correct" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({}), arma::vec({})).diff(),
                     Series(arma::vec({}), arma::vec({})))
                            << "Expect " << "exmpty indices diff() fixture result to be correct" << "";
    }

    TEST(Series, abs) {
        EXPECT_PRED2(Series::equal,
                     Series(arma::vec({0, 3, 4, -2, 1.5, NAN}), arma::vec({1, 2, 3, 4, 5, 6})).abs(),
                     Series(arma::vec({0, 3, 4, 2, 1.5, NAN}), arma::vec({1, 2, 3, 4, 5, 6})))
                            << "Expect " << "negative values to be positive and rest to remain the same.";

        EXPECT_PRED2(Series::equal, Series(arma::vec({}), arma::vec({})).pow(2),
                     Series(arma::vec({}), arma::vec({})))
                            << "Expect " << "empty Series .abs() to return empty Series";
    }

    TEST(Series, PowTest) {
        EXPECT_PRED2(Series::equal, Series(arma::vec({3, 4}), arma::vec({1, 2})).pow(2),
                     Series(arma::vec({9, 16}), arma::vec({1, 2})))
                            << "Expect " << "simple pow() fixture result to be correct" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({3, 4}), arma::vec({1, 2})).pow(3),
                     Series(arma::vec({27, 64}), arma::vec({1, 2})))
                            << "Expect " << "simple pow() fixture result to be correct" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({9}), arma::vec({1})).pow(0.5),
                     Series(arma::vec({3}), arma::vec({1})))
                            << "Expect " << "simple pow() fixture result to be correct" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({}), arma::vec({})).pow(2),
                     Series(arma::vec({}), arma::vec({})))
                            << "Expect " << "empty indices pow() fixture result to be correct" << "";
    }

    TEST(Series, fillna){
        EXPECT_PRED2(
                Series::equal,
                Series({1, 0, 3, 0, 5}, {1, 2, 3, 4, 5}),
                Series({1, NAN, 3, NAN, 5}, {1, 2, 3, 4, 5}).fillna()
        ) << "Expect " << "replace NANs with zeros";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 1, 3, 1, 5}, {1, 2, 3, 4, 5}),
                Series({1, NAN, 3, NAN, 5}, {1, 2, 3, 4, 5}).fillna(1.)
        ) << "Expect " << "replace NANs with ones";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}),
                Series({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}).fillna()
        ) << "Expect " << " remains the same as no NANs";

        EXPECT_PRED2(
                Series::equal,
                Series(),
                Series().fillna()
        ) << "Expect " << " empty array";

    }

    TEST(Series, dropna) {
        EXPECT_PRED2(
                Series::equal,
                Series({1, 3, 5}, {1, 3, 5}),
                Series({1, NAN, 3, NAN, 5}, {1, 2, 3, 4, 5}).dropna()
        ) << "Expect " << "drop NANs so indices only contains finite elements";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}),
                Series({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}).dropna()
        ) << "Expect " << " remains the same as no NANs";

        EXPECT_PRED2(
                Series::equal,
                Series(),
                Series().dropna()
        ) << "Expect " << " empty array";

        EXPECT_PRED2(
            Series::equal,
            Series({1, arma::datum::inf, 3}, {1, 2, 4}),
            Series({1, arma::datum::inf, NAN, 3}, {1, 2, 3, 4}).dropna()
        ) << "Expect " << " drops NA and preserves inf";
    }

    TEST(Series, clipTest) {
        EXPECT_PRED2(
                Series::equal,
                Series(arma::vec({}), arma::vec({})).clip(0, 1),
                Series(arma::vec({}), arma::vec({}))
        )  << "Expect " << "empty Series";

        EXPECT_PRED2(
                Series::equal,
                Series(arma::vec({1, 2, 3, 4}), arma::vec({1, 2, 3, 4})).clip(2, 3),
                Series(arma::vec({2, 2, 3, 3}), arma::vec({1, 2, 3, 4}))
        ) << "Expect " << "indices clipped to 2-3";

        EXPECT_PRED2(
                Series::equal,
                Series(arma::vec({1, 2, 3, 4}), arma::vec({1, 2, 3, 4})).clip(0, 1),
                Series(arma::vec({1, 1, 1, 1}), arma::vec({1, 2, 3, 4}))
        ) << "Expect " << "indices clipped to 2-3";
    }

    TEST(Series, MeanTest) {
        EXPECT_EQ(Series(arma::vec({3, 4}), arma::vec({1, 2})).mean(), 3.5)
                            << "Expect " << "simple mean() fixture result to be correct" << "";

        ASSERT_TRUE(isnan(Series(arma::vec({}), arma::vec({})).mean()))
                                    << "Expect " << "empty series mean() should be NAN" << "";

        EXPECT_EQ(Series(arma::vec({3, NAN, 4}), arma::vec({1, 2, 3})).mean(), 3.5)
                            << "Expect " << "simple mean() fixture result with NAN to be correct, ignoring NANs" << "";


    }

    TEST(Series, RollingQuantileTest) {
        EXPECT_PRED2(Series::equal, Series(arma::vec({}), arma::vec({})),
                     Series(arma::vec({}), arma::vec({})).rolling(3, polars::Quantile(0.5)))
                            << "Expect " << "rolling() test 1 for empty series" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, NAN}), arma::vec({1, 2})),
                     Series(arma::vec({1, 1}), arma::vec({1, 2})).rolling(3, polars::Quantile(0.5)))
                            << "Expect " << "series of size 2, rolling window of 3, should be NANS" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 1, NAN}), arma::vec({1, 2, 3})),
                     Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})).rolling(3, polars::Quantile(0.5)))
                            << "Expect " << "series of size 3, rolling window of 3" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 5, NAN}), arma::vec({1, 2, 3})),
                     Series(arma::vec({10, 1, 5}), arma::vec({1, 2, 3})).rolling(3, polars::Quantile(0.5)))
                            << "Expect " << "series of size 3, rolling window of 3" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, NAN, 5.5, NAN}), arma::vec({1, 2, 3, 4})),
                     Series(arma::vec({10, 1, 5, 6}), arma::vec({1, 2, 3, 4})).rolling(4, polars::Quantile(0.5)))
                            << "Expect " << "series of size 4, rolling window of 4" << "";

        EXPECT_PRED2(Series::almost_equal,
                     Series(arma::vec({NAN, NAN, 3.4, NAN, NAN}), arma::vec({1, 2, 3, 4, 5})),
                     Series(arma::vec({10, 1, 5, 4, 7}), arma::vec({1, 2, 3, 4, 5})).rolling(5, polars::Quantile(0.2)))
                            << "Expect " << "series of size 5, rolling window of 5, quantile 0.2" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 5.5, NAN, NAN, 5.5}), arma::vec({1, 2, 3, 4, 5})),
                     Series(arma::vec({10, 1, NAN, 4, 7}), arma::vec({1, 2, 3, 4, 5})).rolling(2, polars::Quantile(0.5)))
                            << "Expect " << "NAN test case passes" << "";


        EXPECT_PRED2(Series::equal, Series(arma::vec({5.5, 5, 3}), arma::vec({1, 2, 3})),
                     Series(arma::vec({10, 1, 5}), arma::vec({1, 2, 3})).rolling(3, polars::Quantile(0.5), 2, true))
                            << "Expect " << "rolling() test 1 for minperiods" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 7.5, NAN}), arma::vec({1, 2, 3})),
                     Series(arma::vec({10, NAN, 5}), arma::vec({1, 2, 3})).rolling(3, polars::Quantile(0.5), 2, true))
                            << "Expect " << "rolling() test 2 for minperiods, with nans" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})),
                     Series(arma::vec({1, NAN, 1}), arma::vec({1, 2, 3})).rolling(3, polars::Quantile(0.5), 1, true))
                            << "Expect " << "series of size 3, rolling window of 3" << "";

        EXPECT_PRED2(Series::equal, Series(arma::vec({1, 0.5, 1}), arma::vec({1, 2, 3})),
                     Series(arma::vec({1, NAN, 1}), arma::vec({1, 2, 3})).rolling(
                             3, polars::Quantile(0.5), 1, true, false, polars::WindowProcessor::WindowType::triang))
                            << "Expect " << "series of size 3, rolling window of 3 with triang weights (0.5,1,0.5)" << "";
    }


    TEST(Series, rolling_sum) {
        EXPECT_PRED2(
                Series::equal,
                Series().rolling(5, polars::Sum()),
                Series()
        ) << "Expect " << "empty Series returns empty Series";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(1, polars::Sum()),
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with a window of 1 the indices is returned as is";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Sum()),
                Series({NAN, 6.5, 4.5, NAN, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should be the sum, not NAN";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Sum(), 2),
                Series({3, 6.5, 4.5, 2.5, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with window=3, min_periods=2 the edge values should be NAN and the rest should sum the windows";
    }


    TEST(Series, rolling_count) {
        EXPECT_PRED2(
                Series::equal,
                Series().rolling(5, polars::Count()),
                Series()
        ) << "Expect " << "empty Series returns empty Series";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(1, polars::Count()),
                Series({1, 1, 1, 1, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with a window of 1 the indices is returned as is";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Count()),
                Series({NAN, 3, 3, NAN, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should be the count, not NAN";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Count(), 2),
                Series({2, 3, 3, 2, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with window=3, min_periods=2 then any windows with at least 2 values should be non-NAN";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Count(), 1),
                Series({2, 3, 3, 2, 1}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with window=3, min_periods=1 the edge values should be NAN and the rest should count the windows";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, NAN, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Count(0), 1),
                Series({2, 3, 2, 1, 0}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with window=3, min_periods=1 and a default of 0, all windows should have a count";

        // Symmetric = True - Odd array with odd window works (e.g. array of 5 with window of 3)
        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}).rolling(3, polars::Count(), 1, true, true),
                Series({1, 3, 3, 3, 1}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with window of 3 the indices expects smaller windows on the edges so counts are 1";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(7, polars::Count(), 1, true, true),
                Series({1, 3, 5, 5, 3, 1}, {1, 2, 3, 4, 5, 6})
        ) << "Expect " << "with window of 7 the indices expects smaller and smaller counts along the edges";

        // Even array with odd window
        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(3, polars::Count(), 1, true, true),
                Series({1, 3, 3, 3, 3, 1}, {1, 2, 3, 4, 5, 6})
        ) << "Expect " << "with an even array, weighting still works out.";
    }

    TEST(Series, operator__add) {
        EXPECT_PRED2(
                Series::equal,
                Series() + 1,
                Series()
        ) << "Expect " << "empty Series stays empty";

        EXPECT_PRED2(
                Series::equal,
                Series({3, 4}, {1, 2}) + 1,
                Series({4, 5}, {1, 2})
        ) << "Expect " << "adding 1 increases the values, not the index";

        EXPECT_PRED2(
                Series::equal,
                Series({3, 4, arma::datum::nan}, {1, 2, 3}) + 1,
                Series({4, 5, arma::datum::nan}, {1, 2, 3})
        ) << "Expect " << "adding 1 increases the values, not the index, and ignores nan";

        EXPECT_PRED2(
                Series::equal,
                Series({3, 4, 2, arma::datum::nan}, {1, 2, 3, 4}) +
                Series({3, -5, arma::datum::nan, arma::datum::nan}, {1, 2, 3, 4}),
                Series({6, -1, arma::datum::nan, arma::datum::nan}, {1, 2, 3, 4})
        ) << "Expect " << "adding positive and negative values works and nan results in nan";
    }


    TEST(Series, operator__subtract) {
        EXPECT_PRED2(
                Series::equal,
                Series() - 1,
                Series()
        ) << "Expect " << "empty Series stays empty";

        EXPECT_PRED2(
                Series::equal,
                Series({3, 4}, {1, 2}) - 1,
                Series({2, 3}, {1, 2})
        ) << "Expect " << "adding 1 increases the values, not the index";

        EXPECT_PRED2(
                Series::equal,
                Series({3, 4, arma::datum::nan}, {1, 2, 3}) - 1,
                Series({2, 3, arma::datum::nan}, {1, 2, 3})
        ) << "Expect " << "adding 1 increases the values, not the index, and ignores nan";

        EXPECT_PRED2(
                Series::equal,
                Series({-3, 4, 2, arma::datum::nan}, {1, 2, 3, 4}) -
                Series({3, -5, arma::datum::nan, arma::datum::nan}, {1, 2, 3, 4}),
                Series({-6, 9, arma::datum::nan, arma::datum::nan}, {1, 2, 3, 4})
        ) << "Expect " << "subtracting positive and negative values works and nan results in nan";
    }


    TEST(Series, operator__multiply) {
        EXPECT_PRED2(
                Series::equal,
                Series() * 2,
                Series()
        ) << "Expect " << "empty Series stays empty";

        EXPECT_PRED2(
                Series::equal,
                Series({3, 4}, {1, 2}) * 2,
                Series({6, 8}, {1, 2})
        ) << "Expect " << "multiplying by 2 changes the values, not the index";

        EXPECT_PRED2(
                Series::equal,
                Series({3, 4, arma::datum::nan}, {1, 2, 3}) * 2,
                Series({6, 8, arma::datum::nan}, {1, 2, 3})
        ) << "Expect " << "multiplying by 2 changes the values, not the index, and ignores nan";

        EXPECT_PRED2(
                Series::equal,
                Series({-3, 4, 2, arma::datum::nan}, {1, 2, 3, 4}) *
                Series({3, -5, arma::datum::nan, arma::datum::nan}, {1, 2, 3, 4}),
                Series({-9, -20, arma::datum::nan, arma::datum::nan}, {1, 2, 3, 4})
        ) << "Expect " << "multiplying positive and negative values works and nan results in nan";
    }


    TEST(Series, operator__eq) {
        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series() == Series(),
                polars::SeriesMask()
        ) << "Expect " << "empty Series stays empty";

        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series({0, 1, 2, NAN}, {1, 2, 3, 4}) ==
                Series({0, 1, 3, NAN}, {1, 2, 3, 4}),
                polars::SeriesMask({1, 1, 0, 0}, {1, 2, 3, 4})
        ) << "Expect " << "matching should match, but NAN != NAN for the elementwise operator";

        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series() == 1,
                polars::SeriesMask()
        ) << "Expect " << "empty Series stays empty";

        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series({0, 1.99, 2, 1 + 1, NAN}, {1, 2, 3, 4, 5}) == 2,
                polars::SeriesMask({0, 0, 1, 1, 0}, {1, 2, 3, 4, 5})
        ) << "Expect " << "matching should match, but NAN != NAN for the elementwise operator";
    }


    TEST(Series, operator__ne) {
        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series() != 1,
                polars::SeriesMask()
        ) << "Expect " << "empty Series stays empty";

        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series({0, 1.99, 2, 1 + 1, NAN}, {1, 2, 3, 4, 5}) != 2,
                polars::SeriesMask({1, 1, 0, 0, 0}, {1, 2, 3, 4, 5})
        ) << "Expect " << "matching should match, but NAN != NAN for the elementwise operator";
    }

    TEST(Series, operator__gt) {
        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series() > Series(),
                polars::SeriesMask()
        ) << "Expect " << "empty Series stays empty";


        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series({0, -1, 3, NAN}, {1, 2, 3, 4}) >
                Series({0, -2, 2, NAN}, {1, 2, 3, 4}),
                polars::SeriesMask({0, 1, 1, 0}, {1, 2, 3, 4})
        ) << "Expect " << "> should work as per pair, including NAN != NAN";

        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series({0, -1, 3, NAN}, {1, 2, 3, 4}) >= 0,
                polars::SeriesMask({1, 0, 1, 0}, {1, 2, 3, 4})
        ) << "Expect " << ">= should work per item, including NAN != NAN";
    }

    TEST(Series, operator__lt) {
        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series() < Series(),
                polars::SeriesMask()
        ) << "Expect " << "empty Series stays empty";


        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series({0, -2, 2, NAN}, {1, 2, 3, 4}) <
                Series({0, -1, 3, NAN}, {1, 2, 3, 4}),
                polars::SeriesMask({0, 1, 1, 0}, {1, 2, 3, 4})
        ) << "Expect " << "> should work as per pair, including NAN != NAN";

        EXPECT_PRED2(
                polars::SeriesMask::equal,
                Series({0, -1, 3, NAN}, {1, 2, 3, 4}) <= 0,
                polars::SeriesMask({1, 1, 0, 0}, {1, 2, 3, 4})
        ) << "Expect " << "<= should work per item, including NAN != NAN";
    }


    TEST(Series, apply){
        EXPECT_PRED2(
             Series::equal,
             Series({1., 2., 3.}, {1, 2, 3}),
             Series({1., 2., 3.}, {1, 2, 3}).apply(abs)
        ) << "Expect " << " should remain ideantical";

        EXPECT_PRED2(
             Series::equal,
             Series({1., 2., 3.}, {1, 2, 3}),
             Series({-1., -2., -3.}, {1, 2, 3}).apply(abs)
        ) << "Expect " << " should flip signs";

        EXPECT_PRED2(
             Series::equal,
             Series({2.7182818284590451, 7.3890560989306504, 20.085536923187668}, {1, 2, 3}),
             Series({1., 2., 3.}, {1, 2, 3}).apply(exp)
        ) << "Expect " << " should apply exponential";

        EXPECT_PRED2(
             Series::equal,
             Series({0, 0.19, 1.9, 1.9}, {1, 2, 3, 4}).apply(exp),
             Series({1., 1.2092495976572515, 6.6858944422792685, 6.6858944422792685}, {1, 2, 3, 4})
        ) << "Expect " << " should apply exponential";

    }

    TEST(Series, rolling_sum_triangle) {
        EXPECT_PRED2(
                Series::equal,
                Series().rolling(5, polars::Sum(),  0, true, false, polars::WindowProcessor::WindowType::triang),
                Series()
        ) << "Expect " << "empty Series returns empty Series";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(1, polars::Sum(), 0, true, false, polars::WindowProcessor::WindowType::triang),
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with a window of 1 the indices is returned as is";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3}, {1, 2, 3}).rolling(2, polars::Sum(), 0, true, false, polars::WindowProcessor::WindowType::triang),
                Series({NAN, 1.5, 2.5}, {1, 2, 3})
        ) << "Expect " << "with a window of 2";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Sum(), 0, true, false, polars::WindowProcessor::WindowType::triang),
                Series({NAN, 4.25, 4.0, NAN, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should be the sum, not NAN";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Sum(), 2, true, false, polars::WindowProcessor::WindowType::triang),
                Series({2, 4.25, 4.0, 0.75, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with window=3, min_periods=2 the edge values should be NAN and the rest should sum the windows";
    }

    TEST(Series, rolling_mean_triangle) {
        EXPECT_PRED2(
                Series::equal,
                Series().rolling(5, polars::Mean(),  0, true, false, polars::WindowProcessor::WindowType::triang),
                Series()
        ) << "Expect " << "empty Series returns empty Series";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(1, polars::Mean(), 0, true, false, polars::WindowProcessor::WindowType::triang),
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with a window of 1 the indices is returned as is";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3}, {1, 2, 3}).rolling(3, polars::Mean(), 0, true, false, polars::WindowProcessor::WindowType::triang),
                Series({NAN, 2, NAN}, {1, 2, 3})
        ) << "Expect " << "with a window of 3 the indices of length 3 returns just central value";

        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Mean(), 0, true, false, polars::WindowProcessor::WindowType::triang),
                Series({NAN, 2.125, 2.0, NAN, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should give weighted mean, not NAN";

        EXPECT_PRED2(
                Series::almost_equal,
                Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(5, polars::Mean(), 1, true, false, polars::WindowProcessor::WindowType::triang),
                Series({1.66666667, 2.25, 2.75, 3.33333333}, {1, 2, 3, 4})
        ) << "Expect " << "no NANs because min periods is 1.";
    }

    TEST(Series, rolling_mean_exponential) {

        EXPECT_PRED2(
                Series::almost_equal,
                Series({0.1, 0.2, 0.3, 0.4}, {1, 2, 3, 4}).rolling(4, polars::ExpMean(), 1, true, false, polars::WindowProcessor::WindowType::expn, 0.5),
                Series({0.1, 0.16666666666666667, 0.24285714285714284, 0.32666666666666666}, {1, 2, 3, 4})
        ) << "Expect " << " first value to be the same as original series.";


        EXPECT_PRED2(
                Series::equal,
                Series().rolling(4, polars::ExpMean(), 1, true, false, polars::WindowProcessor::WindowType::expn, 0.5),
                Series()
        ) << "Expect " << " empty array back.";

        EXPECT_PRED2(
                Series::equal,
                Series({1, NAN, 3}, {1, 2, 3}).rolling(3, polars::ExpMean(), 1, true, false, polars::WindowProcessor::WindowType::expn, 0.5),
                Series({1., 1., 2.6}, {1, 2, 3})
        ) << "Expect " << " ignore NANs when computing weights.";


        EXPECT_PRED2(
                Series::equal,
                Series({1, NAN, NAN, 4}, {1, 2, 3, 4}).rolling(4, polars::ExpMean(), 1, true, false, polars::WindowProcessor::WindowType::expn, 0.5),
                Series({1, 1, 1, 3.6666666666666665}, {1, 2, 3, 4})
        ) << "Expect " << "with two NANs. This differs from Pandas as it ignores NAN's when computing the weights.";


        EXPECT_PRED2(
                Series::equal,
                Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(4, polars::ExpMean(), 1, true, false, polars::WindowProcessor::WindowType::expn, 0.5),
                Series({1, 1.6666666666666667, 2.4285714285714284, 3.2666666666666666}, {1, 2, 3, 4})
        ) << "Expect " << "with a window of 4";
    }

    TEST(Series, quantile){

        EXPECT_TRUE(std::isnan(Series().quantile())) << "Expect" << " NAN for empty array";

        EXPECT_EQ(Series({1}, {1}).quantile(1), 1) << "Expect" << " one";

        EXPECT_EQ(Series({1, 2, 3}, {1, 2, 3}).quantile(), 2) << "Expect" << " median as default";

        EXPECT_FLOAT_EQ(Series({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}).quantile(0.3), 2.7) << "Expect" << " 0.3 quantile";

    }

    TEST(Series, iloc){

        auto indices = arma::uvec{1, 2};

        EXPECT_PRED2(
                Series::equal,
                Series({20, 40, 34, 10}, {1, 2, 3, 4}).iloc(indices),
                Series({40, 34}, {2, 3})) << "Expect " << "subset including specified indices to be retrieved";


        EXPECT_EQ(
                Series({1, 2, 3, 4}, {1, 2, 3, 4}).iloc(2),
            3
        ) << "Expect " << " element 3 to be retrieved";

    }

    TEST(Series, loc){

        auto labels = arma::vec{3, 4};
        EXPECT_PRED2(
                Series::equal,
                Series({2, 3, 4, 10}, {3, 4, 5, 6}).loc(labels),
                Series({2, 3}, {3, 4})
        ) << "Expect " << " indices values retrieved by label";

        auto non_valid_labels = arma::vec{30, 40};
        EXPECT_PRED2(
            Series::equal,
            Series({30, 40, 53, 32}, {3, 4, 5, 6}).loc(non_valid_labels),
            Series()
        ) << "Expect " << " empty indices since no records match labels";

        EXPECT_PRED2(
                Series::equal,
                Series({10, 30, 40, 20}, {3, 4, 5, 6}).loc(4),
                Series({30}, {4})
        ) << "Expect " << "indices values retrieved by label";

        EXPECT_PRED2(
                Series::equal,
                Series({10, 30, 40, 20}, {3, 4, 5, 6}).loc(8),
                Series()
        ) << "Expect " << "empty indices since no records match label";
    }

    TEST(Series, index_as_series){
        EXPECT_PRED2(
                Series::equal,
                Series().index_as_series(),
                Series()
        ) << "Expect " << " empty series";

        EXPECT_PRED2(
            Series::equal,
            Series({4, 5, 6, 3}, {1, 2, 3, 4}).index_as_series(),
            Series({1, 2, 3, 4}, {1, 2, 3, 4})
        ) << "Expect " << "index as a series";
    }

    TEST(Series, from_vect){

        std::vector<double> z = {1, 2, 3, 4};

        EXPECT_PRED2(
            Series::equal,
            Series::from_vect(z, z),
            Series({1, 2, 3, 4}, {1, 2, 3, 4})
        ) << "Expect " << "identical indices due to passing vectors";

        std::vector<double> y = {};
        EXPECT_PRED2(
                Series::equal,
                Series::from_vect(y, y),
                Series({}, {})
        ) << "Expect " << "identical indices due to passing vectors";
    }

    TEST(Series, to_map){

        Series z = Series({10, 20, 30, 40}, {1, 2, 3, 4});

        int i = 0;
        for (auto& pair : z.to_map()) {
            auto key = pair.first;
            auto value = pair.second;

            EXPECT_EQ(key, z.index()[i]);
            EXPECT_EQ(value, z.values()[i]);
            i++;
        }

        EXPECT_TRUE(Series().to_map().empty());
    }

} // namespace SeriesTests
