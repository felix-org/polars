//
// Created by Calvin Giles on 22/03/2018.
//
#include "gtest/gtest.h"

#include "polars/Series.h"
#include "polars/SeriesMask.h"
#include "polars/numc.h"


namespace WindowProcessorTests {
    using Series = polars::Series;

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
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(1, 0, true, false, polars::WindowProcessor::WindowType::triang).mean(),
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
                Series::equal,
                Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, 0, true, false, polars::WindowProcessor::WindowType::triang).mean(),
                Series({NAN, 2.125, 2.0, NAN, NAN}, {1, 2, 3, 4, 5})
        ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should give weighted mean, not NAN";

        EXPECT_PRED2(
                Series::almost_equal,
                Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(5, polars::Mean(), 1, true, false, polars::WindowProcessor::WindowType::triang),
                Series({1.66666667, 2.25, 2.75, 3.33333333}, {1, 2, 3, 4})
        ) << "Expect " << "no NANs because min periods is 1.";

        EXPECT_PRED2(
                Series::almost_equal,
                Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(5, 1, true, false, polars::WindowProcessor::WindowType::triang).mean(),
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


} // namespace SeriesTests
