//
// Created by Calvin Giles on 22/03/2018.
//

#include "polars/WindowProcessor.h"

#include "polars/Series.h"
#include "polars/SeriesMask.h"
#include "polars/numc.h"

#include "gtest/gtest.h"


namespace WindowProcessorTests {
using Series = polars::Series;

TEST(Series, RollingQuantileTest) {
    EXPECT_PRED2(Series::equal, Series(arma::vec({}), arma::vec({})),
                 Series(arma::vec({}), arma::vec({})).rolling(3, polars::Quantile(0.5)))
                        << "Expect " << "rolling() test 1 for empty series" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, NAN}), arma::vec({1, 2})),
                 Series(arma::vec({1, 1}), arma::vec({1, 2})).rolling(3).quantile(0.5))
                        << "Expect " << "series of size 2, rolling window of 3, should be NANS" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 1, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})).rolling(3).quantile(0.5))
                        << "Expect " << "series of size 3, rolling window of 3" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 5, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, 1, 5}), arma::vec({1, 2, 3})).rolling(3).quantile(0.5))
                        << "Expect " << "series of size 3, rolling window of 3" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, NAN, 5.5, NAN}), arma::vec({1, 2, 3, 4})),
                 Series(arma::vec({10, 1, 5, 6}), arma::vec({1, 2, 3, 4})).rolling(4).quantile(0.5))
                        << "Expect " << "series of size 4, rolling window of 4" << "";

    EXPECT_PRED2(Series::almost_equal,
                 Series(arma::vec({NAN, NAN, 3.4, NAN, NAN}), arma::vec({1, 2, 3, 4, 5})),
                 Series(arma::vec({10, 1, 5, 4, 7}), arma::vec({1, 2, 3, 4, 5})).rolling(5).quantile(0.2))
                        << "Expect " << "series of size 5, rolling window of 5, quantile 0.2" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 5.5, NAN, NAN, 5.5}), arma::vec({1, 2, 3, 4, 5})),
                 Series(arma::vec({10, 1, NAN, 4, 7}), arma::vec({1, 2, 3, 4, 5})).rolling(2).quantile(0.5))
                        << "Expect " << "NAN test case passes" << "";


    EXPECT_PRED2(Series::equal, Series(arma::vec({5.5, 5, 3}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, 1, 5}), arma::vec({1, 2, 3})).rolling(3, 2, true).quantile(0.5))
                        << "Expect " << "rolling() test 1 for minperiods" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 7.5, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, NAN, 5}), arma::vec({1, 2, 3})).rolling(3, 2, true).quantile(0.5))
                        << "Expect " << "rolling() test 2 for minperiods, with nans" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})),
                 Series(arma::vec({1, NAN, 1}), arma::vec({1, 2, 3})).rolling(3, 1, true).quantile(0.5))
                        << "Expect " << "series of size 3, rolling window of 3" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({1, 0.5, 1}), arma::vec({1, 2, 3})),
                 Series(arma::vec({1, NAN, 1}), arma::vec({1, 2, 3})).rolling(
                         3, polars::Quantile(0.5), 1, true, false, polars::WindowProcessor::WindowType::triang))
                        << "Expect " << "series of size 3, rolling window of 3 with triang weights (0.5,1,0.5)" << "";
}

TEST(Series, rolling_median) {
    EXPECT_PRED2(Series::equal, Series(arma::vec({}), arma::vec({})),
                 Series(arma::vec({}), arma::vec({})).rolling(3).median())
                        << "Expect " << "rolling() test 1 for empty series" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, NAN}), arma::vec({1, 2})),
                 Series(arma::vec({1, 1}), arma::vec({1, 2})).rolling(3).median())
                        << "Expect " << "series of size 2, rolling window of 3, should be NANS" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 1, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})).rolling(3).median())
                        << "Expect " << "series of size 3, rolling window of 3" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 5, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, 1, 5}), arma::vec({1, 2, 3})).rolling(3).median())
                        << "Expect " << "series of size 3, rolling window of 3" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, NAN, 5.5, NAN}), arma::vec({1, 2, 3, 4})),
                 Series(arma::vec({10, 1, 5, 6}), arma::vec({1, 2, 3, 4})).rolling(4).median())
                        << "Expect " << "series of size 4, rolling window of 4" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 5.5, NAN, NAN, 5.5}), arma::vec({1, 2, 3, 4, 5})),
                 Series(arma::vec({10, 1, NAN, 4, 7}), arma::vec({1, 2, 3, 4, 5})).rolling(2).median())
                        << "Expect " << "NAN test case passes" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({5.5, 5, 3}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, 1, 5}), arma::vec({1, 2, 3})).rolling(3, 2, true).median())
                        << "Expect " << "rolling() test 1 for minperiods" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 7.5, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, NAN, 5}), arma::vec({1, 2, 3})).rolling(3, 2, true).median())
                        << "Expect " << "rolling() test 2 for minperiods, with nans" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})),
                 Series(arma::vec({1, NAN, 1}), arma::vec({1, 2, 3})).rolling(3, 1, true).median())
                        << "Expect " << "series of size 3, rolling window of 3" << "";
}


TEST(Series, rolling_min) {
    EXPECT_PRED2(Series::equal, Series(arma::vec({}), arma::vec({})),
                 Series(arma::vec({}), arma::vec({})).rolling(3).min())
                        << "Expect " << "rolling() test 1 for empty series" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, NAN}), arma::vec({1, 2})),
                 Series(arma::vec({1, 1}), arma::vec({1, 2})).rolling(3).min())
                        << "Expect " << "series of size 2, rolling window of 3, should be NANS" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 1, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})).rolling(3).min())
                        << "Expect " << "series of size 3, rolling window of 3" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 1, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, 1, 5}), arma::vec({1, 2, 3})).rolling(3).min())
                        << "Expect " << "series of size 3, rolling window of 3" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, NAN, 1, NAN}), arma::vec({1, 2, 3, 4})),
                 Series(arma::vec({10, 1, 5, 6}), arma::vec({1, 2, 3, 4})).rolling(4).min())
                        << "Expect " << "series of size 4, rolling window of 4" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 1, NAN, NAN, 4}), arma::vec({1, 2, 3, 4, 5})),
                 Series(arma::vec({10, 1, NAN, 4, 7}), arma::vec({1, 2, 3, 4, 5})).rolling(2).min())
                        << "Expect " << "NAN test case passes" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, 1, 5}), arma::vec({1, 2, 3})).rolling(3, 2, true).min())
                        << "Expect " << "rolling() test 1 for minperiods" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 5, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, NAN, 5}), arma::vec({1, 2, 3})).rolling(3, 2, true).min())
                        << "Expect " << "rolling() test 2 for minperiods, with nans" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})),
                 Series(arma::vec({1, NAN, 1}), arma::vec({1, 2, 3})).rolling(3, 1, true).min())
                        << "Expect " << "series of size 3, rolling window of 3" << "";
}


TEST(Series, rolling_max) {
    EXPECT_PRED2(Series::equal, Series(arma::vec({}), arma::vec({})),
                 Series(arma::vec({}), arma::vec({})).rolling(3).max())
                        << "Expect " << "rolling() test 1 for empty series" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, NAN}), arma::vec({1, 2})),
                 Series(arma::vec({1, 1}), arma::vec({1, 2})).rolling(3).max())
                        << "Expect " << "series of size 2, rolling window of 3, should be NANS" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 1, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})).rolling(3).max())
                        << "Expect " << "series of size 3, rolling window of 3" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 10, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, 1, 5}), arma::vec({1, 2, 3})).rolling(3).max())
                        << "Expect " << "series of size 3, rolling window of 3" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, NAN, 10, NAN}), arma::vec({1, 2, 3, 4})),
                 Series(arma::vec({10, 1, 5, 6}), arma::vec({1, 2, 3, 4})).rolling(4).max())
                        << "Expect " << "series of size 4, rolling window of 4" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 10, NAN, NAN, 7}), arma::vec({1, 2, 3, 4, 5})),
                 Series(arma::vec({10, 1, NAN, 4, 7}), arma::vec({1, 2, 3, 4, 5})).rolling(2).max())
                        << "Expect " << "NAN test case passes" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({10, 10, 5}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, 1, 5}), arma::vec({1, 2, 3})).rolling(3, 2, true).max())
                        << "Expect " << "rolling() test 1 for minperiods" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({NAN, 10, NAN}), arma::vec({1, 2, 3})),
                 Series(arma::vec({10, NAN, 5}), arma::vec({1, 2, 3})).rolling(3, 2, true).max())
                        << "Expect " << "rolling() test 2 for minperiods, with nans" << "";

    EXPECT_PRED2(Series::equal, Series(arma::vec({1, 1, 1}), arma::vec({1, 2, 3})),
                 Series(arma::vec({1, NAN, 1}), arma::vec({1, 2, 3})).rolling(3, 1, true).max())
                        << "Expect " << "series of size 3, rolling window of 3" << "";
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


TEST(Series, rolling_std) {
    EXPECT_PRED2(
            Series::equal,
            Series().rolling(5, polars::Std()),
            Series()
    ) << "Expect " << "empty Series returns empty Series";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(1, polars::Std()),
            Series({NAN, NAN, NAN, NAN, NAN}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with a window of 1 std is undefined so returns NA";

    EXPECT_PRED2(
            Series::almost_equal,
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Std()),
            Series({NAN, arma::stddev(arma::vec{1, 2, 3.5}), arma::stddev(arma::vec{2, 3.5, -1}), NAN, NAN},
                   {1, 2, 3, 4, 5})
    ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should be the std, not NAN";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Std(), 2),
            Series({
                           arma::stddev(arma::vec{1, 2}),
                           arma::stddev(arma::vec{1, 2, 3.5}),
                           arma::stddev(arma::vec{2, 3.5, -1}),
                           arma::stddev(arma::vec{3.5, -1}),
                           NAN}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with window=3, min_periods=2 any windows with 2 non-NAN values should be the std, not NAN";
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
    ) << "Expect "
      << "with window=3, min_periods=1 the edge values should be NAN and the rest should count the windows";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3.5, NAN, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Count(0), 1),
            Series({2, 3, 2, 1, 0}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with window=3, min_periods=1 and a default of 0, all windows should have a count";

    EXPECT_PRED2(
        Series::equal,
        Series({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(7, polars::Count(), 1, true, false),
        Series({4, 5, 6, 6, 5, 4}, {1, 2, 3, 4, 5, 6})
    ) << "Expect " << "with window of 7 the indices expects smaller and smaller counts along the edges";

    // Symmetric = True - Odd array with odd window works (e.g. array of 5 with window of 3)
    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}).rolling(3, polars::Count(), 1, true, true),
            Series({1, 3, 3, 3, 1}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with window of 3 the indices expects smaller windows on the edges so counts are 1";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(7, polars::Count(), 1, true, true),
            Series({1, 3, 5, 6, 5, 3}, {1, 2, 3, 4, 5, 6})
    ) << "Expect " << "with window of 7 the indices expects smaller and smaller counts along the edges";

    // Even array with odd window
    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(3, polars::Count(), 1, true, true),
            Series({1, 3, 3, 3, 3, 1}, {1, 2, 3, 4, 5, 6})
    ) << "Expect " << "with an even array, weighting still works out.";
}


TEST(Series, rolling_count_alignment){

    EXPECT_PRED2(
           Series::equal,
            Series({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(7, polars::Count(), 1, true, false),
            Series({4, 5, 6, 6,  5, 4}, {1, 2, 3, 4, 5, 6})
    ) << "Expect " << "symmetric False and center true. This matches pandas.";

    // TODO: Cases below need to be looked at further.
    EXPECT_PRED2(
        Series::equal,
        Series({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(7, polars::Count(), 1, false, false),
        Series({3, 4, 5, 6, 6, 5}, {1, 2, 3, 4, 5, 6})
    ) << "Expect " << "symmetric False and center False. Window in pandas is completely left aligned.";

    EXPECT_PRED2(
        Series::equal,
        Series({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(7, polars::Count(), 1, false, true),
        Series({NAN, 2, 4, 6, 6, 4}, {1, 2, 3, 4, 5, 6})
    ) << "Expect " << "symmetric true and center false. ";

    EXPECT_PRED2(
        Series::equal,
        Series({1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(7, polars::Count(), 1, true, true),
        Series({1, 3, 5, 6, 5, 3}, {1, 2, 3, 4, 5, 6})
    ) << "Expect " << "symmetric true and center true. ";

}

TEST(Series, rolling_sum_triangle) {
    EXPECT_PRED2(
            Series::equal,
            Series().rolling(5, polars::Sum(), 0, true, false, polars::WindowProcessor::WindowType::triang),
            Series()
    ) << "Expect " << "empty Series returns empty Series";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(1, polars::Sum(), 0, true, false,
                                                                  polars::WindowProcessor::WindowType::triang),
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with a window of 1 the indices is returned as is";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3}, {1, 2, 3}).rolling(2, polars::Sum(), 0, true, false,
                                                 polars::WindowProcessor::WindowType::triang),
            Series({NAN, 1.5, 2.5}, {1, 2, 3})
    ) << "Expect " << "with a window of 2";

    EXPECT_PRED2(
        Series::equal,
        Series({NAN, 1, 2, 3}, {0, 1, 2, 3}).rolling(2, polars::Sum(), 0, true, false,
                                             polars::WindowProcessor::WindowType::triang),
        Series({NAN, NAN, 1.5, 2.5}, {0, 1, 2, 3})
    ) << "Expect " << "with a window of 2";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Sum(), 0, true, false,
                                                                  polars::WindowProcessor::WindowType::triang),
            Series({NAN, 4.25, 4.0, NAN, NAN}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should be the sum, not NAN";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Sum(), 2, true, false,
                                                                  polars::WindowProcessor::WindowType::triang),
            Series({2, 4.25, 4.0, 0.75, NAN}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with window=3, min_periods=2 the edge values should be NAN and the rest should sum the windows";
}

TEST(Series, rolling_mean_normal) {
    EXPECT_PRED2(
            Series::equal,
            Series().rolling(5, polars::Mean(), 1),
            Series()
    ) << "Expect " << "empty Series returns empty Series";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3, 4, 3, 2, 1}, {1, 2, 3, 4, 5, 6, 7}).rolling(5, polars::Mean(), 1),
            Series({2, 2.5, 2.6, 2.8, 2.6, 2.5, 2}, {1, 2, 3, 4, 5, 6, 7})
    ) << "Expect " << "series containing means when window is smaller than series";

    EXPECT_PRED2(
            Series::equal,
            Series({NAN, 2, 3, NAN, 3, NAN, 1}, {1, 2, 3, 4, 5, 6, 7}).rolling(13, polars::Mean(), 1),
            Series({2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25}, {1, 2, 3, 4, 5, 6, 7})
    ) << "Expect the mean of a series with NAN in results in the mean of the non-nans when the window much bigger than the series";

    EXPECT_PRED2(
            Series::equal,
            Series({NAN, 2, 3, NAN, 3, NAN, 1}, {1, 2, 3, 4, 5, 6, 7}).rolling(5, polars::Mean(), 1),
            Series({2.5, 2.5, 2.6666666666666665, 2.6666666666666665, 2.3333333333333335, 2, 2}, {1, 2, 3, 4, 5, 6, 7})
    ) << "Expect with small windows the mean ignored NANs";
}

TEST(Series, rolling_mean_triangle) {
    EXPECT_PRED2(
            Series::equal,
            Series().rolling(5, polars::Mean(), 0, true, false, polars::WindowProcessor::WindowType::triang),
            Series()
    ) << "Expect " << "empty Series returns empty Series";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(1, polars::Mean(), 0, true, false,
                                                                  polars::WindowProcessor::WindowType::triang),
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with a window of 1 the indices is returned as is";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(1, 0, true, false,
                                                                  polars::WindowProcessor::WindowType::triang).mean(),
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with a window of 1 the indices is returned as is";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3}, {1, 2, 3}).rolling(3, polars::Mean(), 0, true, false,
                                                 polars::WindowProcessor::WindowType::triang),
            Series({NAN, 2, NAN}, {1, 2, 3})
    ) << "Expect " << "with a window of 3 the indices of length 3 returns just central value";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, polars::Mean(), 0, true, false,
                                                                  polars::WindowProcessor::WindowType::triang),
            Series({NAN, 2.125, 2.0, NAN, NAN}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should give weighted mean, not NAN";

    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3.5, -1, NAN}, {1, 2, 3, 4, 5}).rolling(3, 0, true, false,
                                                                  polars::WindowProcessor::WindowType::triang).mean(),
            Series({NAN, 2.125, 2.0, NAN, NAN}, {1, 2, 3, 4, 5})
    ) << "Expect " << "with a window of 3 any windows with 3 non-NAN values should give weighted mean, not NAN";
}


TEST(Series, edge_cases_with_NAN){

    EXPECT_PRED2(Series::equal, Series({1},{1}).rolling(1, 0, true, false, polars::WindowProcessor::WindowType::triang).mean(),
                 Series({1}, {1})
    );

    EXPECT_PRED2(
        Series::almost_equal,
        Series({NAN, 2, 3, 4}, {1, 2, 3, 4}).rolling(30, polars::Mean(), 1, true, false,
                                                   polars::WindowProcessor::WindowType::triang),
        Series({2.9466666666666668, 2.950617283950617, 2.976470588235294, 3.023529411764706}, {1, 2, 3, 4})
    ) << "Expect " << " matching due to center being true";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({NAN, 2, 3, 4}, {1, 2, 3, 4}).rolling(30, polars::Mean(), 1, false, false,
                                                   polars::WindowProcessor::WindowType::triang),
        Series({NAN, 2.0, 2.2500000000000004, 2.5555555555555554}, {1, 2, 3, 4})
    ) << "Expect " << " matching due to center being false";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, NAN, 4}, {1, 2, 3, 4}).rolling(30, polars::Mean(), 1, true, false,
                                                     polars::WindowProcessor::WindowType::triang),
        Series({2.2151898734177218, 2.253012048192771, 2.325301204819277, 2.4074074074074074}, {1, 2, 3, 4})
    ) << "Expect " << " matching due to center being true";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, NAN, 4}, {1, 2, 3, 4}).rolling(30, polars::Mean(), 1, false, false,
                                                     polars::WindowProcessor::WindowType::triang),
        Series({1.0, 1.2500000000000002, 1.3750000000000002, 1.6153846153846152}, {1, 2, 3, 4})
    ) << "Expect " << " matching due to center being false";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, NAN, 3}, {1, 2, 3}).rolling(5, polars::Mean(), 1, true, false,
                                             polars::WindowProcessor::WindowType::triang),
        Series({1.5, 2.0, 2.5000000000000004}, {1, 2, 3})
    ) << "Expect " << " matching center = true";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, NAN, 3}, {1, 2, 3}).rolling(5, polars::Mean(), 1, false, false,
                                             polars::WindowProcessor::WindowType::triang),
        Series({1.0, 1.0, 1.5}, {1, 2, 3})
    ) << "Expect " << " matching center = false";

}

TEST(Series, edge_cases_without_NAN){

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(30, polars::Mean(), 1, true, false,
                                                   polars::WindowProcessor::WindowType::triang),
        Series({2.4038461538461537, 2.436363636363636, 2.5, 2.5636363636363635}, {1, 2, 3, 4})
    ) << "Expect " << " matching due to center being true";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(30, polars::Mean(), 1, false, false,
                                                   polars::WindowProcessor::WindowType::triang),
        Series({1.0, 1.2500000000000002, 1.5555555555555556, 1.875}, {1, 2, 3, 4})
    ) << "Expect " << " matching due to center being false";


    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(6, polars::Mean(), 1, true, false,
                                                   polars::WindowProcessor::WindowType::triang),
        Series({1.5555555555555554, 2.0, 2.5, 3.0}, {1, 2, 3, 4})
    ) << "Expect " << " matching due to center being true";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(6, polars::Mean(), 1, false, false,
                                                   polars::WindowProcessor::WindowType::triang),
        Series({1.0, 1.25, 1.5555555555555554, 2.0}, {1, 2, 3, 4})
    ) << "Expect " << " matching due to center being false";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(5, polars::Mean(), 1, true, false,
                                                   polars::WindowProcessor::WindowType::triang),
        Series({1.6666666666666667, 2.25, 2.7499999999999996, 3.333333333333333}, {1, 2, 3, 4})
    ) << "Expect " << " matching due to center being true";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({10, 2, 3, 4}, {1, 2, 3, 4}).rolling(5, polars::Mean(), 1, false, false,
                                                   polars::WindowProcessor::WindowType::triang),
        Series({10.0, 7.333333333333333, 6.166666666666668, 4.5}, {1, 2, 3, 4})
    ) << "Expect " << " matching due to center being false and difference of 1";

    // ODD ARRAY

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, 3}, {1, 2, 3}).rolling(5, polars::Mean(), 1, true, false,
                                                   polars::WindowProcessor::WindowType::triang),
        Series({1.6666666666666667, 2.0, 2.333333333333333}, {1, 2, 3})
    ) << "Expect " << " matching center = true";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, 3}, {1, 2, 3}).rolling(5, polars::Mean(), 1, false, false,
                                             polars::WindowProcessor::WindowType::triang),
        Series({1.0, 1.3333333333333333, 1.6666666666666667}, {1, 2, 3})
    ) << "Expect " << " matching center = false";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, 3}, {1, 2, 3}).rolling(4, polars::Mean(), 1, true, false,
                                             polars::WindowProcessor::WindowType::triang),
        Series({1.25, 1.7142857142857142, 2.2857142857142856}, {1, 2, 3})
    ) << "Expect " << " matching center = true";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, 3}, {1, 2, 3}).rolling(4, polars::Mean(), 1, false, false,
                                             polars::WindowProcessor::WindowType::triang),
        Series({1.0, 1.25, 1.7142857142857142}, {1, 2, 3})
    ) << "Expect " << " matching center = false";

    // If we are just one above window we need to add NAN at each side for window centred

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(8, 1, true, false,
                                                   polars::WindowProcessor::WindowType::triang).mean(),
        Series({1.875, 2.1818181818181817, 2.5, 2.8181818181818183}, {1, 2, 3, 4})
    ) << "Expect " << "no NANs because min periods is 1.";

}


TEST(Series, rolling_mean_triangle__larger_window) {

    arma::vec input_values = {0.1, 0.3, 0.5, 0.4, 0.7, 0.9, 0.3, 0.1};
    arma::vec input_timestamps = {1, 2, 3, 4, 5, 6, 7, 8};

    Series input = Series(input_values, input_timestamps);

    int window_size = 20;
    // THESE ARE NOT CENTER = TRUE values :P
    arma::vec expected_values_center_false = {0.10000000000000002, 0.14999999999999999, 0.21111111111111114, 0.25624999999999998,
                                 0.29599999999999999, 0.34166666666666667, 0.37551020408163266,0.38906250000000003};

    auto actual = input.rolling(window_size, polars::Mean(), 1, false, false, polars::WindowProcessor::WindowType::triang);

    EXPECT_PRED2(Series::almost_equal, actual, Series(expected_values_center_false, input_timestamps));

    arma::vec expected_values_center_true = {0.39687500000000003, 0.40454545454545465, 0.4175, 0.4293650793650794,
                                              0.43984375, 0.44682539682539685, 0.44250000000000006, 0.4318181818181819};

    actual = input.rolling(window_size, polars::Mean(), 1, true, false, polars::WindowProcessor::WindowType::triang);

    EXPECT_PRED2(Series::almost_equal, actual, Series(expected_values_center_true, input_timestamps));

}

// Tests for exponential rolling window
TEST(Series, rolling_mean_exponential__varying_window_size){

    arma::vec input_values = {8.082052269494929, 7.0994737808621, 6.055734030373877, 4.9285535425112945, 4.2505326035450794};
    arma::vec input_timestamps = {1, 2, 3, 4, 5};

    auto expected = Series(
        {8.082052269494929,
         7.564905696530282,
         7.008015782819431,
         6.40334516917938,
         5.87764065265126}, input_timestamps
    );

    Series input = Series(input_values, input_timestamps);

    for(int i = 1; i<= 60; i++) {
        Series actual = input.rolling(
            i, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.1
        );

        EXPECT_PRED2(Series::almost_equal, actual, expected) << "Expect " << " matches expectation set by pandas.";
    }
}


TEST(Series, rolling_mean_exponential__shorter_window_size){

    // Test cases in which window size < array size

    // even window size
    int window_size = 2;
    int decay_windows = 2;

    arma::vec input_values = {5, 6, 7}; // odd sized array
    arma::vec input_timestamps = {1, 2, 3};
    arma::vec expected = {5.0, 5.666666666666667, 6.428571428571429};

    Series input = Series(input_values, input_timestamps);

    Series actual = input.rolling(
        window_size, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 1./decay_windows
    );

    EXPECT_PRED2(Series::almost_equal, actual, Series(expected, input_timestamps))  << "Expect " << " for the odd/even case matches expectation set by pandas.";

    input_values = {5, 6, 7, 8}; // even sized array
    input_timestamps = {1, 2, 3, 4};
    expected = {5.0, 5.666666666666667, 6.428571428571429, 7.266666666666667};

    input = Series(input_values, input_timestamps);

    actual = input.rolling(
        window_size, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 1./decay_windows
    );

    EXPECT_PRED2(Series::almost_equal, actual, Series(expected, input_timestamps)) << "Expect " << " for the even/even case matches expectation set by pandas.";

    // odd window size
    window_size = 3;

    input_values = {5, 6, 7, 8, 9}; // odd sized array
    input_timestamps = {1, 2, 3, 4, 5};
    expected = {5.0, 5.666666666666667, 6.428571428571429, 7.266666666666667, 8.161290322580646};

    input = Series(input_values, input_timestamps);

    actual = input.rolling(
        window_size, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 1./decay_windows
    );

    EXPECT_PRED2(Series::almost_equal, actual, Series(expected, input_timestamps)) << "Expect " << " for the odd/odd case matches expectation set by pandas.";

    input_values = {5, 6, 7, 8, 9, 10}; // even sized array
    input_timestamps = {1, 2, 3, 4, 5, 6};
    expected = {5.0, 5.666666666666667, 6.428571428571429, 7.266666666666667, 8.161290322580646, 9.095238095238095};

    input = Series(input_values, input_timestamps);

    actual = input.rolling(
        window_size, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 1./decay_windows
    );

    EXPECT_PRED2(Series::almost_equal, actual, Series(expected, input_timestamps)) << "Expect " << " for the even/odd case matches expectation set by pandas.";
}


TEST(Series, rolling_mean_exponential_cases){

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, NAN, 4, 5, 6}, {1, 2, 3, 4, 5, 6}).rolling(
            6, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.5
        ),
        Series({1.0, 1.6666666666666667, 1.6666666666666667, 3.3636363636363638, 4.333333333333333, 5.237288135593221},
               {1, 2, 3, 4, 5, 6})
    ) << "Expect " << " matches pandas expectation";


  EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, NAN, 4, NAN, 6}, {1,2,3,4,5,6}).rolling(
            8, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.5
        ),
        Series({1.0, 1.6666666666666667, 1.6666666666666667, 3.3636363636363638, 3.3636363636363638, 5.325581395348837},
               {1, 2, 3, 4, 5, 6})
    ) << "Expect " << " matches pandas expectation";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({1, 2, NAN, 4, NAN}, {1,2,3,4,5}).rolling(
            7, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.5
        ),
        Series({1.0, 1.6666666666666667, 1.6666666666666667, 3.3636363636363638, 3.3636363636363638},
               {1, 2, 3, 4, 5})
    ) << "Expect " << " matches pandas expectation";


    EXPECT_PRED2(
        Series::almost_equal,
        Series({0.1,0.2,0.3,0.4}, {1,2,3,4}).rolling(
            9, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.1
        ),
        Series({0.1, 0.15263157894736845, 0.20701107011070113, 0.2631288165164292}, {1,2,3,4})
    ) << "Expect " << " matches pandas expectation";


    EXPECT_PRED2(
        Series::almost_equal,
        Series({0.1,0.2,0.3,0.4}, {1,2,3,4}).rolling(
            8, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.1
        ),
        Series({0.1, 0.15263157894736845, 0.20701107011070113, 0.2631288165164292}, {1,2,3,4})
    ) << "Expect " << " matches pandas expectation";


    EXPECT_PRED2(
        Series::almost_equal,
        Series({0.1,0.2,0.3,0.4}, {1,2,3,4}).rolling(
            7, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.1
        ),
        Series({0.1, 0.15263157894736845, 0.20701107011070113, 0.2631288165164292}, {1,2,3,4})
    ) << "Expect " << " matches pandas expectation";

    //Values at end duplicated.
    EXPECT_PRED2(
        Series::almost_equal,
        Series({0.1,0.2,0.3,0.4}, {1,2,3,4}).rolling(
            6, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.1
        ),
        Series({0.1, 0.15263157894736845, 0.20701107011070113, 0.2631288165164292}, {1,2,3,4})
    ) << "Expect " << " matches pandas expectation";

    // Double first value and then correct
    EXPECT_PRED2(
        Series::almost_equal,
        Series({0.1,0.2,0.3,0.4}, {1,2,3,4}).rolling(
            5, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.1
        ),
        Series({0.1, 0.15263157894736845, 0.20701107011070113, 0.2631288165164292}, {1,2,3,4})
    ) << "Expect " << " matches pandas expectations";

    // False and False array same size
    EXPECT_PRED2(
        Series::almost_equal,
        Series({0.1,0.2,0.3,0.4}, {1,2,3,4}).rolling(
            4, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.1
        ),
        Series({0.1, 0.15263157894736845, 0.20701107011070113, 0.2631288165164292}, {1,2,3,4})
    ) << "Expect " << " matches pandas expectation";

    EXPECT_PRED2(
        Series::almost_equal,
        Series({0.1,0.2,0.3,0.4}, {1,2,3,4}).rolling(
            4, polars::ExpMean(), 1, true, false, polars::WindowProcessor::WindowType::expn, 0.1
        ),
        Series({0.1, 0.15263157894736845, 0.20701107011070113, 0.2631288165164292}, {1,2,3,4})
    ) << "Expect " << " matches pandas expectation";

}


TEST(Series, rolling_mean_exponential) {

    EXPECT_PRED2(
        Series::almost_equal,
        Series({0.1, NAN, 0.3, 0.4}, {1, 2, 3, 4}).fillna(0).rolling(3, polars::ExpMean(), 1, false, false,
                                                                        polars::WindowProcessor::WindowType::expn, 0.5),
        Series({0.1, 0.03333333333333333, 0.18571428571428572, 0.3}, {1, 2, 3, 4})
    );

    EXPECT_PRED2(
            Series::almost_equal,
            Series({0.1, 0.2, 0.3, 0.4}, {1, 2, 3, 4}).rolling(4, polars::ExpMean(), 1, true, false,
                                                               polars::WindowProcessor::WindowType::expn, 0.5),
            Series({0.1, 0.16666666666666667, 0.24285714285714284, 0.32666666666666666}, {1, 2, 3, 4})
    ) << "Expect " << " first value to be the same as original series.";

    EXPECT_PRED2(
            Series::almost_equal,
            Series({0.1, 0.2, 0.3, 0.4}, {1, 2, 3, 4}).rolling(4, polars::ExpMean(), 1, false, false,
                                                               polars::WindowProcessor::WindowType::expn, 0.5),
            Series({0.1, 0.16666666666666667, 0.24285714285714284, 0.32666666666666666}, {1, 2, 3, 4})
    ) << "Expect " << " first value to be the same as original series.";

    EXPECT_PRED2(
            Series::equal,
            Series().rolling(4, polars::ExpMean(), 1, false, false, polars::WindowProcessor::WindowType::expn, 0.5),
            Series()
    ) << "Expect " << " empty array back.";

    EXPECT_PRED2(
            Series::equal,
            Series({1, NAN, 3}, {1, 2, 3}).rolling(3, polars::ExpMean(), 1, false, false,
                                                   polars::WindowProcessor::WindowType::expn, 0.5),
            Series({1., 1., 2.6}, {1, 2, 3})
    ) << "Expect " << " ignore NANs when computing weights.";


    EXPECT_PRED2(
            Series::almost_equal,
            Series({1, NAN, NAN, 4}, {1, 2, 3, 4}).rolling(4, polars::ExpMean(), 1, false, false,
                                                           polars::WindowProcessor::WindowType::expn, 0.5),
            Series({1, 1, 1, 3.6666666666666665}, {1, 2, 3, 4})
    ) << "Expect " << "with two NANs. This differs from Pandas as it ignores NAN's when computing the weights.";


    EXPECT_PRED2(
            Series::equal,
            Series({1, 2, 3, 4}, {1, 2, 3, 4}).rolling(4, polars::ExpMean(), 1, false, false,
                                                       polars::WindowProcessor::WindowType::expn, 0.5),
            Series({1, 1.6666666666666667, 2.4285714285714284, 3.2666666666666666}, {1, 2, 3, 4})
    ) << "Expect " << "with a window of 4";
}


TEST(Series, rolling_mean_exponential_NAN) {

    arma::vec inp(6);
    inp.fill(NAN);

    EXPECT_PRED2(
        Series::equal,
        Series({NAN, NAN, 2, 3, NAN, 4}, {1, 2, 3, 4, 5, 6}).rolling(10, polars::ExpMean(), 1, false, false,
                                                             polars::WindowProcessor::WindowType::expn, 0.5),
        Series({NAN, NAN, 2.0, 2.6666666666666665, 2.6666666666666665, 3.6363636363636362}, {1, 2, 3, 4, 5, 6})
    ) << "Expect " << " same result as without nans when window size is larger than array and intermediate NAN";

    EXPECT_PRED2(
        Series::equal,
        Series({NAN, NAN, 2, 3, 4}, {1, 2, 3, 4, 5}).rolling(10, polars::ExpMean(), 1, false, false,
                                                             polars::WindowProcessor::WindowType::expn, 0.5),
        Series({NAN, NAN, 2.0, 2.6666666666666665, 3.4285714285714284}, {1, 2, 3, 4, 5})
    ) << "Expect " << " same result as without nans when window size is larger than array";

    EXPECT_PRED2(
        Series::equal,
        Series({NAN, NAN, 2, 3, 4}, {1, 2, 3, 4, 5}).rolling(3, polars::ExpMean(), 1, false, false,
                                                 polars::WindowProcessor::WindowType::expn, 0.5),
        Series({NAN, NAN, 2.0, 2.6666666666666665, 3.4285714285714284}, {1, 2, 3, 4, 5})
    ) << "Expect " << " same result as without nans when window size is smaller than array";

    EXPECT_PRED2(
        Series::equal,
        Series({NAN, NAN, 2}, {1, 2, 3}).rolling(3, polars::ExpMean(), 1, false, false,
                                         polars::WindowProcessor::WindowType::expn, 0.5),
        Series({NAN, NAN, 2}, {1, 2, 3})
    ) << "Expect " << "identity preserving two nans";


    EXPECT_PRED2(
        Series::equal,
        Series({NAN, 2}, {1, 2}).rolling(3, polars::ExpMean(), 1, false, false,
                                                                         polars::WindowProcessor::WindowType::expn, 0.5),
        Series({NAN, 2}, {1, 2})
    ) << "Expect " << "identity preserving one nan";

    EXPECT_PRED2(
        Series::equal,
        Series(inp, {1, 2, 3, 4, 5, 6}).rolling(6, polars::ExpMean(), 1, false, false,
                                                polars::WindowProcessor::WindowType::expn, 0.5),
        Series(inp, {1, 2, 3, 4, 5, 6})
    ) << "Expect " << "with a window of 6, all nans back";

}


} // namespace SeriesTests
