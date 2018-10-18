//
// Created by Calvin Giles on 22/03/2018.
//

#include "polars/Series.h"

#include "polars/SeriesMask.h"
#include "polars/numc.h"

#include "gtest/gtest.h"


namespace SeriesTests {
using namespace polars;

TEST(Series, constructors) {
    EXPECT_PRED2(Series::equal, Series(SeriesMask()), Series())
    << "Expect constructing with empty SeriesMask is the same as empty constructor";
    EXPECT_PRED2(Series::equal, Series(SeriesMask({true, false}, {1, 2})), Series({1, 0}, {1, 2}))
    << "Expect true becomes 1 and false becomes 0";

    auto foo = [](Series s){return s;};
    EXPECT_PRED2(Series::equal, foo(SeriesMask({true, false}, {1, 2})), Series({1, 0}, {1, 2}))
    << "Expect implicit conversion from SeriesMask to Series is possible so that you can pass a SeriesMask to a function expecting a Series";

}

TEST(Series, from_map) {
    EXPECT_PRED2(Series::equal, Series::from_map({}), Series());

    EXPECT_PRED2(Series::equal, Series::from_map({{1, 3},
                                                  {2, 4}}), Series({3, 4}, {1, 2}));

}

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

TEST(Series, fillna) {
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
    ) << "Expect " << "empty Series";

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

TEST(Series, CountTest) {
    EXPECT_EQ(Series(arma::vec({3, 4}), arma::vec({1, 2})).count(), 2)
                        << "Expect " << "simple count() fixture result to be correct" << "";

    EXPECT_EQ(Series().count(), 0) << "Expect " << "empty series count() should be 0" << "";

    EXPECT_EQ(Series(arma::vec({3, NAN, 4}), arma::vec({1, 2, 3})).count(), 2)
                        << "Expect " << "simple count() fixture result with NAN to be correct, ignoring NANs" << "";


}

TEST(Series, SumTest) {
    EXPECT_EQ(Series(arma::vec({3, 4}), arma::vec({1, 2})).sum(), 7)
                        << "Expect " << "simple sum() fixture result to be correct" << "";

    ASSERT_TRUE(std::isnan(Series(arma::vec({}), arma::vec({})).sum()))
                                << "Expect " << "empty series sum() should be NAN" << "";

    EXPECT_EQ(Series(arma::vec({3, NAN, 4}), arma::vec({1, 2, 3})).sum(), 7)
                        << "Expect " << "simple sum() fixture result with NAN to be correct, ignoring NANs" << "";


}

TEST(Series, MeanTest) {
    EXPECT_EQ(Series(arma::vec({3, 4}), arma::vec({1, 2})).mean(), 3.5)
                        << "Expect " << "simple mean() fixture result to be correct" << "";

    ASSERT_TRUE(std::isnan(Series(arma::vec({}), arma::vec({})).mean()))
                                << "Expect " << "empty series mean() should be NAN" << "";

    EXPECT_EQ(Series(arma::vec({3, NAN, 4}), arma::vec({1, 2, 3})).mean(), 3.5)
                        << "Expect " << "simple mean() fixture result with NAN to be correct, ignoring NANs" << "";


}

TEST(Series, StdTest) {
    auto root_2 = std::pow(2, .5);
    EXPECT_FLOAT_EQ(Series(arma::vec({3, 4}), arma::vec({1, 2})).std(), 1 / root_2)
                        << "Expect " << "simple std() fixture result to be correct" << "";

    EXPECT_FLOAT_EQ(Series(arma::vec({3, 4}), arma::vec({1, 2})).std(0), 0.5)
                        << "Expect " << "simple std() fixture result to be correct" << "";

    ASSERT_TRUE(std::isnan(Series(arma::vec({}), arma::vec({})).std()))
                                << "Expect " << "empty series std() should be NAN" << "";

    ASSERT_TRUE(std::isnan(Series(arma::vec({3}), arma::vec({1})).std()))
                                << "Expect " << "Series with 1 element std() should be NAN";

    EXPECT_FLOAT_EQ(Series(arma::vec({3, NAN, 4}), arma::vec({1, 2, 3})).std(), 1 / root_2)
                        << "Expect " << "simple std() fixture result with NAN to be correct, ignoring NANs" << "";


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


TEST(Series, apply) {
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

TEST(Series, quantile) {

    EXPECT_TRUE(std::isnan(Series().quantile())) << "Expect" << " NAN for empty array";

    EXPECT_EQ(Series({1}, {1}).quantile(1), 1) << "Expect" << " one";

    EXPECT_EQ(Series({1, 2, 3}, {1, 2, 3}).quantile(), 2) << "Expect" << " median as default";

    EXPECT_FLOAT_EQ(Series({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}).quantile(0.3), 2.7)
                        << "Expect" << " 0.3 quantile";

}

TEST(Series, iloc) {

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

TEST(Series, loc) {

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

TEST(Series, index_as_series) {
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

TEST(Series, from_vect) {

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

TEST(Series, to_map) {

    Series z = Series({10, 20, 30, 40}, {1, 2, 3, 4});

    int i = 0;
    for (auto &pair : z.to_map()) {
        auto key = pair.first;
        auto value = pair.second;

        EXPECT_EQ(key, z.index()[i]);
        EXPECT_EQ(value, z.values()[i]);
        i++;
    }

    EXPECT_TRUE(Series().to_map().empty());
}

TEST(Series, head) {

    Series z = Series({10, 20, 30, 40, 50, 60}, {1, 2, 3, 4, 5, 6});

    EXPECT_PRED2(
            Series::equal,
            z.head(),
            Series({10, 20, 30, 40, 50}, {1, 2, 3, 4, 5})
    );

    EXPECT_PRED2(
            Series::equal,
            z.head(3),
            Series({10, 20, 30}, {1, 2, 3,})
    );

    EXPECT_PRED2(
            Series::equal,
            z.head(0),
            Series()
    );

    EXPECT_PRED2(
            Series::equal,
            z.head(10),
            z
    );

}

TEST(Series, tail) {

    Series z = Series({10, 20, 30, 40, 50, 60}, {1, 2, 3, 4, 5, 6});

    EXPECT_PRED2(
            Series::equal,
            z.tail(),
            Series({20, 30, 40, 50, 60}, {2, 3, 4, 5, 6})
    );

    EXPECT_PRED2(
            Series::equal,
            z.tail(3),
            Series({40, 50, 60}, {4, 5, 6})
    );

    EXPECT_PRED2(
            Series::equal,
            z.tail(0),
            Series()
    );

    EXPECT_PRED2(
            Series::equal,
            z.tail(10),
            z
    );
}

} // namespace SeriesTests
