//
// Created by Calvin Giles on 23/03/2018.
//

#include "polars/SeriesMask.h"

#include "polars/Series.h"

#include "gtest/gtest.h"


namespace SeriesMaskTests {
    using Series = polars::Series;
    TEST(SeriesMask, equal) {
        EXPECT_PRED2(
                polars::SeriesMask::equal,
                polars::SeriesMask(),
                polars::SeriesMask()
        ) << "Expect " << "empty SeriesMask' match";

        EXPECT_PRED2(
                polars::SeriesMask::equal,
                polars::SeriesMask({0, 1}, {1, 2}),
                polars::SeriesMask({0, 1}, {1, 2})
        ) << "Expect " << "simple SeriesMask match";
    }

    TEST(SeriesMask, operator__or) {
        EXPECT_PRED2(
                polars::SeriesMask::equal,
                polars::SeriesMask(),
                polars::SeriesMask() | polars::SeriesMask()
        ) << "Expect " << "empty SeriesMask' match";

        EXPECT_PRED2(
                polars::SeriesMask::equal,
                polars::SeriesMask({0, 1, 1, 1}, {1, 2, 3, 4}),
                polars::SeriesMask({0, 0, 1, 1}, {1, 2, 3, 4}) |
                polars::SeriesMask({0, 1, 0, 1}, {1, 2, 3, 4})
        ) << "Expect " << "logical OR";

    }

    TEST(SeriesMask, operator__and) {
        EXPECT_PRED2(
                polars::SeriesMask::equal,
                polars::SeriesMask(),
                polars::SeriesMask() & polars::SeriesMask()
        ) << "Expect " << "empty SeriesMask' match";

        EXPECT_PRED2(
                polars::SeriesMask::equal,
                polars::SeriesMask({0, 0, 0, 1}, {1, 2, 3, 4}),
                polars::SeriesMask({0, 0, 1, 1}, {1, 2, 3, 4}) &
                polars::SeriesMask({0, 1, 0, 1}, {1, 2, 3, 4})
        ) << "Expect " << "logical OR";

    }

    TEST(SeriesMask, operator__not) {
        EXPECT_PRED2(
                polars::SeriesMask::equal,
                polars::SeriesMask(),
                !polars::SeriesMask()
        ) << "Expect " << "empty SeriesMask' match";

        EXPECT_PRED2(
                polars::SeriesMask::equal,
                polars::SeriesMask({0, 1}, {1, 2}),
                !polars::SeriesMask({1, 0}, {1, 2})
        ) << "Expect " << "logical NOT";

    }

    TEST(SeriesMask, operator__logical_float){

      EXPECT_PRED2(polars::SeriesMask::equal,
                   polars::SeriesMask({1,0,1},{1,2,3}),
                   polars::SeriesMask({0,1,0},{1,2,3})  == 0
      ) <<  "Expect " << "logical ==";

      EXPECT_PRED2(polars::SeriesMask::equal,
                   polars::SeriesMask({0,1,0},{1,2,3}),
                   polars::SeriesMask({0,1,0},{1,2,3})  != 0
      ) << "Expect " << "logical !=";
    }

}  // SeriesMaskTests
