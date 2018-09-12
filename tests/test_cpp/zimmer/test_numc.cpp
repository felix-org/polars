//
// Created by Linda Uruchurtu on 30/08/2018.
//

#include "gtest/gtest.h"
#include "zimmer/numc.h"


TEST(numc, arange) {
    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0,1,2}),
            zimmer::numc::arange(0,3)
    )  << "Expect " << " should give sequence from start to stop by 1";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0,2,4}),
            zimmer::numc::arange(0,6,2)
    )  << "Expect " << " should give sequence from start to stop by 2";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0}),
            zimmer::numc::arange(0,1)
    )  << "Expect " << " should return array with 0";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0., 2.5, 5., 7.5}),
            zimmer::numc::arange(0, 10, 2.5)
    )  << "Expect " << " should work for doubles";

}

TEST(numc, sum_finite){
    EXPECT_EQ(0, zimmer::numc::sum_finite({})) << "Expect " << " zero since empty array";
    EXPECT_EQ(arma::sum(arma::vec({1, 2, 3})), zimmer::numc::sum_finite(arma::vec({1, 2, 3})))
                        << "Expect " << " same result as sum because no NANs";
    EXPECT_EQ(3, zimmer::numc::sum_finite(arma::vec({1,2,NAN}))) << "Expect " << " sum of first two terms";
}

TEST(numc, triang){

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0.5, 1., 0.5}),
            zimmer::numc::triang(3)
    ) << "Expect " << " generate symmetric window of 3";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0.25, 0.75, 0.75}),
            zimmer::numc::triang(3, false)
    ) << "Expect " << " generate non-symmetric window of 3";

    EXPECT_PRED2(
            zimmer::numc::almost_equal_handling_nans,
            arma::vec({0.33333333, 0.66666667, 1., 0.66666667, 0.33333333}),
            zimmer::numc::triang(5)
    ) << "Expect " << " generate symmetric window of 5";

    EXPECT_PRED2(
            zimmer::numc::almost_equal_handling_nans,
            arma::vec({0.16666666666666666, 0.5000, 0.83333333333333337, 0.83333333333333337, 0.5000}),
            zimmer::numc::triang(5, false)
    ) << "Expect " << " generate non-symmetric window of 5";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({1}),
            zimmer::numc::triang(1)
    ) << "Expect " << " return array with 1";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({1}),
            zimmer::numc::triang(1, false)
    ) << "Expect " << " return array with 1";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({}),
            zimmer::numc::triang(0)
    ) << "Expect " << " return empty array";
}

TEST(numc, exponential){
    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({}),
            zimmer::numc::exponential(0)
    ) << "Expect " << " return empty array";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({1}),
            zimmer::numc::exponential(1)
    ) << "Expect " << " return array with 1";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0.36787944117144233, 0.60653065971263342, 1., 0.60653065971263342, 0.36787944117144233}),
            zimmer::numc::exponential(5, 2.0, false)
    ) << "Expect " << " return odd non-symmetric array";


   EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0.36787944117144233, 0.60653065971263342, 1., 0.60653065971263342}),
            zimmer::numc::exponential(4, 2.0, false)
    ) << "Expect " << " return even non-symmetric array";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0.60653065971263342, 0.84648172489061413, 0.84648172489061413, 0.60653065971263342}),
            zimmer::numc::exponential(4, 3.0)
    ) << "Expect " << " return even symmetric array";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0.51341711903259202, 0.71653131057378927, 1., 0.71653131057378927, 0.51341711903259202}),
            zimmer::numc::exponential(5, 3.0)
    ) << "Expect " << " return odd symmetric array";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0.71653131057378927, 1, 0.71653131057378927, 0.51341711903259202, 0.36787944117144233}),
            zimmer::numc::exponential(5, 3.0, false, 1)
    ) << "Expect " << " return odd non symmetric and center = 1 array";

    EXPECT_PRED2(
            zimmer::numc::equal_handling_nans,
            arma::vec({0.51341711903259202, 0.71653131057378927, 1., 0.71653131057378927, 0.51341711903259202}),
            zimmer::numc::exponential(5, 3.0, true, 1)
    ) << "Expect " << " symmetric centered array since center = 1 is override";
}
