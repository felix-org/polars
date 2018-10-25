//
// Created by Linda Uruchurtu on 03/10/2018.
//

#include "polars/TimeSeriesMask.h"

#include "polars/numc.h"

#include "gtest/gtest.h"

#include <sstream>
#include <string>


using namespace std::chrono;
using namespace polars;

TEST(TimeSeriesMask, constructor) {

    using TimePoint = time_point<system_clock, milliseconds>;

    time_t t1 = 1525971600;  // 10th May 2018 6pm BST
    time_t t2 = 1525971780;

    TimePoint t1_p{duration_cast<milliseconds>(polars::unix_epoch_seconds(t1))};
    TimePoint t2_p{duration_cast<milliseconds>(polars::unix_epoch_seconds(t2))};

    std::vector<TimePoint> tpoints = {t1_p, t2_p};
    arma::uvec vals = {0, 1};

    EXPECT_PRED2(
            polars::MillisecondsTimeSeriesMask::equal,
            polars::MillisecondsTimeSeriesMask(),
            polars::MillisecondsTimeSeriesMask()
    ) << "Expect " << "empty TimeSeriesMask' match";

    auto ts = polars::MillisecondsTimeSeriesMask(vals, tpoints);

    EXPECT_PRED2(
            polars::numc::equal,
            ts.values(),
            arma::uvec({0, 1})
    ) << "Expect " << "retrieves values of timeseries";

    EXPECT_PRED2(
            polars::numc::equal_handling_nans,
            ts.index(),
            arma::vec({1525971600000, 1525971780000}) // in milliseconds
    ) << "Expect " << " timestamps in milliseconds";
}

TEST(TimeSeriesMask, from_series_mask) {
    using TP = time_point<system_clock, seconds>;

    EXPECT_PRED2(SecondsTimeSeriesMask::equal, SecondsTimeSeriesMask::from_series_mask({}), SecondsTimeSeriesMask());

    auto ts = SecondsTimeSeriesMask({3, 4}, {TP(1s), TP(2s)});
    EXPECT_PRED2(SecondsTimeSeriesMask::equal, SecondsTimeSeriesMask::from_series_mask(SeriesMask(ts)), ts)
                        << "Expect perfect round trip conversions when comparing as TimeSeriesMask";

    auto s = SeriesMask({3, 4}, {1, 2});
    EXPECT_PRED2(SeriesMask::equal, SeriesMask(SecondsTimeSeriesMask::from_series_mask(s)), s)
                        << "Expect perfect round trip conversions when comparing as SeriesMask";
}

TEST(TimeSeriesMask, get_timestamps) {

    using TimePoint = time_point<system_clock, seconds>;

    time_t t = 1525971600;
    TimePoint t_p{duration_cast<seconds>(polars::unix_epoch_seconds(t))};

    time_t t2 = 1525971780;
    TimePoint t2_p{duration_cast<seconds>(polars::unix_epoch_seconds(t2))};

    std::vector<TimePoint> tpoints = {t_p, t2_p};
    arma::vec timestamps = {(double) t, (double) t2};

    // Build TimeSeriesMask
    polars::SecondsTimeSeriesMask ts = polars::SecondsTimeSeriesMask(arma::uvec{1, 2}, tpoints);
    std::vector<TimePoint> tstamps = ts.timestamps();

    for (int i = 0; i < tstamps.size(); i++) {
        EXPECT_EQ(tstamps[i].time_since_epoch().count(), timestamps[i]);
    }

    // Case in which we pass an empty timeseries
    polars::SecondsTimeSeriesMask ts_empty = polars::SecondsTimeSeriesMask();
    std::vector<TimePoint> tstamps_empty = ts_empty.timestamps();

    EXPECT_TRUE(tstamps_empty.empty()) << "Expect " << " true since timeseries is empty";
};

TEST(TimeSeriesMask, to_timeseries_map) {

    using TimePoint = time_point<system_clock, seconds>;

    time_t t = 1525971600;
    TimePoint t_p{duration_cast<seconds>(polars::unix_epoch_seconds(t))};

    time_t t2 = 1525971780;
    TimePoint t2_p{duration_cast<seconds>(polars::unix_epoch_seconds(t2))};

    std::vector<TimePoint> tpoints = {t_p, t2_p};
    arma::uvec vals = {1, 0};

    // Build TimeSeriesMask
    polars::SecondsTimeSeriesMask ts = polars::SecondsTimeSeriesMask(vals, tpoints);

    std::map<TimePoint, bool> ts_map = ts.to_timeseries_map();

    int i = 0;
    for (auto &pair : ts_map) {
        auto key = pair.first;
        auto value = pair.second;

        EXPECT_EQ(key.time_since_epoch().count(), (double) ts.index()[i]);
        EXPECT_EQ(value, ts.values()[i]);
        i++;
    }

    // Case in which we pass an empty timeseries
    polars::SecondsTimeSeriesMask ts_empty = polars::SecondsTimeSeriesMask();
    std::map<TimePoint, bool> ts_empty_map = ts_empty.to_timeseries_map();

    EXPECT_TRUE(ts_empty_map.empty()) << "Expect " << " true since map is empty";
};


TEST(TimeSeriesMask, loc) {

    using TimePoint = time_point<system_clock, minutes>;

    time_t t1 = 1525971600;
    time_t t2 = 1525971780;
    time_t t3 = 1525971960;

    TimePoint t1_p{duration_cast<minutes>(polars::unix_epoch_seconds(t1))};
    TimePoint t2_p{duration_cast<minutes>(polars::unix_epoch_seconds(t2))};
    TimePoint t3_p{duration_cast<minutes>(polars::unix_epoch_seconds(t3))};

    std::vector<TimePoint> tpoints = {t1_p, t2_p, t3_p};
    std::vector<TimePoint> index_labels = {t2_p, t3_p};

    arma::uvec vals = {1, 0, 1};

    // Build TimeSeriesMask
    polars::MinutesTimeSeriesMask ts = polars::MinutesTimeSeriesMask(vals, tpoints);

    // Index Labels
    polars::MinutesTimeSeriesMask ts_subset = ts.loc(index_labels);

    EXPECT_PRED2(
            polars::numc::equal,
            ts_subset.values(),
            arma::uvec({0, 1})
    ) << "Expect " << "subset of values corresponding to provided indices";

    // Case in which labels are not in the ts
    std::vector<TimePoint> index_labels_empty = {};
    polars::MinutesTimeSeriesMask ts_empty = ts.loc(index_labels_empty);

    EXPECT_TRUE(ts_empty.empty()) << "Expect " << " true since timeseries is empty";
}

TEST(TimeSeriesMask, prettyprint) {

    // TODO: Add test for larger timeseries.

    std::stringstream out;
    using TimePoint = time_point<system_clock, seconds>;

    time_t t = 1525971600;
    TimePoint t_p{duration_cast<seconds>(polars::unix_epoch_seconds(t))};

    time_t t2 = 1525971780;
    TimePoint t2_p{duration_cast<seconds>(polars::unix_epoch_seconds(t2))};

    std::vector<TimePoint> tpoints = {t_p, t2_p};
    arma::uvec vals = {1, 0};

    // Build TimeSeriesMask
    polars::SecondsTimeSeriesMask ts = polars::SecondsTimeSeriesMask(vals, tpoints);

    // Check output
    out << ts;

    EXPECT_EQ(out.str(),
              "TimeSeriesMask: \nTimestamp:\n2018 May 10 17:00:00 Value:\n1Timestamp:\n2018 May 10 17:03:00 Value:\n0");

}

TEST(TimeSeriesMask, head_and_tail) {
    // Method work as expected
    using TimePoint = time_point<system_clock, seconds>;

    time_t t1 = 1525971600;
    time_t t2 = 1525971780;
    time_t t3 = 1525971960;

    TimePoint t1_p{duration_cast<minutes>(polars::unix_epoch_seconds(t1))};
    TimePoint t2_p{duration_cast<minutes>(polars::unix_epoch_seconds(t2))};
    TimePoint t3_p{duration_cast<minutes>(polars::unix_epoch_seconds(t3))};

    std::vector<TimePoint> tpoints = {t1_p, t2_p, t3_p};
    arma::uvec vals = {1, 0, 1};

    // Build TimeSeriesMask
    polars::SecondsTimeSeriesMask ts = polars::SecondsTimeSeriesMask(vals, tpoints);

    EXPECT_PRED2(
            polars::TimeSeriesMask<TimePoint>::equal,
            ts.head(2),
            polars::SecondsTimeSeriesMask({1, 0}, {t1_p, t2_p})
    );

    EXPECT_PRED2(
            polars::TimeSeriesMask<TimePoint>::equal,
            ts.tail(2),
            polars::SecondsTimeSeriesMask({0, 1}, {t2_p, t3_p})
    );

    EXPECT_PRED2(polars::TimeSeriesMask<TimePoint>::equal, ts.head(10), ts);

}

TEST(TimeSeriesMask, left_shift_operator_test) {
    // Method work as expected
    using TimePoint = time_point<system_clock, seconds>;

    time_t t1 = 1525971600;
    time_t t2 = 1525971780;
    time_t t3 = 1525971960;

    TimePoint t1_p{duration_cast<minutes>(polars::unix_epoch_seconds(t1))};
    TimePoint t2_p{duration_cast<minutes>(polars::unix_epoch_seconds(t2))};
    TimePoint t3_p{duration_cast<minutes>(polars::unix_epoch_seconds(t3))};

    std::vector<TimePoint> tpoints = {t1_p, t2_p, t3_p};
    arma::uvec vals = {1, 0, 1};

    // Build TimeSeriesMask
    polars::SecondsTimeSeriesMask ts = polars::SecondsTimeSeriesMask(vals, tpoints);

    std::ostringstream ss;
    ss << ts;
    ASSERT_EQ(
            ss.str(),
            "TimeSeriesMask: \nTimestamp:\n2018 May 10 17:00:00 Value:\n1Timestamp:\n2018 May 10 17:03:00 Value:\n0Timestamp:\n2018 May 10 17:06:00 Value:\n1"
    );
}
