//
// Created by Linda Uruchurtu on 03/10/2018.
//

#include "polars/TimeSeries.h"

#include "polars/numc.h"

#include "gtest/gtest.h"

#include <sstream>
#include <string>


namespace TimeSeriesTests {
using namespace polars;
using namespace std::chrono;

TEST(TimeSeries, constructor) {

    using TimePoint = time_point<system_clock, milliseconds>;

    time_t t1 = 1525971600;  // 10th May 2018 6pm BST
    time_t t2 = 1525971780;

    TimePoint t1_p{duration_cast<milliseconds>(polars::unix_epoch_seconds(t1))};
    TimePoint t2_p{duration_cast<milliseconds>(polars::unix_epoch_seconds(t2))};

    std::vector<TimePoint> tpoints = {t1_p, t2_p};
    arma::vec vals = {0.1, 0.2};

    EXPECT_PRED2(
            polars::MillisecondsTimeSeries::equal,
            polars::MillisecondsTimeSeries(),
            polars::MillisecondsTimeSeries()
    ) << "Expect " << "empty TimeSeries' match";

    auto ts = polars::MillisecondsTimeSeries(vals, tpoints);

    EXPECT_PRED2(
            polars::numc::equal_handling_nans,
            ts.values(),
            arma::vec({0.1, 0.2})
    ) << "Expect " << "retrieves values of timeseries";

    EXPECT_PRED2(
            polars::numc::equal_handling_nans,
            ts.index(),
            arma::vec({1525971600000, 1525971780000}) // in milliseconds
    ) << "Expect " << " timestamps in milliseconds";

    // TODO deal with duplicate values in constructor one way or the other


    using SecTP = time_point<system_clock, seconds>;
    using HourTP = time_point<system_clock, hours>;

    EXPECT_PRED2(DaysTimeSeries::equal, DaysTimeSeries(DaysTimeSeriesMask()), DaysTimeSeries())
                        << "Expect constructing with empty SeriesMask is the same as empty constructor";
    EXPECT_PRED2(
            SecondsTimeSeries::equal,
            SecondsTimeSeries(SecondsTimeSeriesMask({true, false}, {SecTP(1s), SecTP(2s)})),
            SecondsTimeSeries({1, 0}, {SecTP(1s), SecTP(2s)})
    ) << "Expect true becomes 1 and false becomes 0";

    auto foo = [](HoursTimeSeries s) { return s; };
    EXPECT_PRED2(
            HoursTimeSeries::equal,
            foo(HoursTimeSeriesMask({true, false}, {HourTP(1h), HourTP(2h)})),
            HoursTimeSeries({1, 0}, {HourTP(1h), HourTP(2h)})
    ) << "Expect implicit conversion from SeriesMask to Series is possible so that you can pass a SeriesMask to a function expecting a Series";

}

TEST(TimeSeries, from_map) {
    using TP = time_point<system_clock, seconds>;

    EXPECT_PRED2(SecondsTimeSeries::equal, SecondsTimeSeries::from_map({}), SecondsTimeSeries());

    EXPECT_PRED2(
            SecondsTimeSeries::equal,
            SecondsTimeSeries::from_map({{TP(1s), 3},
                                         {TP(2s), 4}}),
            SecondsTimeSeries({3, 4}, {TP(1s), TP(2s)})
    );

    using MTP = time_point<system_clock, minutes>;

    auto ts = MinutesTimeSeries({3, 4}, {MTP(1min), MTP(2min)});
    EXPECT_PRED2(
            MinutesTimeSeries::equal,
            MinutesTimeSeries::from_map(ts.to_timeseries_map()),
            ts
    );



}

TEST(TimeSeries, from_series) {
    using TP = time_point<system_clock, seconds>;

    EXPECT_PRED2(SecondsTimeSeries::equal, SecondsTimeSeries::from_series({}), SecondsTimeSeries());

    auto ts = SecondsTimeSeries({3, 4}, {TP(1s), TP(2s)});
    EXPECT_PRED2(SecondsTimeSeries::equal, SecondsTimeSeries::from_series(Series(ts)), ts)
                        << "Expect perfect round trip conversions when comparing as TimeSeries";

    auto s = Series({3, 4}, {1, 2});
    EXPECT_PRED2(Series::equal, Series(SecondsTimeSeries::from_series(s)), s)
                        << "Expect perfect round trip conversions when comparing as Series";
}

TEST(TimeSeries, get_timestamps) {

    using TimePoint = time_point<system_clock, seconds>;

    time_t t = 1525971600;
    TimePoint t_p{duration_cast<seconds>(polars::unix_epoch_seconds(t))};

    time_t t2 = 1525971780;
    TimePoint t2_p{duration_cast<seconds>(polars::unix_epoch_seconds(t2))};

    std::vector<TimePoint> tpoints = {t_p, t2_p};
    arma::vec timestamps = {(double) t, (double) t2};

    // Build TimeSeries
    polars::SecondsTimeSeries ts = polars::SecondsTimeSeries(arma::vec{1, 2}, tpoints);
    std::vector<TimePoint> tstamps = ts.timestamps();

    for (int i = 0; i < tstamps.size(); i++) {
        EXPECT_EQ(tstamps[i].time_since_epoch().count(), timestamps[i]);
    }

    // Case in which we pass an empty timeseries
    polars::SecondsTimeSeries ts_empty = polars::SecondsTimeSeries();
    std::vector<TimePoint> tstamps_empty = ts_empty.timestamps();

    EXPECT_TRUE(tstamps_empty.empty()) << "Expect " << " true since timeseries is empty";
};

TEST(TimeSeries, to_timeseries_map) {

    using TimePoint = time_point<system_clock, seconds>;

    time_t t = 1525971600;
    TimePoint t_p{duration_cast<seconds>(polars::unix_epoch_seconds(t))};

    time_t t2 = 1525971780;
    TimePoint t2_p{duration_cast<seconds>(polars::unix_epoch_seconds(t2))};

    std::vector<TimePoint> tpoints = {t_p, t2_p};
    arma::vec vals = {1, 2};

    // Build TimeSeries
    polars::SecondsTimeSeries ts = polars::SecondsTimeSeries(vals, tpoints);

    std::map<TimePoint, double> ts_map = ts.to_timeseries_map();

    int i = 0;
    for (auto &pair : ts_map) {
        auto key = pair.first;
        auto value = pair.second;

        EXPECT_EQ(key.time_since_epoch().count(), (double) ts.index()[i]);
        EXPECT_EQ(value, ts.values()[i]);
        i++;
    }

    EXPECT_EQ(ts_map.count(t_p), 1) << "Expect to find the first time point in the map";
    EXPECT_EQ(ts_map.count(t2_p), 1) << "Expect to find the second time point in the map";

    // Case in which we pass an empty timeseries
    polars::SecondsTimeSeries ts_empty = polars::SecondsTimeSeries();
    std::map<TimePoint, double> ts_empty_map = ts_empty.to_timeseries_map();

    EXPECT_TRUE(ts_empty_map.empty()) << "Expect " << " true since map is empty";
};


TEST(TimeSeries, to_timeseries_map__minutes) {

    using TimePoint = time_point<system_clock, minutes>;

    TimePoint t_p{minutes(1525971600 / 60)};

    TimePoint t2_p{minutes(1525971780 / 60)};

    std::vector<TimePoint> tpoints = {t_p, t2_p};
    arma::vec vals = {1, 2};

    // Build TimeSeries
    polars::MinutesTimeSeries ts = polars::MinutesTimeSeries(vals, tpoints);

    std::map<TimePoint, double> ts_map = ts.to_timeseries_map();

    int i = 0;
    for (auto &pair : ts_map) {
        auto key = pair.first;
        auto value = pair.second;

        EXPECT_EQ(key.time_since_epoch().count(), (double) ts.index()[i]);
        EXPECT_EQ(value, ts.values()[i]);
        i++;
    }

    EXPECT_EQ(ts_map.count(t_p), 1) << "Expect to find the first time point in the map";
    EXPECT_EQ(ts_map.count(t2_p), 1) << "Expect to find the second time point in the map";

    // Case in which we pass an empty timeseries
    polars::MinutesTimeSeries ts_empty = polars::MinutesTimeSeries();
    std::map<TimePoint, double> ts_empty_map = ts_empty.to_timeseries_map();

    EXPECT_TRUE(ts_empty_map.empty()) << "Expect " << " true since map is empty";
};


TEST(TimeSeries, loc) {

    using TimePoint = time_point<system_clock, minutes>;

    time_t t1 = 1525971600;
    time_t t2 = 1525971780;
    time_t t3 = 1525971960;

    TimePoint t1_p{duration_cast<minutes>(polars::unix_epoch_seconds(t1))};
    TimePoint t2_p{duration_cast<minutes>(polars::unix_epoch_seconds(t2))};
    TimePoint t3_p{duration_cast<minutes>(polars::unix_epoch_seconds(t3))};

    std::vector<TimePoint> tpoints = {t1_p, t2_p, t3_p};
    std::vector<TimePoint> index_labels = {t2_p, t3_p};

    arma::vec vals = {1, 2, 3};

    // Build TimeSeries
    polars::MinutesTimeSeries ts = polars::MinutesTimeSeries(vals, tpoints);

    // Index Labels
    polars::MinutesTimeSeries ts_subset = ts.loc(index_labels);

    EXPECT_PRED2(
            polars::numc::equal_handling_nans,
            ts_subset.values(),
            arma::vec({2, 3})
    ) << "Expect " << "subset of values corresponding to provided indices";

    // Case in which labels are not in the ts
    std::vector<TimePoint> index_labels_empty = {};
    polars::MinutesTimeSeries ts_empty = ts.loc(index_labels_empty);

    EXPECT_TRUE(ts_empty.empty()) << "Expect " << " true since timeseries is empty";
}

TEST(TimeSeries, prettyprint) {

    // TODO: Add test for larger timeseries.

    std::stringstream out;
    using TimePoint = time_point<system_clock, seconds>;

    time_t t = 1525971600;
    TimePoint t_p{duration_cast<seconds>(polars::unix_epoch_seconds(t))};

    time_t t2 = 1525971780;
    TimePoint t2_p{duration_cast<seconds>(polars::unix_epoch_seconds(t2))};

    std::vector<TimePoint> tpoints = {t_p, t2_p};
    arma::vec vals = {1, 2};

    // Build TimeSeries
    polars::SecondsTimeSeries ts = polars::SecondsTimeSeries(vals, tpoints);

    // Check output
    out << ts;

    EXPECT_EQ(out.str(),
              "Timeseries: \nTimestamp:\n2018 May 10 17:00:00 Value:\n1Timestamp:\n2018 May 10 17:03:00 Value:\n2");

}

TEST(TimeSeries, head_and_tail) {
    // Method work as expected
    using TimePoint = time_point<system_clock, seconds>;

    time_t t1 = 1525971600;
    time_t t2 = 1525971780;
    time_t t3 = 1525971960;

    TimePoint t1_p{duration_cast<minutes>(polars::unix_epoch_seconds(t1))};
    TimePoint t2_p{duration_cast<minutes>(polars::unix_epoch_seconds(t2))};
    TimePoint t3_p{duration_cast<minutes>(polars::unix_epoch_seconds(t3))};

    std::vector<TimePoint> tpoints = {t1_p, t2_p, t3_p};
    arma::vec vals = {1, 2, 3};

    // Build TimeSeries
    polars::SecondsTimeSeries ts = polars::SecondsTimeSeries(vals, tpoints);

    EXPECT_PRED2(
            polars::TimeSeries<TimePoint>::equal,
            ts.head(2),
            polars::SecondsTimeSeries({1, 2}, {t1_p, t2_p})
    );

    EXPECT_PRED2(
            polars::TimeSeries<TimePoint>::equal,
            ts.tail(2),
            polars::SecondsTimeSeries({2, 3}, {t2_p, t3_p})
    );

    EXPECT_PRED2(polars::TimeSeries<TimePoint>::equal, ts.head(10), ts);

}

TEST(TimeSeries, left_shift_operator_test) {
    // Method work as expected
    using TimePoint = time_point<system_clock, seconds>;

    time_t t1 = 1525971600;
    time_t t2 = 1525971780;
    time_t t3 = 1525971960;

    TimePoint t1_p{duration_cast<minutes>(polars::unix_epoch_seconds(t1))};
    TimePoint t2_p{duration_cast<minutes>(polars::unix_epoch_seconds(t2))};
    TimePoint t3_p{duration_cast<minutes>(polars::unix_epoch_seconds(t3))};

    std::vector<TimePoint> tpoints = {t1_p, t2_p, t3_p};
    arma::vec vals = {1, 2, 3};

    // Build TimeSeries
    polars::SecondsTimeSeries ts = polars::SecondsTimeSeries(vals, tpoints);

    std::ostringstream ss;
    ss << ts;
    ASSERT_EQ(
            ss.str(),
            "Timeseries: \nTimestamp:\n2018 May 10 17:00:00 Value:\n1Timestamp:\n2018 May 10 17:03:00 Value:\n2Timestamp:\n2018 May 10 17:06:00 Value:\n3"
    );
}
} // namespace TimeSeriesTests
