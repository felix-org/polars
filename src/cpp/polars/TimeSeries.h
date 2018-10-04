//
// Created by Linda Uruchurtu on 03/10/2018.
//

#ifndef POLARS_TIMESERIES_H
#define POLARS_TIMESERIES_H

#include <cassert>
#include <vector>
#include <cmath>
#include <string>
#include "armadillo"
#include <chrono>
#include "date/date.h"

#include "Series.h"


namespace polars {

    using namespace std::chrono;
    typedef std::chrono::duration<double> unix_epoch_seconds;

    template<class TimePointType>
    class TimeSeries : public Series {

    public:
        using Series::loc;

        TimeSeries() = default;

        TimeSeries(arma::vec v0, std::vector<TimePointType> t0){
            v = v0;
            t = chrono_to_double_vector(t0);
        };

        std::vector<TimePointType> timestamps() const {
            // Pass indices and return vector of timepoints
            return double_to_chrono_vector(index());
        };

        std::map<TimePointType, double> to_timeseries_map() const {

            std::map<TimePointType, double> m;

            std::vector<TimePointType> timepoints = double_to_chrono_vector(index());

            // Put pairs into map
            for(int i = 0; i < size(); i++) {
                m.insert(std::make_pair(timepoints[i], values()[i]));
            }

            return m;
        };

        TimeSeries loc(const std::vector<TimePointType> &index_labels) const {
            std::vector<double> indices;

            // Turn time_points to doubles
            auto tstamps = chrono_to_double_vector(index_labels);

            for (int j = 0; j < tstamps.n_elem; j++) {

                arma::uvec idx = arma::find(index() == tstamps[j]);

                if (!idx.empty()) {
                    indices.push_back(idx[0]);
                }
            }

            if (indices.empty()) {
                return TimeSeries();
            } else {
                arma::uvec indices_v = arma::conv_to<arma::uvec>::from(indices);
                return TimeSeries(values().elem(indices_v), double_to_chrono_vector(index().elem(indices_v)));
            }
        };

    private:
        static double chrono_to_double(TimePointType timepoint){
            return time_point_cast<typename TimePointType::duration>(timepoint).time_since_epoch().count();
        };

        static arma::vec chrono_to_double_vector(const std::vector<TimePointType>& timepoints){

            arma::vec tstamps(timepoints.size());
            for(int i = 0; i < timepoints.size() ; i++){
                tstamps[i] = chrono_to_double(timepoints[i]);
            }
            return tstamps;
        };


        static TimePointType double_to_chrono(double timestamp){
            return TimePointType{duration_cast<typename TimePointType::duration>(unix_epoch_seconds(timestamp))};
        };

        static std::vector<TimePointType> double_to_chrono_vector(const arma::vec& tstamps){

            std::vector<TimePointType> timepoints;

            for(const auto &value:tstamps){
                timepoints.push_back(double_to_chrono(value));
            }
            return timepoints;
        };

    };

    typedef std::chrono::duration<float, std::ratio<24*60*60, 1>> days;

    typedef TimeSeries<time_point<system_clock, milliseconds>> MillisecondsTimeSeries;
    typedef TimeSeries<time_point<system_clock, seconds>> SecondsTimeSeries;
    typedef TimeSeries<time_point<system_clock, minutes>> MinutesTimeSeries;
    typedef TimeSeries<time_point<system_clock, hours>> HoursTimeSeries;
    typedef TimeSeries<time_point<system_clock, days>> DaysTimeSeries;
    typedef TimeSeries<date::local_time<milliseconds>> LocalMillisecondsTimeSeries;
    typedef TimeSeries<date::local_time<seconds>> LocalSecondsTimeSeries;
    typedef TimeSeries<date::local_time<minutes>> LocalMinutesTimeSeries;
    typedef TimeSeries<date::local_time<hours>> LocalHoursTimeSeries;
    typedef TimeSeries<date::local_time<days>> LocalDaysTimeSeries;

}

template<class TimePointType>
std::ostream &operator<<(std::ostream &os, const polars::TimeSeries<TimePointType> &ts) {

    std::vector<TimePointType> timestamps = ts.timestamps();
    arma::vec vals = ts.values();

    os << "Timeseries: \n";

    for(int i = 0; i < timestamps.size() ; i++){
        time_t elem = std::chrono::system_clock::to_time_t(timestamps[i]);
        double val = vals[i];
        os << "Timestamp:\n" << std::put_time(std::gmtime(&elem), "%Y %b %d %H:%M:%S") << " Value:\n" << val;
    }

    return os;
}


#endif //POLARS_TIMESERIES_H
