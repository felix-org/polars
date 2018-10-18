//
// Created by Linda Uruchurtu on 03/10/2018.
//

#ifndef POLARS_TIMESERIES_H
#define POLARS_TIMESERIES_H

#include "Series.h"

#include "TimeSeriesMask.h"

#include "armadillo"
#include "date/date.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <string>
#include <vector>


namespace polars {

    using namespace std::chrono;
    typedef std::chrono::duration<double> unix_epoch_seconds;

    template<class TimePointType>
    class TimeSeries : public Series {
    using Mask = TimeSeriesMask<TimePointType>;
    public:
        // TODO:: Expose methods as required
        using Series::loc;
        using Series::head;
        using Series::tail;

        TimeSeries() = default;

        TimeSeries(arma::vec v0, std::vector<TimePointType> t0) : Series(v0, chrono_to_double_vector(t0)) {};

        /**
         * Converting constructor - this takes a TimeSeriesMask and creates a TimeSeries from it.
         *
         * This is intentionally implicit (not marked explicit) so that a function expecting a TimeSeries can be passed
         * a TimeSeriesMask and it will be automatically converted since this is a loss-less process.
         */
        TimeSeries(const Mask &sm) : Series(arma::conv_to<arma::vec>::from(sm.values()), sm.index()) {}

        static TimeSeries from_map(const std::map<TimePointType, double> &iv_map) {
            arma::vec index(iv_map.size());
            arma::vec values(iv_map.size());
            int i = 0;
            for (auto& pair : iv_map) {
                index[i] = chrono_to_double(pair.first);
                values[i] = pair.second;
                ++i;
            }
            return {values, index};
        }

        // TODO: Rename to_map once we sort out base methods, etc.
        std::map<TimePointType, double> to_timeseries_map() const {
            std::map<TimePointType, double> m;

            std::vector<TimePointType> timepoints = double_to_chrono_vector(index());

            // Put pairs into map
            for(int i = 0; i < size(); i++) {
                m.insert(std::make_pair(timepoints[i], values()[i]));
            }

            return m;
        };

        // TODO:: Make this more efficient.
        TimeSeries loc(const std::vector<TimePointType> &index_labels) const {
            return Series::loc(chrono_to_double_vector(index_labels));
        };

        TimeSeries head(int n) const  {
            Series ser_head = Series::head(n);
            return {ser_head.values(), double_to_chrono_vector(ser_head.index())};
        };

        TimeSeries tail(int n) const  {
            Series ser_tail = Series::tail(n);
            return {ser_tail.values(), double_to_chrono_vector(ser_tail.index())};
        };

        std::vector<TimePointType> timestamps() const {
            // Pass indices and return vector of timepoints
            return double_to_chrono_vector(index());
        };

    private:
        TimeSeries(arma::vec v0, arma::vec t0) : Series(v0, t0) {};
        TimeSeries(const Series& ser) : Series(ser) {};

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

    typedef TimeSeries<time_point<system_clock, milliseconds>> MillisecondsTimeSeries;
    typedef TimeSeries<time_point<system_clock, seconds>> SecondsTimeSeries;
    typedef TimeSeries<time_point<system_clock, minutes>> MinutesTimeSeries;
    typedef TimeSeries<time_point<system_clock, hours>> HoursTimeSeries;
    typedef TimeSeries<time_point<system_clock, date::days>> DaysTimeSeries;
    typedef TimeSeries<date::local_time<milliseconds>> LocalMillisecondsTimeSeries;
    typedef TimeSeries<date::local_time<seconds>> LocalSecondsTimeSeries;
    typedef TimeSeries<date::local_time<minutes>> LocalMinutesTimeSeries;
    typedef TimeSeries<date::local_time<hours>> LocalHoursTimeSeries;
    typedef TimeSeries<date::local_time<date::days>> LocalDaysTimeSeries;

}

template<class TimePointType>
std::ostream &operator<<(std::ostream &os, const polars::TimeSeries<TimePointType> &ts) {

    std::vector<TimePointType> timestamps = ts.timestamps();
    arma::vec vals = ts.values();

    os << "Timeseries: \n";

    for (auto& pair : ts.head(5).to_timeseries_map()) {
        time_t elem = std::chrono::system_clock::to_time_t(pair.first);
        os << "Timestamp:\n" << std::put_time(std::gmtime(&elem), "%Y %b %d %H:%M:%S") << " Value:\n" << pair.second;
    }

    if(ts.size() > 5){
        os << "\n ... \n";

        for (auto& pair : ts.tail(5).to_timeseries_map()) {
            time_t elem = std::chrono::system_clock::to_time_t(pair.first);
            os << "Timestamp:\n" << std::put_time(std::gmtime(&elem), "%Y %b %d %H:%M:%S") << " Value:\n" << pair.second;
        }
    }

    return os;
}

#endif //POLARS_TIMESERIES_H
