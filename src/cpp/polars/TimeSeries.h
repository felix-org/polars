//
// Created by Linda Uruchurtu on 03/10/2018.
//

#ifndef POLARS_TIMESERIES_H
#define POLARS_TIMESERIES_H

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include <chrono>
#include "armadillo"
#include "date/date.h"

#include "Series.h"


namespace polars {

    using namespace std::chrono;
    typedef std::chrono::duration<double> unix_epoch_seconds;

    template<class TimePointType>
    class TimeSeries : public Series {

    public:
        // TODO:: Expose methods as required
        using Series::loc;
        using Series::head;
        using Series::tail;

        TimeSeries() = default;

        TimeSeries(arma::vec v0, std::vector<TimePointType> t0) : Series(v0, chrono_to_double_vector(t0)) {};

        std::vector<TimePointType> timestamps() const {
            // Pass indices and return vector of timepoints
            return double_to_chrono_vector(index());
        };

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
                return {};
            } else {
                arma::uvec indices_v = arma::conv_to<arma::uvec>::from(indices);
                return {values().elem(indices_v), double_to_chrono_vector(index().elem(indices_v))};
            }
        };

        TimeSeries head(int n) const  {
            Series ser(values(), index());
            Series ser_head = ser.head(n);
            return {ser_head.values(), double_to_chrono_vector(ser_head.index())};
        };

        TimeSeries tail(int n) const  {
            Series ser(values(), index());
            Series ser_tail = ser.tail(n);
            return {ser_tail.values(), double_to_chrono_vector(ser_tail.index())};
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
