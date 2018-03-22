//
// Created by Calvin Giles on 23/03/2018.
//

#ifndef ZIMMER_TIMESERIESMASK_H
#define ZIMMER_TIMESERIESMASK_H

#include "armadillo"

class TimeSeries;
namespace zimmer {
    // TODO combine TimeSeries and TimeSeriesMask via a BaseTimeSeries
    // TODO Create IntTimeSeries and EnumTimeSeries
    // TODO rename TimeSeries -> DoubleTimeSeries and TimeSeriesMask -> BoolTimeSeries

    class TimeSeriesMask {
    public:
        typedef arma::uword SeriesSize;
        TimeSeriesMask();

        TimeSeriesMask(const arma::vec &t, const arma::uvec &v);

        TimeSeriesMask operator|(const TimeSeriesMask &rhs) const;
        TimeSeriesMask operator&(const TimeSeriesMask &rhs) const;
        TimeSeriesMask operator!() const;

        bool equals(const TimeSeriesMask &rhs) const;

        TimeSeries to_time_series() const;

        SeriesSize size() const;

        static bool equal(const TimeSeriesMask &lhs, const TimeSeriesMask &rhs);

        // done this way so default copy / assignment works.
        // todo; make copies of timeseries share memory as they are const?
        const arma::vec timestamps() const;
        const arma::uvec values() const;

    private:
        arma::vec t;
        arma::uvec v;
    };


    //bool not_equal(const TimeSeriesMask &lhs, const TimeSeriesMask &rhs);
    std::ostream &operator<<(std::ostream &os, const TimeSeriesMask &ts);

}
#endif //ZIMMER_TIMESERIESMASK_H
