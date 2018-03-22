//
// Created by Calvin Giles on 23/03/2018.
//

#include <cassert>

#include "TimeSeries.h"
#include "TimeSeriesMask.h"

namespace zimmer {

    TimeSeriesMask::TimeSeriesMask() = default;


    TimeSeriesMask::TimeSeriesMask(const arma::vec &t, const arma::uvec &v) : t(t), v(v) {
        assert(t.n_cols == 1 && v.n_cols == 1);
        assert(t.n_rows == v.n_rows);
        assert(!arma::any(v > 1));  // Np TimeSeriesMask values may be greater than 1
    };


    TimeSeriesMask TimeSeriesMask::operator|(const TimeSeriesMask &rhs) const {
        assert(!arma::any(timestamps() != rhs.timestamps()));  // Use not any != to handle empty array case
        return TimeSeriesMask(timestamps(), (values() + rhs.values()) > 0);
    }


    TimeSeriesMask TimeSeriesMask::operator&(const TimeSeriesMask &rhs) const {
        assert(!arma::any(timestamps() != rhs.timestamps()));  // Use not any != to handle empty array case
        return TimeSeriesMask(timestamps(), (values() + rhs.values()) == 2);
    }


    TimeSeriesMask TimeSeriesMask::operator!() const {
        return TimeSeriesMask(timestamps(), values() == 0);
    }


    bool TimeSeriesMask::equals(const TimeSeriesMask &rhs) const {
        if ((values().n_rows != rhs.values().n_rows)) return false;
        if ((timestamps().n_rows != rhs.timestamps().n_rows)) return false;

        if (any(values() != rhs.values())) return false;

        if (any(timestamps() != rhs.timestamps())) return false;
        return true;
    }


    TimeSeries TimeSeriesMask::to_time_series() const {
        return TimeSeries(timestamps(), arma::conv_to<arma::vec>::from(values()));
    }

    TimeSeriesMask::SeriesSize TimeSeriesMask::size() const {
        assert(timestamps().size() == values().size());
        return timestamps().size();
    }


    /**
     * Static method
     * @param lhs
     * @param rhs
     * @return
     */
    bool TimeSeriesMask::equal(const TimeSeriesMask &lhs, const TimeSeriesMask &rhs) {
        return lhs.equals(rhs);
    }

    //
    //bool not_equal(const TimeSeriesMask &lhs, const TimeSeriesMask &rhs) {
    //    return !lhs.equals(rhs);
    //}


    /**
     * Add support for pretty printing of a TimeSeriesMask object.
     * @param os the output stream that will be written to
     * @param ts the TimeSeriesMask instance to output
     * @return the ostream for further piping
     */
    std::ostream &operator<<(std::ostream &os, const TimeSeriesMask &ts) {
        return os << "TimeSeriesMask:\ntimestamps\n" << ts.timestamps() << "values\n" << ts.values();
    }


    const arma::vec TimeSeriesMask::timestamps() const { return t; }


    const arma::uvec TimeSeriesMask::values() const { return v; }
}
