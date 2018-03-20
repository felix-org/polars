//
// Created by Calvin Giles on 16/03/2018.
//

#include <vector>

#include "armadillo"

#include "time_series.h"


TimeSeries::TimeSeries() = default;


TimeSeries::TimeSeries(const arma::vec &t, const arma::vec &v) : timestamps(t), values(v) {};


bool TimeSeries::operator==(const TimeSeries &rhs) const {
    if ((values.n_rows != rhs.values.n_rows)) return false;
    if ((values.n_cols != rhs.values.n_cols)) return false;
    if ((timestamps.n_rows != rhs.timestamps.n_rows)) return false;
    if ((timestamps.n_cols != rhs.timestamps.n_cols)) return false;
    if (any(values != rhs.values)) return false;
    if (any(timestamps != rhs.timestamps)) return false;
    return true;
}


bool TimeSeries::operator!=(const TimeSeries &rhs) const {
    return !(*this == rhs);
}


/**
 * Add support for pretty printing of a TimeSeries object.
 * @param os the output stream that will be written to
 * @param ts the TimeSeries instance to output
 * @return the ostream for further piping
 */
std::ostream &operator<<(std::ostream &os, const TimeSeries &ts) {
    return os << "TimeSeries:\ntimestamps\n" << ts.timestamps << "values\n" << ts.values;
}