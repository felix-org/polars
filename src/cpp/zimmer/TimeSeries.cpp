//
// Created by Calvin Giles on 16/03/2018.
//

/**
 * TimeSeries
 *
 * This module defines a TimeSeries class that is inspired by a python pandas Series class.
 *
 *
 * Style guide when extending TimeSeries
 * =====================================
 * Where possible, this class will behave the same as a pandas Series with a DataTimeIndex would. The supported features
 * of this class will be a subset of what is available by pandas.Series, and the typing system of C++ will be used to
 * full effect here where it makes sense.
 *
 * TODO: Add a TimeSeriesMask to use as return type from operator== and similar
 * TODO: Add support indexing with a TimeSeriesMask like ts[mask]
 *
 */


#include <vector>

#include "armadillo"

#include "TimeSeries.h"


TimeSeries::TimeSeries() = default;


TimeSeries::TimeSeries(const arma::vec &t, const arma::vec &v) : timestamps(t), values(v) {};


bool TimeSeries::equals(const TimeSeries &rhs) const {
    if ((values.n_rows != rhs.values.n_rows)) return false;
    if ((values.n_cols != rhs.values.n_cols)) return false;
    if ((timestamps.n_rows != rhs.timestamps.n_rows)) return false;
    if ((timestamps.n_cols != rhs.timestamps.n_cols)) return false;
    if (any(values != rhs.values)) return false;
    if (any(timestamps != rhs.timestamps)) return false;
    return true;
}


bool equal(const TimeSeries &lhs, const TimeSeries &rhs) {
    return lhs.equals(rhs);
}


bool not_equal(const TimeSeries &lhs, const TimeSeries &rhs) {
    return !lhs.equals(rhs);
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