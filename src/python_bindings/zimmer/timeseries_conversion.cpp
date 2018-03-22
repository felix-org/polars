//
// Created by Calvin Giles on 17/03/2018.
//

#include "timeseries_conversion.h"


TimeSeries to_timeseries(py::array_t<double> timestamps, py::array_t<double> values) {
    // allocate std::vector (to pass to the C++ function)

    // allocate std::vector (to pass to the C++ function)
    stdvec array_vec_timestamps(timestamps.size());
    stdvec array_vec_values(values.size());

    // copy py::array -> std::vector
    std::memcpy(array_vec_timestamps.data(), timestamps.data(), timestamps.size() * sizeof(double));
    std::memcpy(array_vec_values.data(), values.data(), values.size() * sizeof(double));

    // Convert to Armadillo colvec
    arma::vec data_timestamps = arma::conv_to<arma::vec>::from(array_vec_timestamps);
    arma::vec data_values = arma::conv_to<arma::vec>::from(array_vec_values);

    // Build a timeseries object
    TimeSeries ts = TimeSeries(data_timestamps, data_values);

    return ts;
};


tupvec from_timeseries_to_tuple(const TimeSeries& ts) {

    stdvec tstmps = arma::conv_to<stdvec>::from(ts.timestamps());
    stdvec vals = arma::conv_to<stdvec>::from(ts.values());

    // TODO: Return proper numpy arrays.
    return std::make_tuple(tstmps, vals);

}
