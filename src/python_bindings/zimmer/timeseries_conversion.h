//
// Created by Calvin Giles on 17/03/2018.
//

#ifndef ZIMMER_TIMESERIES_CONVERSION_H
#define ZIMMER_TIMESERIES_CONVERSION_H

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "../../cpp/zimmer/time_series.h"
#include "../../cpp/zimmer/filters.h"


namespace py = pybind11;


TimeSeries to_timeseries(py::array_t<double> timestamps, py::array_t<double> values);
tupvec from_timeseries_to_tuple(const TimeSeries& ts);

#endif //ZIMMER_TIMESERIES_CONVERSION_H
