//
// Created by Calvin Giles on 16/03/2018.
//

#ifndef ZIMMER_TIME_SERIES_H
#define ZIMMER_TIME_SERIES_H

#include <vector>

#include "armadillo"


class TimeSeries {
public:
    TimeSeries();
    TimeSeries(const arma::vec &t, const arma::vec &v);

    bool operator==(const TimeSeries &rhs) const;
    bool operator!=(const TimeSeries &rhs) const;

    const arma::vec timestamps;
    const arma::vec values;
};

std::ostream &operator<<(std::ostream &os, const TimeSeries &ts);


#endif //ZIMMER_TIME_SERIES_H
