//
// Created by Calvin Giles on 16/03/2018.
//

#ifndef ZIMMER_TIME_SERIES_H
#define ZIMMER_TIME_SERIES_H

#include <vector>

#include "armadillo"

#include <cassert>

class TimeSeries;

bool AlmostEqual(double a, double b);
bool equal_handling_nans(const arma::vec &lhs, const arma::vec &rhs);
bool almost_equal_handling_nans(const arma::vec &lhs, const arma::vec &rhs);

namespace zimmer {

    template<typename T>
    arma::vec arange(T start, T stop, T step = 1);

    arma::vec triang(int M, bool sym = true);

    class WindowProcessor {
    public:
        virtual double processWindow(const TimeSeries &window) const = 0;

        virtual double defaultValue() const = 0;
    };

    class Quantile : public WindowProcessor {
    public:
        Quantile(double quantile);

        double processWindow(const TimeSeries &window) const;

        inline double defaultValue() const {
            return NAN;
        }

    private:
        double quantile;
    };


    class TimeSeriesMask;

    class Sum : public WindowProcessor {
    public:
        Sum() = default;

        double processWindow(const TimeSeries &window) const;

        inline double defaultValue() const {
            return NAN;
        }
    };

    class Count : public WindowProcessor {
    public:
        Count() = default;

        Count(double default_value);

        double processWindow(const TimeSeries &window) const;

        inline double defaultValue() const {
            return default_value;
        }

    private:
        double default_value = NAN;
    };

    class Mean : public WindowProcessor {
    public:
        Mean() = default;

        Mean(double default_value);

        double processWindow(const TimeSeries &window) const;

        inline double defaultValue() const {
            return default_value;
        }

    private:
        double default_value = NAN;
    };


}  // zimmer


class TimeSeries {
public:
    using TimeSeriesMask=zimmer::TimeSeriesMask;
    typedef arma::uword SeriesSize;

    TimeSeries();

    TimeSeries(const arma::vec &t, const arma::vec &v);

    zimmer::TimeSeriesMask operator==(const int rhs) const;

    zimmer::TimeSeriesMask operator!=(const int rhs) const;

    TimeSeries operator+(const double &rhs) const;

    TimeSeries operator-(const double &rhs) const;

    TimeSeries operator*(const double &rhs) const;

    zimmer::TimeSeriesMask operator>(const double &rhs) const;

    zimmer::TimeSeriesMask operator>=(const double &rhs) const;

    zimmer::TimeSeriesMask operator<=(const double &rhs) const;

    zimmer::TimeSeriesMask operator==(const TimeSeries &rhs) const;  // TODO test for floating point stability

    zimmer::TimeSeriesMask operator>(const TimeSeries &rhs) const;

    zimmer::TimeSeriesMask operator<(const TimeSeries &rhs) const;

    TimeSeries operator+(const TimeSeries &rhs) const;

    TimeSeries operator-(const TimeSeries &rhs) const;

    TimeSeries operator*(const TimeSeries &rhs) const;

    bool equals(const TimeSeries &rhs) const;

    bool almost_equals(const TimeSeries &rhs) const;

    TimeSeries where(const TimeSeriesMask &condition, double other=NAN) const;

    TimeSeries diff() const;

    TimeSeries abs() const;

    TimeSeries pow(double power) const;

    TimeSeries rolling(SeriesSize windowSize,
                       const zimmer::WindowProcessor &processor,
                       SeriesSize minPeriods = 0, /* 0 treated as windowSize */
                       bool center = true,
                       bool symmetric = false,
                       std::string win_type = "None") const;

    double mean() const;

    SeriesSize size() const;

    arma::vec finiteValues() const;

    SeriesSize finiteSize() const;

    // done this way so default copy / assignment works.
    // todo; make copies of timeseries share memory as they are const?
    const arma::vec timestamps() const;

    const arma::vec values() const;

    static bool equal(const TimeSeries &lhs, const TimeSeries &rhs);

    static bool almost_equal(const TimeSeries &lhs, const TimeSeries &rhs);

    static bool not_equal(const TimeSeries &lhs, const TimeSeries &rhs);

private:
    arma::vec t;
    arma::vec v;
};

namespace zimmer { typedef TimeSeries TimeSeries; }  // Aid moving TimeSeries inside zimmer namespace

std::ostream &operator<<(std::ostream &os, const TimeSeries &ts);

#endif //ZIMMER_TIME_SERIES_H
