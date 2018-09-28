//
// Created by Calvin Giles on 16/03/2018.
//

#ifndef ZIMMER_SERIES_H
#define ZIMMER_SERIES_H

#include <cassert>
#include <vector>
#include <cmath>
#include "armadillo"


class Series;


namespace zimmer {

    class WindowProcessor {
    public:

        enum class WindowType {
            none,
            triang,
            expn
        };

        virtual double processWindow(const Series &window, const arma::vec weights) const = 0;

        virtual double defaultValue() const = 0;


    };


    class Quantile : public WindowProcessor {
    public:
        Quantile(double quantile);

        double processWindow(const Series &window, const arma::vec weights = {}) const;

        inline double defaultValue() const {
            return NAN;
        }

    private:
        double quantile;
    };


    class SeriesMask;

    class Sum : public WindowProcessor {
    public:
        Sum() = default;

        double processWindow(const Series &window, const arma::vec weights = {}) const;

        inline double defaultValue() const {
            return NAN;
        }
    };

    class Count : public WindowProcessor {
    public:
        Count() = default;

        Count(double default_value);

        double processWindow(const Series &window, const arma::vec weights = {}) const;

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

        double processWindow(const Series &window, const arma::vec weights = {}) const;

        inline double defaultValue() const {
            return default_value;
        }

    private:
        double default_value = NAN;
    };

    class ExpMean : public WindowProcessor {
    public:
        ExpMean() = default;

        double processWindow(const Series &window, const arma::vec weights = {}) const;

        inline double defaultValue() const {
            return default_value;
        }

    private:
        double default_value = NAN;
    };

    arma::vec calculate_window_weights(zimmer::WindowProcessor::WindowType win_type, arma::uword windowSize,
                                       double alpha = -1);

    arma::vec _ewm_correction(const arma::vec &results, const arma::vec &v0, zimmer::WindowProcessor::WindowType win_type);

}  // zimmer


class Series {
public:
    using SeriesMask=zimmer::SeriesMask;
    typedef arma::uword SeriesSize;

    Series();

    Series(const arma::vec &v, const arma::vec &t);

    static Series from_vect(std::vector<double> &t_v, std::vector<double> &v_v);

    zimmer::SeriesMask operator==(const int rhs) const;

    zimmer::SeriesMask operator!=(const int rhs) const;

    Series operator+(const double &rhs) const;

    Series operator-(const double &rhs) const;

    Series operator*(const double &rhs) const;

    zimmer::SeriesMask operator>(const double &rhs) const;

    zimmer::SeriesMask operator>=(const double &rhs) const;

    zimmer::SeriesMask operator<=(const double &rhs) const;

    zimmer::SeriesMask operator==(const Series &rhs) const;  // TODO test for floating point stability

    zimmer::SeriesMask operator>(const Series &rhs) const;

    zimmer::SeriesMask operator<(const Series &rhs) const;

    Series operator+(const Series &rhs) const;

    Series operator-(const Series &rhs) const;

    Series operator*(const Series &rhs) const;

    bool equals(const Series &rhs) const;

    bool almost_equals(const Series &rhs) const;

    Series iloc(const arma::uvec &pos) const;

    double iloc(arma::uword pos) const;

    Series loc(const arma::vec &index_labels) const;

    Series loc(arma::uword) const;

    Series where(const SeriesMask &condition, double other=NAN) const;

    Series diff() const;

    Series abs() const;

    double quantile(double q=0.5) const;

    Series fillna(double value=0.) const;

    Series dropna() const;

    Series clip(double lower_limit, double upper_limit) const;

    Series pow(double power) const;

    Series rolling(SeriesSize windowSize,
                       const zimmer::WindowProcessor &processor,
                       SeriesSize minPeriods = 0, /* 0 treated as windowSize */
                       bool center = true,
                       bool symmetric = false,
                       zimmer::WindowProcessor::WindowType win_type = zimmer::WindowProcessor::WindowType::none,
                       double alpha = -1) const;

    Series apply(double (*f)(double)) const;

    double mean() const;

    SeriesSize size() const;

    arma::vec finiteValues() const;

    SeriesSize finiteSize() const;

    // done this way so default copy / assignment works.
    // todo; make copies of timeseries share memory as they are const?
    const arma::vec index() const;

    const arma::vec values() const;

    static bool equal(const Series &lhs, const Series &rhs);

    static bool almost_equal(const Series &lhs, const Series &rhs);

    static bool not_equal(const Series &lhs, const Series &rhs);

    Series index_as_series() const;

    std::map<double, double> to_map() const;

private:
    arma::vec t;
    arma::vec v;
};

namespace zimmer { typedef Series Series; }  // Aid moving Series inside zimmer namespace

std::ostream &operator<<(std::ostream &os, const Series &ts);

#endif //ZIMMER_SERIES_H
