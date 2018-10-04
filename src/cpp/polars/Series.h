//
// Created by Calvin Giles on 16/03/2018.
//

#ifndef ZIMMER_SERIES_H
#define ZIMMER_SERIES_H

#include <cassert>
#include <vector>
#include <cmath>
#include "armadillo"
#include "WindowProcessor.h"


namespace polars {
    class SeriesMask;


    class Series {
    public:
        typedef arma::uword SeriesSize;

        Series();

        Series(const arma::vec &v, const arma::vec &t);

        static Series from_vect(std::vector<double> &t_v, std::vector<double> &v_v);

        polars::SeriesMask operator==(const int rhs) const;

        polars::SeriesMask operator!=(const int rhs) const;

        Series operator+(const double &rhs) const;

        Series operator-(const double &rhs) const;

        Series operator*(const double &rhs) const;

        polars::SeriesMask operator>(const double &rhs) const;

        polars::SeriesMask operator>=(const double &rhs) const;

        polars::SeriesMask operator<=(const double &rhs) const;

        polars::SeriesMask operator==(const Series &rhs) const;  // TODO test for floating point stability

        polars::SeriesMask operator>(const Series &rhs) const;

        polars::SeriesMask operator<(const Series &rhs) const;

        Series operator+(const Series &rhs) const;

        Series operator-(const Series &rhs) const;

        Series operator*(const Series &rhs) const;

        bool equals(const Series &rhs) const;

        bool almost_equals(const Series &rhs) const;

        Series iloc(const arma::uvec &pos) const;

        double iloc(arma::uword pos) const;

        Series loc(const arma::vec &index_labels) const;

        Series loc(arma::uword) const;

        Series where(const polars::SeriesMask &condition, double other = NAN) const;

        Series diff() const;

        Series abs() const;

        double quantile(double q = 0.5) const;

        Series fillna(double value = 0.) const;

        Series dropna() const;

        Series clip(double lower_limit, double upper_limit) const;

        Series pow(double power) const;

        Series rolling(SeriesSize windowSize,
                       const WindowProcessor &processor,
                       SeriesSize minPeriods = 0, /* 0 treated as windowSize */
                       bool center = true,
                       bool symmetric = false,
                       WindowProcessor::WindowType win_type = WindowProcessor::WindowType::none,
                       double alpha = -1) const;

        Window rolling(SeriesSize windowSize,
                       SeriesSize minPeriods = 0, /* 0 treated as windowSize */
                       bool center = true,
                       bool symmetric = false,
                       polars::WindowProcessor::WindowType win_type = polars::WindowProcessor::WindowType::none,
                       double alpha = -1) const;

        Rolling rolling(SeriesSize windowSize,
                        SeriesSize minPeriods = 0, /* 0 treated as windowSize */
                        bool center = true,
                        bool symmetric = false) const;

        Series apply(double (*f)(double)) const;

        int count() const;

        double sum() const;

        double mean() const;

        double std(int ddof=1) const;

        SeriesSize size() const;

        arma::vec finiteValues() const;

        SeriesSize finiteSize() const;

        // done this way so default copy / assignment works.
        // todo; make copies of indices share memory as they are const?
        const arma::vec index() const;

        const arma::vec values() const;

        static bool equal(const Series &lhs, const Series &rhs);

        static bool almost_equal(const Series &lhs, const Series &rhs);

        static bool not_equal(const Series &lhs, const Series &rhs);

        Series index_as_series() const;

        std::map<double, double> to_map() const;

        bool empty() const;

        Series head(int rows=5) const;

    protected:
        arma::vec t;
        arma::vec v;
    };

    std::ostream &operator<<(std::ostream &os, const Series &ts);
}


#endif //ZIMMER_SERIES_H
