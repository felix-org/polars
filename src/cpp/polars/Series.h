//
// Created by Calvin Giles on 16/03/2018.
//

#ifndef ZIMMER_SERIES_H
#define ZIMMER_SERIES_H

#include "WindowProcessor.h"

#include "armadillo"

#include <cassert>
#include <cmath>
#include <vector>
#include <map>


namespace polars {
    class SeriesMask;


    class Series {
    public:
        typedef arma::uword SeriesSize;

        Series();

        Series(const arma::vec &v, const arma::vec &t);

        Series(const SeriesMask &sm);

        static Series from_vect(const std::vector<double> &t_v, const std::vector<double> &v_v);

        static Series from_map(const std::map<double, double> &iv_map);

        SeriesMask operator==(const int rhs) const;

        SeriesMask operator!=(const int rhs) const;

        Series operator+(const double &rhs) const;

        Series operator-(const double &rhs) const;

        Series operator*(const double &rhs) const;

        SeriesMask operator>(const double &rhs) const;

        SeriesMask operator>=(const double &rhs) const;

        SeriesMask operator<=(const double &rhs) const;

        SeriesMask operator==(const Series &rhs) const;  // TODO test for floating point stability

        SeriesMask operator!=(const Series &rhs) const;  // TODO test for floating point stability

        SeriesMask operator>(const Series &rhs) const;

        SeriesMask operator<(const Series &rhs) const;

        Series operator+(const Series &rhs) const;

        Series operator-(const Series &rhs) const;

        Series operator*(const Series &rhs) const;

        bool equals(const Series &rhs) const;

        bool almost_equals(const Series &rhs) const;

        Series iloc(const arma::uvec &pos) const;

        double iloc(arma::uword pos) const;

        Series iloc(int from, int to, int step = 1) const;

        Series loc(const arma::vec &index_labels) const;

        Series loc(arma::uword) const;

        Series where(const SeriesMask &condition, double other = NAN) const;

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
                       SeriesSize minPeriods, /* 0 treated as windowSize */
                       bool center,
                       bool symmetric,
                       polars::WindowProcessor::WindowType win_type,
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

        Series head(int n=5) const;

        Series tail(int n=5) const;

    protected:
        arma::vec t;
        arma::vec v;
    };

    std::ostream &operator<<(std::ostream &os, const Series &ts);


    polars::Series _window_size_correction(int window_size, bool center, const polars::Series &input);
    polars::Series _ewm_input_correction(const polars::Series &input);
}


#endif //ZIMMER_SERIES_H
