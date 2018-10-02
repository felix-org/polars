//
// Created by Calvin Giles on 2018-10-02.
//

#ifndef POLARS_WINDOWPROCESSOR_H
#define POLARS_WINDOWPROCESSOR_H

#include "armadillo"



namespace polars {
    class Series;

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

    arma::vec calculate_window_weights(polars::WindowProcessor::WindowType win_type, arma::uword windowSize,
                                       double alpha = -1);

    arma::vec _ewm_correction(const arma::vec &results, const arma::vec &v0, polars::WindowProcessor::WindowType win_type);

}  // polars


#endif //POLARS_WINDOWPROCESSOR_H
