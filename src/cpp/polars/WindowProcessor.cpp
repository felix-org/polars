//
// Created by Calvin Giles on 2018-10-02.
//

#include "Series.h"
#include "numc.h"
#include "WindowProcessor.h"

namespace polars {


    // check if a double is an integer value, e.g. 2.0 vs 2.1
    bool double_is_int(double v) {
        double intpart;
        return modf(v, &intpart) == 0;
    }


    Quantile::Quantile(double quantile) : quantile(quantile) {
        //assert(quantile >= 0);
        //assert(quantile <= 1);
    }


    double Quantile::processWindow(const Series &window, const arma::vec weights) const {

        arma::vec v;
        v = weights % window.values();
        v = sort(v.elem(arma::find_finite(v)));

        // note, this is based on how q works in python numpy percentile rather than the more usual quantile defn.
        double quantilePosition = quantile * ((double) v.size() - 1);

        if (double_is_int(quantilePosition)) {
            return v(quantilePosition);
        } else {
            // interpolate estimate
            arma::uword quantileIdx = floor(quantilePosition);
            double fraction = quantilePosition - quantileIdx;
            return v(quantileIdx) + (v(quantileIdx + 1) - v(quantileIdx)) * fraction;
        }
    }


    double polars::Sum::processWindow(const Series &window, const arma::vec weights) const {
        return polars::numc::sum_finite((weights % window.values()));
    }


    polars::Count::Count(double default_value) : default_value(default_value) {}


    double polars::Count::processWindow(const Series &window, const arma::vec weights) const {
        return window.finiteSize();
    }


    polars::Mean::Mean(double default_value) : default_value(default_value) {}


    double polars::Mean::processWindow(const Series &window, const arma::vec weights) const {
        // This method doesn't support exponential window so results will be faulty. Please use ExpMean instead.
        arma::vec weighted_values = window.values() % weights;
        return polars::numc::sum_finite(weighted_values) / arma::sum(weights);
    }

    double polars::ExpMean::processWindow(const Series &window, const arma::vec weights) const {
        // This ensures deals with NAs like pandas for the case ignore_na = False which is the default setting.
        arma::vec weights_for_sum = weights.elem(arma::find_finite(window.values()));
        arma::vec weighted_values = window.values() % weights;
        return polars::numc::sum_finite(weighted_values) / arma::sum(weights_for_sum);
    }

} // polars