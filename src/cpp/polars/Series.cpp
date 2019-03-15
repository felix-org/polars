//
// Created by Calvin Giles on 16/03/2018.
//

/**
 * Series
 *
 * This module defines a Series class that is inspired by a python pandas Series class.
 *
 *
 * Style guide when extending Series
 * =====================================
 * Where possible, this class will behave the same as a pandas Series. The supported features
 * of this class will be a subset of what is available by pandas.Series, and the typing system of C++ will be used to
 * full effect here where it makes sense.
 *
 */

#include "Series.h"

#include "SeriesMask.h"
#include "numc.h"


namespace polars {


    using SeriesMask = polars::SeriesMask;

    Series::Series() = default;


// todo; check for 1-D series & that the lengths match
    Series::Series(const arma::vec &v, const arma::vec &t) : t(t), v(v) {
        //assert(t.n_cols == 1 && v.n_cols == 1);
        //assert(t.n_rows == v.n_rows);
    };

    /**
     * Converting constructor - this takes a SeriesMask and creates a Series from it.
     *
     * This is intentionally implicit (not marked explicit) so that a function expecting a Series can be passed a
     * SeriesMask and it will be automatically converted since this is a loss-less process.
     */
    Series::Series(const SeriesMask &sm) : t(sm.index()), v(arma::conv_to<arma::vec>::from(sm.values())) {}

    Series Series::from_vect(const std::vector<double> &t_v, const std::vector<double> &v_v) {
        return Series(arma::conv_to<arma::vec>::from(v_v), arma::conv_to<arma::vec>::from(t_v));
    }

    Series Series::from_map(const std::map<double, double> &iv_map) {
        arma::vec index(iv_map.size());
        arma::vec values(iv_map.size());
        int i = 0;
        for (auto& pair : iv_map) {
            index[i] = pair.first;
            values[i] = pair.second;
            ++i;
        }
        return {values, index};
    }


// Series [op] int methods
    SeriesMask Series::operator==(const int rhs) const {
        arma::vec rhs_vec = arma::ones(this->size()) * rhs;
        arma::vec abs_diff = arma::abs(values() - rhs_vec);
        // We can't use a large difference test like .1 despite the rhs is an int since the lhs is double so could be close.
        double threshold = 1E-50;
        return SeriesMask(abs_diff < threshold, index());
    }


    SeriesMask Series::operator!=(const int rhs) const {  // TODO implement as negation of operator==
        arma::vec rhs_vec = arma::ones(this->size()) * rhs;
        arma::vec abs_diff = arma::abs(values() - rhs_vec);
        // We can't use a large difference test like .1 despite the rhs is an int since the lhs is double so could be close.
        double threshold = 1E-50;
        return SeriesMask(abs_diff > threshold, index());
    }


    // Series [op] Series methods
    SeriesMask Series::operator==(const Series &rhs) const {
        // TODO: make this fast enough to always check at runtime
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return SeriesMask(values() == rhs.values(), index());
    }


    SeriesMask Series::operator!=(const Series &rhs) const {
        // TODO: make this fast enough to always check at runtime
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return SeriesMask(values() != rhs.values(), index());
    }


    SeriesMask Series::operator>(const Series &rhs) const {
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return SeriesMask(values() > rhs.values(), index());
    }


    SeriesMask Series::operator<(const Series &rhs) const {
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return polars::SeriesMask(values() < rhs.values(), index());
    }


    Series Series::operator+(const Series &rhs) const {
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return polars::Series(values() + rhs.values(), index());
    }


    Series Series::operator-(const Series &rhs) const {
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return polars::Series(values() - rhs.values(), index());
    }


    Series Series::operator*(const Series &rhs) const {
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return polars::Series(values() % rhs.values(), index());
    }


// Series [op] double methods
    SeriesMask Series::operator>(const double &rhs) const {
        return SeriesMask(values() > (arma::ones(size()) * rhs), index());
    }

    SeriesMask Series::operator>=(const double &rhs) const {
        return SeriesMask(values() >= (arma::ones(size()) * rhs), index());
    }

    SeriesMask Series::operator<=(const double &rhs) const {
        return SeriesMask(values() <= (arma::ones(size()) * rhs), index());
    }

    Series Series::operator+(const double &rhs) const {
        return Series(values() + rhs, index());
    }


    Series Series::operator-(const double &rhs) const {
        return Series(values() - rhs, index());
    }


    Series Series::operator*(const double &rhs) const {
        return Series(values() * rhs, index());
    }


// todo; do we need a flavor that *doesn't* take account of NANs?
    bool Series::equals(const Series &rhs) const {
        if ((index().n_rows != rhs.index().n_rows)) return false;
        if ((index().n_cols != rhs.index().n_cols)) return false;
        if (!polars::numc::equal_handling_nans(values(), rhs.values())) return false;
        if (any(index() != rhs.index())) return false;
        return true;
    }


// todo; do we need a flavor that *doesn't* take account of NANs?
    bool Series::almost_equals(const Series &rhs) const {
        if ((index().n_rows != rhs.index().n_rows)) return false;
        if ((index().n_cols != rhs.index().n_cols)) return false;
        if (!polars::numc::almost_equal_handling_nans(values(), rhs.values())) return false;
        if (any(index() != rhs.index())) return false;
        return true;
    }

// Location.

    Series Series::iloc(int from, int to, int step) const {

        if(empty() || (from == to)){
            return Series();
        }

        arma::uvec pos;
        int effective_from;
        int effective_to;

        if(from < 0){
            effective_from = values().size() + from;
        } else {
            effective_from = from;
        }

        if(to < 0){
            effective_to = values().size() + to - 1;
        } else if(to == 0) {
            effective_to = to;
        } else {
            effective_to = to - 1;
        }

        pos = arma::regspace<arma::uvec>(effective_from,  step,  effective_to);

        if(pos.size() > size()){
            pos = pos.subvec(0, size() - 1);
        }

        return Series(values().elem(pos),index().elem(pos));
    }

    // TODO: Add slicing logic of the form .iloc(int start, int stop, int step=1) so it can be called like ser.iloc(0, -10).
    Series Series::iloc(const arma::uvec &pos) const {
        return Series(values().elem(pos), index().elem(pos));
    }


    double Series::iloc(arma::uword pos) const {
        arma::vec val = values().elem(arma::uvec{pos});
        return val[0];
    }

// by label of indices
    Series Series::loc(const arma::vec &index_labels) const {

        std::vector<int> indices;

        for (int j = 0; j < index_labels.n_elem; j++) {

            arma::uvec idx = arma::find(index() == index_labels[j]);

            if (!idx.empty()) {
                indices.push_back(idx[0]);
            }
        }

        if (indices.empty()) {
            return Series();
        } else {
            arma::uvec indices_v = arma::conv_to<arma::uvec>::from(indices);
            return Series(values().elem(indices_v), index().elem(indices_v));
        }
    }

    Series Series::loc(arma::uword pos) const {
        arma::uvec idx = arma::find(index() == pos);

        if (!idx.empty()) {
            return Series(values(), index()).iloc(idx);
        } else {
            return Series();
        }
    }

    Series Series::where(const SeriesMask &condition, double other) const {
        arma::vec result = values();
        result.elem(find((!condition).values())).fill(other);
        return Series(result, index());
    }


    Series Series::diff() const {

        arma::uword resultSize = values().size();

        arma::vec resultv(resultSize);

        double previousValue = NAN;
        for (arma::uword idx = 0; idx < resultSize; idx++) {
            resultv[idx] = values()[idx] - previousValue;
            previousValue = values()[idx];
        }

        return Series(resultv, index());
    }


    Series Series::abs() const {
        return Series(arma::abs(values()), index());
    }

    double Series::quantile(double q) const {
        return polars::numc::quantile(values(), q);
    }

    Series Series::fillna(double value) const {
        arma::vec vals = values();
        vals.replace(arma::datum::nan, value);
        return Series(vals, index());
    }

    Series Series::dropna() const {
        // Get indices of finite elements
        arma::uvec indices = arma::sort(arma::join_cols(
                arma::find_finite(values()),
                arma::find(arma::abs(values()) == arma::datum::inf))
        );
        return Series(values().elem(indices), index().elem(indices));
    }


    arma::vec calculate_window_weights(
            polars::WindowProcessor::WindowType win_type,
            arma::uword windowSize,
            double alpha
    ) {
        switch (win_type) {
            case (polars::WindowProcessor::WindowType::none):
                return arma::ones(windowSize);
            case (polars::WindowProcessor::WindowType::triang):
                return polars::numc::triang(windowSize);
            case (polars::WindowProcessor::WindowType::expn):
                return reverse(polars::numc::exponential(windowSize, -1. / log(1 - alpha), false, 0));
            default:
                return arma::ones(windowSize);
        }
    }

    // TODO: This method needs to be re-factored.
    arma::vec _ewm_correction(const arma::vec &results, const arma::vec &vals,
                                      polars::WindowProcessor::WindowType win_type) {
        /* Method that shifts result from rolling average with exp window so it yields correct normalisation and allows usage
         * of rolling method hereby implemented.
         * This matches pandas ewm for its default case */

        arma::uvec res_fin = arma::find_finite(results);

        if (results.empty() || res_fin.empty()){
            arma::vec effective_results(vals.size());
            effective_results.fill(NAN);
            return effective_results;
        }

        if (win_type == polars::WindowProcessor::WindowType::expn) {
            // Correction to match pandas ewm - shift by one.
            arma::vec effective_results = results;
            arma::vec results_ewm;
            int missing_initial_nans =  0;

            // Check if it was a case of originally front NANs - no added padding
            arma::uvec finite_values_idx = arma::find_finite(vals);
            if(finite_values_idx[0] != 0){
                // we had originally some NANs.
                missing_initial_nans = finite_values_idx[0];
            }

            arma::vec effective_vals = vals.subvec(missing_initial_nans, vals.size() - 1);

            // For the cases in which window_size < size we need to correct
            if( (results.size() > effective_vals.size())){
                effective_results = effective_results.elem( arma::find_finite(effective_results) );
                effective_results = effective_results.head(effective_vals.size());
            }

            if ( effective_results.at(0) == effective_vals[0] ){
                // difference of 1 for center = False.
                results_ewm = effective_results;
            } else {
                results_ewm.copy_size(effective_results) ;
                results_ewm.at(0) = vals[0];

                for (int j = 1; j < results_ewm.n_elem; j++) {
                    results_ewm[j] = effective_results[j - 1];
                }
            }

            if(missing_initial_nans > 0){
                auto added_nans = vals.size() - results_ewm.size();
                arma::vec pn(added_nans);
                pn.fill(NAN);
                return arma::join_cols(pn, results_ewm);
            } else {
                return results_ewm;
            }

        } else {
            return results;
        }
    }

    // TODO: Combine cases here with those in input for exponential case.
    polars::Series _window_size_correction(int window_size, bool center, const polars::Series &input){
        // This is required because of relative size of window size vs array size.

        if(input.size() == 1){
            return input;
        }

        auto n = window_size / 2;
        if(input.size() % 2 == 0){ n = n - 1; };

        // Get index delta
        auto ts = input.index();
        double delta = std::ceil(std::abs(ts(1) -  ts(0)));

        // Vector with n nans
        arma::vec multi_nan(n);
        multi_nan.fill(NAN);

        // Vector with single nan
        arma::vec single_nan(1);
        single_nan.fill(NAN);

        // Base case is padding goes to the right
        arma::vec new_input = arma::join_cols(input.values(),  multi_nan);

        auto start_idx = ts(0);
        auto end_idx = ts(ts.size()-1) + n * delta;
        auto effective_size = n + input.size();

        if (center == false){

            if( window_size > input.size() + 1 ){
                new_input = arma::join_cols(multi_nan, input.values());
                start_idx = start_idx - n * delta;
                end_idx = end_idx - n * delta;
            } else if (n == 1) {
                new_input = arma::join_cols(single_nan, arma::join_cols(single_nan, input.values()));
                start_idx = start_idx - 2 * delta;
                end_idx = end_idx - delta;

                effective_size = effective_size + 1;
            } else if (n == 2) {
                new_input = arma::join_cols(single_nan, arma::join_cols(input.values(), single_nan));
                start_idx = start_idx - delta;
                end_idx = end_idx - delta;
            }
        }

        arma::vec new_timestamps = arma::linspace(start_idx, end_idx, effective_size);

        return Series(new_input, new_timestamps);
    }

    // TODO: Refactor this method and combine with window_size_correction since similar logic
    polars::Series _ewm_input_correction(const polars::Series &input){
        // only gets called when window_size < input.size() and we have win_type = expn
        Series new_input = input;

        if(input.empty() || input.dropna().empty()){
            return input;
        }

        // remove front NANs
        if(std::isnan(input.values()(0))){
            arma::uvec finite_input_idx = arma::find_finite(input.values());
            auto number_of_nans = finite_input_idx(0);
            new_input = input.iloc(number_of_nans, input.size());
        }

        arma::vec v(new_input.size());
        v.fill(NAN);
        arma::vec new_values = arma::join_cols(v, new_input.values());

        // Get new timestamps
        auto ts = new_input.index();
        auto delta = std::ceil(std::abs(input.index()(1) -  input.index()(0))); // use original input

        auto new_index_0 = ts(0) - new_input.size() * delta;
        arma::vec new_timestamps = arma::linspace(new_index_0, ts(ts.size()-1), 2 * new_input.size());

        return Series(new_values, new_timestamps);
    }


// todo; allow passing in transformation function rather than WindowProcessor.
    Series
    Series::rolling(SeriesSize windowSize, const polars::WindowProcessor &processor, SeriesSize minPeriods,
                    bool center, bool symmetric, polars::WindowProcessor::WindowType win_type, double alpha) const {

        //assert(center); // todo; implement center:false
        //assert(windowSize > 0);
        //assert(windowSize % 2 == 0); // TODO: Make symmetric = true work for even windows. See tests for reference.
        arma::vec input_values = v;
        arma::vec input_idx = t;

        if(win_type == polars::WindowProcessor::WindowType::expn){
            Series padded_input = _ewm_input_correction(*this);
            input_values = padded_input.values();
            input_idx = padded_input.index();

            // Set window size to be the same as the size of the array
            windowSize = this->size();
        } else if((size() > 0 && windowSize > size())) {

            // This deals with issues of alignment that arises with non linear windows.
            Series padded_input = _window_size_correction(windowSize, center, *this);
            input_values = padded_input.values();
            input_idx = padded_input.index();
        }

        if (minPeriods == 0) {
            minPeriods = windowSize;
        }

        arma::uword resultSize = input_values.size();
        arma::vec resultv(resultSize);

        arma::uword centerOffset = round(((float) windowSize - 1) / 2.0);

        // roll a window [left,right], of up to size windowSize, centered on centerIdx, and hand to processor if there are minPeriods finite values.
        for (arma::uword centerIdx = 0; centerIdx < input_idx.size(); centerIdx++) {

            arma::sword leftIdx = centerIdx - centerOffset;
            arma::sword rightIdx = centerIdx - centerOffset + windowSize - 1;
            arma::sword weightLeftIdx = 0;
            arma::sword weightRightIdx = windowSize - 1;

            if (symmetric) {
                // This option works for odd windows only.
                if (leftIdx < 0) {
                    arma::sword left_err = leftIdx;
                    arma::sword right_err = windowSize - 1 - centerIdx - centerOffset;

                    weightLeftIdx = weightLeftIdx - left_err;
                    weightRightIdx = weightRightIdx - right_err;

                    rightIdx = rightIdx - right_err;
                    leftIdx = leftIdx - left_err;
                }

                if (rightIdx >= input_idx.size()) {
                    arma::sword r_clipped = rightIdx - input_idx.size();

                    arma::sword left_err = -r_clipped - 1;
                    arma::sword right_err = rightIdx - (input_idx.size() - 1);

                    weightLeftIdx = weightLeftIdx - left_err;
                    weightRightIdx = weightRightIdx - right_err;

                    leftIdx = leftIdx - left_err;
                    rightIdx = rightIdx - right_err;
                }

            } else {

                if (leftIdx < 0) {
                    arma::sword left_err = leftIdx;

                    weightLeftIdx = weightLeftIdx - left_err;
                    leftIdx = leftIdx - left_err;
                }

                if (rightIdx >= input_idx.size()) {
                    arma::sword right_err = rightIdx - (input_idx.size() - 1);

                    weightRightIdx = weightRightIdx - right_err;
                    rightIdx = rightIdx - right_err;
                }
            }

            arma::vec values = input_values.subvec(leftIdx, rightIdx);

            // Define weights vector required for specific windows
            arma::vec weights;
            weights.copy_size(values);
            weights = polars::calculate_window_weights(win_type, windowSize, alpha).subvec(weightLeftIdx,
                                                                                           weightRightIdx);

            const Series subSeries = Series(values, input_idx.subvec(leftIdx, rightIdx));

            if (subSeries.finiteSize() >= minPeriods) {
                resultv(centerIdx) = processor.processWindow(subSeries, weights);
            } else {
                resultv(centerIdx) = processor.defaultValue();
            }
        }

        Series result = Series(polars::_ewm_correction(resultv, v, win_type), t);

        if(size() > 0 & windowSize > size()){
            return Series(result.values().head(size()), t);
        } else {
            return result;
        }
        //return result;
    }

    Window Series::rolling(SeriesSize windowSize,
                   SeriesSize minPeriods,
                   bool center,
                   bool symmetric,
                   polars::WindowProcessor::WindowType win_type,
                   double alpha) const {
        return Window((*this), windowSize, minPeriods, center, symmetric, win_type, alpha);
    };

    Rolling Series::rolling(SeriesSize windowSize,
                    SeriesSize minPeriods,
                    bool center,
                    bool symmetric) const {
        return Rolling((*this), windowSize, minPeriods, center, symmetric);
    };


    Series Series::clip(double lower_limit, double upper_limit) const {
        SeriesMask upper = SeriesMask(values() < upper_limit, index());
        SeriesMask lower = SeriesMask(values() > lower_limit, index());
        return Series(values(), index()).where(upper, upper_limit).where(lower, lower_limit);
    };


    Series Series::pow(double power) const {
        return Series(arma::pow(values(), power), index());
    }


    int Series::count() const {
        return finiteSize();
    }


    double Series::sum() const {
        arma::vec finites = finiteValues();
        if (finites.size() == 0) {
            return NAN;
        } else {
            return arma::sum(finites);
        }
    }


    double Series::mean() const {
        arma::vec finites = finiteValues();
        if (finites.size() == 0) {
            return NAN;
        } else {
            return arma::mean(finites);
        }
    }


    double Series::std(int ddof) const {
        arma::vec finites = finiteValues();
        if (ddof < 0) {
            ddof = 0;
        }
        auto n = finites.size();
        if (n <= ddof) {
            return NAN;
        } else {
            auto dev = (*this) - this->mean();
            auto squared_deviation = dev.pow(2);
            return std::pow(squared_deviation.sum() / (n - ddof), 0.5);
        }
    }


    Series::SeriesSize Series::size() const {
        //assert(index().size() == values().size());
        return index().size();
    }


    arma::vec Series::finiteValues() const {
        return values().elem(find_finite(values()));
    }


    Series::SeriesSize Series::finiteSize() const {
        //assert(index().size() == values().size());
        return finiteValues().size();
    }


// done this way so default copy / assignment works.
// todo; make copies of indices share memory as they are const?
    const arma::vec Series::index() const {
        return t;
    }


    const arma::vec Series::values() const {
        return v;
    }


    bool Series::equal(const Series &lhs, const Series &rhs) {
        return lhs.equals(rhs);
    }


    bool Series::almost_equal(const Series &lhs, const Series &rhs) {
        return lhs.almost_equals(rhs);
    }


    bool Series::not_equal(const Series &lhs, const Series &rhs) {
        return !lhs.equals(rhs);
    }

    Series Series::apply(double (*f)(double)) const {
        arma::vec vals = values();
        vals.transform([=](double val) { return (f(val)); });
        return Series(vals, index());
    }

    Series Series::index_as_series() const {
        return Series(index(), index());
    }

    std::map<double, double> Series::to_map() const {

        std::map<double, double> m;
        // put pairs into map
        for (int i = 0; i < size(); i++) {
            m.insert(std::make_pair(index()[i], values()[i]));
        }

        return m;
    }

    bool Series::empty() const {
        return (index().is_empty() & values().is_empty());
    }

    // TODO: Modify head once iloc has been refactored to accept slicing logic.
    Series Series::head(int n) const  {
        Series ser(values(), index());
        if(n >= ser.size()){
            return ser;
        } else {
            arma::uvec indices = arma::conv_to<arma::uvec>::from(polars::numc::arange(0, n));
            return ser.iloc(indices);
        }
    }

    // TODO: Modify tail once iloc has been refactored to accept slicing logic.
    Series Series::tail(int n) const  {

        Series ser(values(), index());

        if(n >= ser.size()){
            return ser;
        } else {
            arma::uword l = ser.size() - n;
            arma::uvec indices = arma::conv_to<arma::uvec>::from(polars::numc::arange(l, ser.size()));
            return ser.iloc(indices);
        }
    }

    /**
     * Add support for pretty printing of a Series object.
     * @param os the output stream that will be written to
     * @param ts the Series instance to output
     * @return the ostream for further piping
     */
    std::ostream &operator<<(std::ostream &os, const Series &ts) {

        if(ts.size() >= 5){
            os << "Series:\nindices\n" << ts.head(5).index() << "values\n" << ts.head(5).values();
            os << "\n....\n";
            os << "Series:\nindices\n" << ts.tail(5).index() << "values\n" << ts.tail(5).values();
            return os;
        } else {
            return os << "Series:\nindices\n" << ts.index() << "values\n" << ts.values();
        }
    }
}; // polars
