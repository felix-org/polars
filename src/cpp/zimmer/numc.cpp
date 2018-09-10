//
// Created by Linda Uruchurtu on 30/08/2018.
//

#include "numc.h"

#define EPSILON  (1.0E-150)
#define VERYSMALL    (1.0E-8)

namespace zimmer {

    namespace numc {

        // compare two vecs, taking account of NANs (normal comparison operators don't give true for NAN == NAN)
        bool equal_handling_nans(const arma::vec &lhs, const arma::vec &rhs) {

            //assert(lhs.n_cols == 1 && rhs.n_cols == 1);

            if ((lhs.n_rows != rhs.n_rows)) return false;
            if ((lhs.n_cols != rhs.n_cols)) return false;

            for (arma::uword idx = 0; idx < lhs.n_rows; idx++) {
                if (lhs[idx] != rhs[idx] && !(isnan(lhs[idx]) && isnan(rhs[idx]))) {
                    return false;
                }
            }

            return true;
        }

        bool almost_equal_doubles(double a, double b) {
            // TODO: this will have bugs, replace with AlmostEquals from Google Test if used for anything but test comparisons!
            double absDiff = fabs(a - b);
            if (absDiff < EPSILON) {
                return true;
            }

            double maxAbs = fmax(fabs(a), fabs(b));
            return (absDiff / maxAbs) < VERYSMALL;
        }

        // compare two vecs, taking account of NANs (normal comparison operators don't give true for NAN == NAN)
        bool almost_equal_handling_nans(const arma::vec &lhs, const arma::vec &rhs) {

            //assert(lhs.n_cols == 1 && rhs.n_cols == 1);

            if ((lhs.n_rows != rhs.n_rows)) return false;
            if ((lhs.n_cols != rhs.n_cols)) return false;

            for (arma::uword idx = 0; idx < lhs.n_rows; idx++) {
                if (!(isnan(lhs[idx]) && isnan(rhs[idx])) && !almost_equal_doubles(lhs[idx], rhs[idx])) {
                    return false;
                }
            }

            return true;
        }

        arma::vec arange(double start, double stop, double step) {
            // Equivalent to numpy arange
            return arma::regspace(start, step, stop-step);
        }

        double sum_finite(const arma::vec &series){
            return arma::sum(series.elem(arma::find_finite(series)));
        }


        arma::vec triang(int M, bool sym) {
            /* Same implementation as scipy.signal */

            if (M <= 0) {
                // TODO: Replace by cassert
                return arma::vec({});
            }

            if (M <= 1) {
                // Handle small arrays
                return arma::vec({1.0});
            }

            arma::vec w;
            arma::vec n = arange(1, floor((M + 1) / 2.0) + 1);

            if (sym == false) {
                M = M + 1;
            }

            if (M % 2 == 0) {
                w = (2 * n - 1.0) / M;
                w = arma::join_vert(w, arma::flipud(w));
            } else {
                w = 2 * n / (M + 1.0);
                arma::vec w_inv = arma::flipud(w);
                arma::vec pos = arange(1, w_inv.size());
                w = arma::join_vert(w, w_inv.elem(arma::conv_to<arma::uvec>::from(pos)));
            }

            if (sym == false) {
                arma::vec pos = arange(0, w.size() - 1);
                return w.elem(arma::conv_to<arma::uvec>::from(pos));
            } else {
                return w;
            }
        }

        arma::vec exponential(int M, double tau, bool sym){
            /* Same implementation as scipy.signal */

            if (M < 1) {
                // TODO: Replace by cassert
                return arma::vec({});
            }

            if (M == 1) {
                // Handle small arrays
                return arma::vec({1.0});
            }

            auto odd = M % 2;

            if(sym == false and odd == false){
                M = M + 1;
            }

            double center = (M - 1) / 2.0;
            arma::vec n = arange(0, M);
            arma::vec w = -1 * abs(n - center) / tau;
            w.transform( [](double val) { return (exp(val)); } );

            if(sym == false and odd == false){
                arma::vec pos = arange(0, w.size() - 1);
                return w.elem(arma::conv_to<arma::uvec>::from(pos));
            } else {
                return w;
            }
        }

    }

}
