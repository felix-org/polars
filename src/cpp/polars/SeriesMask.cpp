//
// Created by Calvin Giles on 23/03/2018.
//

#include "SeriesMask.h"

#include "Series.h"
#include "numc.h"

#include <cassert>


namespace polars {

    SeriesMask::SeriesMask() = default;


    SeriesMask::SeriesMask(const arma::uvec &v, const arma::vec &t) : t(t), v(v) {
        //assert(t.n_cols == 1 && v.n_cols == 1);
        //assert(t.n_rows == v.n_rows);
        //assert(!arma::any(v > 1));  // Np SeriesMask values may be greater than 1
    };

    SeriesMask SeriesMask::iloc(int from, int to, int step) const {

        if(empty()  || (from == to)){
            return SeriesMask();
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

        return SeriesMask(values().elem(pos),index().elem(pos));
    }

    // TODO: Add slicing logic of the form .iloc(int start, int stop, int step=1) so it can be called like ser.iloc(0, -10).
    SeriesMask SeriesMask::iloc(const arma::uvec &pos) const {
        return SeriesMask(values().elem(pos), index().elem(pos));
    }


    double SeriesMask::iloc(arma::uword pos) const {
        arma::uvec val = values().elem(arma::uvec{pos});
        return val[0];
    }

    // by label of indices
    SeriesMask SeriesMask::loc(const arma::vec &index_labels) const {

        std::vector<int> indices;

        for (int j = 0; j < index_labels.n_elem; j++) {

            arma::uvec idx = arma::find(index() == index_labels[j]);

            if (!idx.empty()) {
                indices.push_back(idx[0]);
            }
        }

        if (indices.empty()) {
            return SeriesMask();
        } else {
            arma::uvec indices_v = arma::conv_to<arma::uvec>::from(indices);
            return SeriesMask(values().elem(indices_v), index().elem(indices_v));
        }
    }

    SeriesMask SeriesMask::loc(arma::uword pos) const {
        arma::uvec idx = arma::find(index() == pos);

        if (!idx.empty()) {
            return SeriesMask(values(), index()).iloc(idx);
        } else {
            return SeriesMask();
        }
    }

    // Series [op] int methods
    SeriesMask SeriesMask::operator==(const bool rhs) const {
        arma::ivec rhs_vec = arma::ones<arma::ivec>(this->size()) * (int)rhs;
        arma::ivec abs_diff = arma::abs(values() - rhs_vec);
        return {abs_diff == 0, index()};
    }


    SeriesMask SeriesMask::operator!=(const bool rhs) const {  // TODO implement as negation of operator==
        arma::ivec rhs_vec = arma::ones<arma::ivec>(this->size()) * (int)rhs;
        arma::ivec abs_diff = arma::abs(values() - rhs_vec);
        return {abs_diff != 0, index()};
    }


    // Series [op] Series methods
    SeriesMask SeriesMask::operator==(const SeriesMask &rhs) const {
        // TODO: make this fast enough to always check at runtime
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return SeriesMask(values() == rhs.values(), index());
    }


    SeriesMask SeriesMask::operator!=(const SeriesMask &rhs) const {
        // TODO: make this fast enough to always check at runtime
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return SeriesMask(values() != rhs.values(), index());
    }

    SeriesMask SeriesMask::operator|(const SeriesMask &rhs) const {
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return SeriesMask((values() + rhs.values()) > 0, index());
    }


    SeriesMask SeriesMask::operator&(const SeriesMask &rhs) const {
        //assert(!arma::any(index() != rhs.index()));  // Use not any != to handle empty array case
        return SeriesMask((values() + rhs.values()) == 2, index());
    }


    SeriesMask SeriesMask::operator!() const {
        return SeriesMask(values() == 0, index());
    }


    bool SeriesMask::equals(const SeriesMask &rhs) const {
        if ((values().n_rows != rhs.values().n_rows)) return false;
        if ((index().n_rows != rhs.index().n_rows)) return false;

        if (any(values() != rhs.values())) return false;

        if (any(index() != rhs.index())) return false;
        return true;
    }

    SeriesMask::SeriesSize SeriesMask::size() const {
        //assert(index().size() == values().size());
        return index().size();
    }


    /**
     * Static method
     * @param lhs
     * @param rhs
     * @return
     */
    bool SeriesMask::equal(const SeriesMask &lhs, const SeriesMask &rhs) {
        return lhs.equals(rhs);
    }

    //
    //bool not_equal(const SeriesMask &lhs, const SeriesMask &rhs) {
    //    return !lhs.equals(rhs);
    //}


    /**
     * Add support for pretty printing of a SeriesMask object.
     * @param os the output stream that will be written to
     * @param ts the SeriesMask instance to output
     * @return the ostream for further piping
     */
    std::ostream &operator<<(std::ostream &os, const SeriesMask &ts) {
        return os << "SeriesMask:\nindices\n" << ts.index() << "values\n" << ts.values();
    }


    const arma::vec SeriesMask::index() const { return t; }


    const arma::uvec SeriesMask::values() const { return v; }


    std::map<double, bool> SeriesMask::to_map() const {

        std::map<double, bool> m;
        // put pairs into map
        for (int i = 0; i < size(); i++) {
            m.insert(std::make_pair(index()[i], values()[i]));
        }

        return m;
    }

    bool SeriesMask::empty() const {
        return (index().is_empty() & values().is_empty());
    }

    // TODO: Modify head once iloc has been refactored to accept slicing logic.
    SeriesMask SeriesMask::head(int n) const  {
        if(n >= size()){
            return *this;
        } else {
            arma::uvec indices = arma::conv_to<arma::uvec>::from(polars::numc::arange(0, n));
            return iloc(indices);
        }
    }

    // TODO: Modify tail once iloc has been refactored to accept slicing logic.
    SeriesMask SeriesMask::tail(int n) const  {
        if(n >= size()){
            return *this;
        } else {
            arma::uvec indices = arma::conv_to<arma::uvec>::from(polars::numc::arange(size() - n, size()));
            return iloc(indices);
        }
    }
}
