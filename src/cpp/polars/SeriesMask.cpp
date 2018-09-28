//
// Created by Calvin Giles on 23/03/2018.
//

#include <cassert>

#include "Series.h"
#include "SeriesMask.h"

namespace zimmer {

    SeriesMask::SeriesMask() = default;


    SeriesMask::SeriesMask(const arma::uvec &v, const arma::vec &t) : t(t), v(v) {
        //assert(t.n_cols == 1 && v.n_cols == 1);
        //assert(t.n_rows == v.n_rows);
        //assert(!arma::any(v > 1));  // Np SeriesMask values may be greater than 1
    };


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


    Series SeriesMask::to_series() const {
        return Series(arma::conv_to<arma::vec>::from(values()), index());
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
        return os << "SeriesMask:\ntimestamps\n" << ts.index() << "values\n" << ts.values();
    }


    const arma::vec SeriesMask::index() const { return t; }


    const arma::uvec SeriesMask::values() const { return v; }
}
