//
// Created by Calvin Giles on 23/03/2018.
//

#ifndef ZIMMER_SERIESMASK_H
#define ZIMMER_SERIESMASK_H

#include "armadillo"

class Series;
namespace zimmer {
    // TODO combine Series and SeriesMask via a BaseSeries
    // TODO Create IntSeries and EnumSeries
    // TODO rename Series -> DoubleSeries and SeriesMask -> BoolSeries

    class SeriesMask {
    public:
        typedef arma::uword SeriesSize;
        SeriesMask();

        SeriesMask(const arma::uvec &v, const arma::vec &t);

        SeriesMask operator|(const SeriesMask &rhs) const;
        SeriesMask operator&(const SeriesMask &rhs) const;
        SeriesMask operator!() const;

        bool equals(const SeriesMask &rhs) const;

        Series to_series() const;

        SeriesSize size() const;

        static bool equal(const SeriesMask &lhs, const SeriesMask &rhs);

        // done this way so default copy / assignment works.
        // todo; make copies of series share memory as they are const?
        const arma::vec index() const;
        const arma::uvec values() const;

    private:
        arma::vec t;
        arma::uvec v;
    };


    //bool not_equal(const SeriesMask &lhs, const SeriesMask &rhs);
    std::ostream &operator<<(std::ostream &os, const SeriesMask &ts);

}
#endif //ZIMMER_SERIESMASK_H
