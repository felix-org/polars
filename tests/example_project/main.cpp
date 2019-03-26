//
// Created by Calvin Giles on 2018-10-04.
//

#include <iostream>
#include "polars/Series.h"

int main() {
    Series s{{1, 2, 3, 4, 5}};
    Series t{{1, 2, 3, 4, 5}};
    cout << s + t << endl;
    return 0;
}