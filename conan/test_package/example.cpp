#include <iostream>
#include "polars/Series.h"

int main() {
  auto a = polars::Series({1, 2, 1, 2}, {1, 2, 3, 4});
  auto b = a * a * a - 2;
  std::cout << b << std::endl;
}
