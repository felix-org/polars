# polars
A c++ library that defines a TimeSeries class that behaves a bit like [pandas.Series](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html)

![swimming polar](http://cache.lovethispic.com/uploaded_images/247072-Swimming-Polar-Bear.jpg "Swimming Polar Bear")


## Getting started

Clone the repo as usual with git clone <url>

then:

```
git submodule update --init --recursive
```

This will fetch the dependencies (google test, date.h and armadillo at present).

The library should be easily integratable with `add_submodule` but this is yet to be tested.

If you get a dirty tree in dependencies/armadillo-code/examples/Makefile you may want to:

```
cd dependencies/armadillo-code
git update-index --assume-unchanged examples/Makefile
```

The file that changed is actually built by cmake so the changes can be readily ignored.

To install - WIP (need to add the install() command to the CMakeLists.txt)

## What is polars?

Polars was built to make cross platform mobile deployment easy - prototype in python, port quickly into C++, wrap into a library and deploy into ios and android.

Being strongly and staticly typed it catches many bugs at compile time.

In general:
* CMake installation (WIP)
* Developed and tested on macos primarily but should support linux distros
* Benefits from Accellerate on osx and ios through underlying library
* Benefits from other BLAS acceleration on linux and android through underlying library
* Good unit test coverage
* Alpha release - expect breaking changes in the near future, but will settle down quickly as we are following pandas.

Provides the following methods for a Series:

* Construction from armadillo vectors or std::lib vectors - xtensor to follow soon.
* `.to_map()` (like `.iter_rows()`)
* arithmetic operators +, -, *, / with another Series or with a numeric type
* comparison operators ==, !=, >=, <=, >, < with another Series or with a numeric type
* `.diff()`
* `.where()`
* `.abs()`
* `.pow()`
* `.mean()`
* `.quantile()`
* `.dropna()`
* `.fillna()`
* `.diff()`
* `.clip()`
* `.apply()`
* `<<` operator overloading (pretty printing)
* `.rolling()` supporting mean, quantile, std, sum for flat windows, triangle windows, and exponential windows (approximated)

It also provides a SeriesMask class which is the result of any comparison operation and is used as the input to `.where()`.

The SeriesMask provides the following methods:
* comparison operators &, |, ! with another Series
* `.to_series()` to convert to bool type to double
* `<<` operator overloading (pretty printing)

To make working with time series easier, we also have an experimental TimeSeries class derived from Series. This is a Series under the hood, but can be constructed and indexed with std::chrono types to remove the burden of working with times.


What is coming up?
* All unit tested (with moving CI to open source project) - Circle has granted us free containers just need to switch on
* testing in docker against ubuntu environment
* Example python bindings project
* Further development and tests for TimeSeriesMask
* LocalTimeSeries that works with `std::chrono::local_clock` where TimeSeries works with `std::chrono::system_clock` (i.e. utc)
* Making `.rolling()` more pandas with `.rolling().mean()` syntax
* date literals for TimeSeries
* `[]` syntax for subsetting Series
* `EnumSeries` to support strongly typed categorical series
