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
