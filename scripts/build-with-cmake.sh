#!/bin/bash
(mkdir -p build/cmake-makefile-debug
cd build/cmake-makefile-debug
cmake -DCMAKE_BUILD_TYPE=Debug -G "Unix Makefiles" ../..
cmake --build . --target all -- -j 2)
