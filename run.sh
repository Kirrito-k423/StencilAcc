#!/bin/bash

set -e
set -v
rm -rf build
mkdir build
cd build
cmake ..
make
../compareAvgTime.sh