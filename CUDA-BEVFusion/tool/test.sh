#!/bin/bash

. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

set -e

mkdir -p build

cd build
cmake ..
make -j

cd ..

./build/tester test_lidarmapsegm lidarmapsegm $DEBUG_PRECISION
