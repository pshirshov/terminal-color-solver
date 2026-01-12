#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p build-test
cd build-test
cmake .. -DUSE_ROCM=OFF
cmake --build . --target color_test -j
ctest --output-on-failure
