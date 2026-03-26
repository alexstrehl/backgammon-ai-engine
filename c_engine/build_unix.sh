#!/bin/bash
# Build the C backgammon engine as a shared library (Linux/macOS).
set -e

echo "Building libbg_engine.so ..."
gcc -O2 -shared -fPIC -o libbg_engine.so bg_engine.c -Wall
echo "Success: libbg_engine.so created."
