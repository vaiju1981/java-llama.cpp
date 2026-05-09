#!/bin/bash

mkdir -p build
cmake -Bbuild $@ || exit 1
cmake --build build --config Release -j$(nproc) || exit 1
