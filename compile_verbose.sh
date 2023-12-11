#!/bin/bash

set -ex
nvcc -c -D_VERBOSE phs.cu -o phs.o
nvcc -c -D_VERBOSE main.cpp -o main.o
nvcc main.o phs.o -o phs.x

