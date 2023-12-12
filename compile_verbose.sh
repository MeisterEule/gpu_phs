#!/bin/bash

set -ex
nvcc -c -D_VERBOSE phs.cu -o phs.o
nvcc -c -D_VERBOSE monitoring.cpp -o monitoring.o
nvcc -c -D_VERBOSE main.cpp -o main.o
nvcc main.o monitoring.o phs.o -o phs.x

