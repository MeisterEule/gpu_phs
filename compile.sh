#!/bin/bash

set -ex
nvcc -c phs.cu -o phs.o
nvcc -c monitoring.cpp -o monitoring.o
nvcc -c main.cpp -o main.o
nvcc main.o monitoring.o phs.o -o phs.x

