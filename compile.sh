#!/bin/bash

set -ex
nvcc -c phs.cu -o phs.o
nvcc -c main.cpp -o main.o
nvcc main.o phs.o -o phs.x

