#!/bin/bash

set -ex
nvcc -c phs.cu -o phs.o
nvcc -c phs_cpu.cpp -o phs_cpu.o
#nvcc -c mappings.cu -o mappings.o
nvcc -c monitoring.cpp -o monitoring.o
nvcc -c main.cpp -o main.o
#nvcc main.o monitoring.o mappings.o phs.o -o phs.x
nvcc main.o monitoring.o phs_cpu.o phs.o -o phs.x

