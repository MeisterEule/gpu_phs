#!/bin/bash

INC="./external/include"
set -ex
nvcc -I$INC -res-usage -c phs.cu -o phs.o
nvcc -I$INC -c phs_cpu.cpp -o phs_cpu.o
nvcc -I$INC -c monitoring.cpp -o monitoring.o
nvcc -I$INC -c file_input.cpp -o file_input.o
nvcc -I$INC -c rng.cpp -o rng.o
nvcc -I$INC -c main.cpp -o main.o
nvcc main.o rng.o monitoring.o file_input.o phs_cpu.o phs.o -o phs.x

