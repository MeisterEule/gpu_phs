#!/bin/bash

set -ex
nvcc -c phs.cu -o phs.o
nvcc -c phs_cpu.cpp -o phs_cpu.o
nvcc -c monitoring.cpp -o monitoring.o
nvcc -c mom_generator.cpp -o mom_generator.o
nvcc -c main.cpp -o main.o
nvcc main.o monitoring.o mom_generator.o phs_cpu.o phs.o -o phs.x

