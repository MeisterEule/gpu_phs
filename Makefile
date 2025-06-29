NVCC=nvcc
CXX=g++
FC=gfortran
# The linker needs to be the Nvidia wrapper so that
# all library dependencies can be resolved automatically.
LD=$(NVCC)

CXXFLAGS=-fPIC
NVCCFLAGS=-res-usage -Xcompiler -fPIC
FFLAGS=-fPIC -ffree-line-length-0
CUDA_HOME=/usr/local/cuda-12.1

INC=-I./external/include -I$(CUDA_HOME)/include

bin_sources = main.o \
              file_input.o \
              monitoring.o \
              phs_cpu.o \
              rng.o \
              phs.o \
              global_phs.o 

lib_sources = monitoring.o \
              file_input.o \
              phs.o \
              global_phs.o \
              whizard_gpu_phs.o

%.o: %.cpp 
	$(CXX) $(INC) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(INC) $(NVCCFLAGS) -c $< -o $@

%.mod: %.f90
	$(FC) $(FCFLAGS) -c $<

phs.x: $(bin_sources)
	$(LD) $^ -o $@ -L$(CUDA_HOME)/lib -lcudart

libphs.so: $(lib_sources)
	g++ -shared $^ -L$(CUDA_HOME)/lib64 -lcudart -o $@
		
all: phs.x libphs.so gpu_phs_whizard_interface.mod

clean:
	rm -f *.o *.mod phs.x libphs.so
