NVCC=nvcc
CXX=g++
# The linker needs to be the Nvidia wrapper so that
# all library dependencies can be resolved automatically.
LD=$(NVCC)

CXXFLAGS=
NVCCFLAGS=-res-usage

INC=./external/include

bin_sources = main.o \
              file_input.o \
              monitoring.o \
              phs_cpu.o \
              rng.o \
              phs.o \
              global_phs.o 

lib_sources = monitoring.o \
              phs.o \
              global_phs.o \
              whizard_gpu_phs.o

%.o: %.cpp 
	$(CXX) -fPIC -I$(INC) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) -Xcompiler -fPIC -I$(INC) $(NVCCFLAGS) -c $< -o $@

%.mod: %.f90
	gfortran -fPIC -ffree-line-length-0 -c $<

phs.x: $(bin_sources)
	$(LD) $^ -o $@	

libphs.so: $(lib_sources)
	g++ -shared $^ -L/usr/local/cuda-12.1/lib64 -lcudart -o $@
		
all: phs.x libphs.so gpu_phs_whizard_interface.mod

clean:
	rm -f *.o *.mod phs.x libphs.so
