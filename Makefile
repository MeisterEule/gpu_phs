NVCC=nvcc
CXX=g++
FC=gfortran
# The linker needs to be the Nvidia wrapper so that
# all library dependencies can be resolved automatically.
LD=$(NVCC)

CXXFLAGS=-fPIC
NVCCFLAGS=-res-usage -Xcompiler -fPIC
FFLAGS=-fPIC -ffree-line-length-0

INC=-I./external/include -I/home/christian/local/Linux_x86_64/24.3/cuda/12.3/targets/x86_64-linux/include

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

phs.x: $(sources)
	$(LD) $^ -o $@ -L/home/christian/local/Linux_x86_64/24.3/cuda/12.3/targets/x86_64-linux/lib -lcudart

libphs.so: $(lib_sources)
	g++ -shared $^ -L/usr/local/cuda-12.1/lib64 -lcudart -o $@
		
all: phs.x libphs.so gpu_phs_whizard_interface.mod

clean:
	rm -f *.o *.mod phs.x libphs.so
