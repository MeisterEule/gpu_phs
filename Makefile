NVCC=nvcc
CXX=g++
# The linker needs to be the Nvidia wrapper so that
# all library dependencies can be resolved automatically.
LD=$(NVCC)

CXXFLAGS=
NVCCFLAGS=-res-usage

INC=-I./external/include -I/usr/local/cuda-12.1/include

sources = main.o \
          file_input.o \
          monitoring.o \
          phs_cpu.o \
          rng.o \
          phs.o

%.o: %.cpp 
	$(CXX) $(INC) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(INC) $(NVCCFLAGS) -c $< -o $@

phs.x: $(sources)
	$(LD) $^ -o $@ -L/home/christian/local/Linux_x86_64/24.3/cuda/12.3/targets/x86_64-linux/lib -lcudart
	

clean:
	rm -f *.o phs.x
