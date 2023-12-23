NVCC=nvcc
CXX=g++
# The linker needs to be the Nvidia wrapper so that
# all library dependencies can be resolved automatically.
LD=$(NVCC)

CXXFLAGS=
NVCCFLAGs=-res-usage

INC=./external/include

sources = main.o \
          file_input.o \
          monitoring.o \
          phs_cpu.o \
          rng.o \
          phs.o

%.o: %.cpp 
	$(CXX) -I$(INC) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) -I$(INC) $(NVCCFLAGS) -c $< -o $@

phs.x: $(sources)
	$(LD) $^ -o $@	
	

clean:
	rm -f *.o phs.x
