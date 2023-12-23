# gpu_phs
Recursive phase-space generation on GPUs using CUDA.

This project models the phase space generation step in the HEP event simulator Whizard. The original algorithm has a very imperative nature by recursively iterating through the branches of kinematic trees. The program in this repository is a data-centric portation which can be executed on GPUs.
The phase space generation is one crucial step in the Whizard program flow. Most, having a parallelized implementation allows for the efficient usage of parallelized QFT matrix element evaluations.

## Building the Project

We use rapidjson as a git submodule to read input files. Therefore, when cloning this repository, you must do it like this:
```
git clone --recursive https://github.com/MeisterEule/gpu_phs
```

If you have cloned this repository without this option, you can retroactively activate the submodule via
```
git submodule update --init
```

The program is build using `make`. You need an Nvida compiler (nvcc) to compile.

## Program Layout

TBD

## Performance Considerations

TBD

