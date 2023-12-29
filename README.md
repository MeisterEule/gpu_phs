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

## Layout of the serial algorithm

The serial phase space generation in Whizard for an individual phase space tree consitsts of two steps. In the first one, each internal branch of the tree gets assigned an off-shell four-vector magnitude `msq` and decay momenta `p_decay`. The construction happens in each particle's rest frame, so that 1 -> 2 decay kinematics can be applied. 
The momenta are applied in a recursive way. The recursion starts at the root branch proceeds on child branches, if they have children themselves. This way, only internal branches are traversed. On each of them, `msq` is computed from a random number x with a mapping function. The form of the mapping function depends on the process and the branch and is determined by Whizard. 

The second step of four-momenta construction is the generation of scattering angles `theta` and `phi` and the boost from the rest frame into the lab frame. This is also a recursive algorithm, which has two parts. One part computes the scattering angles from random numbers, where `phi` is simply `2*pi*x`, whereas `theta` is generated as `cos(theta)` from a mapping function similarly to `msq`. This is done for internal branches only. The other part are boosts from the rest frame to the lab frame. The boost velocity is obtained from `p_decay`, computed in the first step, the direction from the scattering angles. It is crucial to note that the boost applied to a branch is the product of the boosts of all the parent branches. The boost matrix is an argument to the recursive function `set_angles`, and for each call a new array of dimension 16 is put on the stack. 

It must be noted that in step one, the random numbers given to the function can correspond to kinematically forbidden regions. In that case, the function returns `ok = false`, and the second step is skipped. The integration grids will be adapted in Whizard, so that the number of points for which this test will fail decreases.

## Porting approach

When porting to a GPU, these are the major points that need to be addressed

1. Generally, branches (if-statements) should be avoided as much as possible. A GPU core (SMT) cannot evaluate branches like a CPU does. Instead, for each branch, a separate run of the kernel is performed. Each time, all the threads take part, but only the ones for which the condition is true are actually computed on. This is a waste of resources, especially for deeply nested branch conditions. In the phase space generation, branches take place at two important places: The recursion and the selection of mapping functions. The first issue is discussed in the next point below. 

## Performance Considerations

TBD

