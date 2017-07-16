# Qbasis
Basis of condensed matter quantum lattice problems, for usage in exact diagonalization (ED). The code is designed for any general bosonic or fermionic problem (or a mix of both), as long as the user can provide the [matrix form of the elementary operators](Manual.pdf) of the Hamiltonian.

## Examples
To learn how to use this library to design ED code for your own models, please refer to the folder "examples":
- Heisenberg spin-1/2 chain
- Heisenberg spin-1 chain
- Heisenberg spin-1/2 triangular lattice
- Heisenberg spin-1/2 kagome lattice
- Spinless fermion on honeycomb lattice
- Fermi-Hubbard Model on square lattice
- Bose-Hubbard Model on square lattice
- Kondo Lattice Model on a chain (currently the gap has been checked, but E0 not checked yet)
- tJ Model on chain lattice (E0 checked, but E1 inconsistent with ALPS)
- tJ Model on kagome lattice (E0 checked, but E1 inconsistent with ALPS)

## Dependencies:
- [boost](http://www.boost.org/)
- [MKL](https://software.intel.com/en-us/intel-mkl) (some old versions may not work)
- [arpack](https://github.com/opencollab/arpack-ng) (dependency to be removed in a future release)
- [arpack++](https://github.com/wztzjhn/arpackpp) (dependency to be removed in a future release)

**Note: using a modified version of arpack++ to be compatible with MKL.**

## Restrictions on lattice:
When using translational symmetry, at least one of the dimensions (Lx, Ly, Lz, or number of sublattices) has to be an even number (current implementation of Lin Table).

## Parallel scheme: 
MKL + OpenMP

## Compilation
Currently the known working compilers are g++, icpc, and clang++ (c++11 required).
(openmp in clang++ not tested yet)

The code can be compiled in two modes:
- 32-bit integer (less cost)
- 64-bit integer (if the unrestricted Hilbert space reaches 10^9, it is necessary to use 64-bit mode)

**In the 64-bit mode**:
- arpack has to be compiled with "-fdefault-integer-8"
- arpack++ has to use the branch "long"
- linking to MKL has to use the ILP64 mode
