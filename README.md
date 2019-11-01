# Qbasis
Basis of condensed matter quantum lattice problems, for usage in exact diagonalization (ED). The code is designed for any general bosonic or fermionic problem (or a mix of both), as long as the user can provide the [matrix form of the elementary operators](docs/Manual.pdf) of the Hamiltonian.

## Human-friendly usage (Heisenberg model as *example*)
*Note: the grammars below may differ slightly from the acutal code, please refer to the folder "examples" for exact usage.*
- Writing a two-site spin-1/2 Heisenberg model can be as easy as:
```
auto Heisenberg = Sx1 * Sx2 + Sy1 * Sy2 + Sz1 * Sz2;
```
- Add a site "3"?
```
Heisenberg += Sx2 * Sx3 + Sy2 * Sy3 + Sz2 * Sz3;
```
- Add more exotic interactions?
```
Heisenberg += (Sx1 * Sx2 + Sy1 * Sy2 + Sz1 * Sz2) * (Sx1 * Sx2 + Sy1 * Sy2 + Sz1 * Sz2);
```
- Eigen problem?
```
Heisenberg.locate_E0_lanczos();
```
- Measurement?
```
Heisenberg.measure(Sx1*Sx2);
```
- More featuers (good quantum numbers, translation symmetry, dynamical response, etc.)? Explore the folder "examples"!
- Need design very different Hamiltonians? Explore the folder "examples" for ideas!

## Examples
To learn how to use this library to design ED code for your own models, please refer to the folder "examples":
- Chain
  - Heisenberg spin-1/2
  - Heisenberg spin-1
  - Kondo Lattice model
  - t-J model
- Honeycomb lattice
  - Spinless fermion
- Kagome lattice
  - Heisenberg spin-1/2
  - t-J model
- Square lattice
  - Bose-Hubbard model
  - Fermi-Hubbard model
  - Kondo Lattice model
- Triangular lattice
  - Heisenberg spin-1/2

## Dependencies:
- [Boost](http://www.boost.org/) (Boost lib and Qbasis lib have to be built with compatible compilers)
- [MKL](https://software.intel.com/en-us/intel-mkl) (some old versions may not work)
- [arpack](https://github.com/opencollab/arpack-ng) (dependency to be removed in future)
- [arpack++](https://github.com/wztzjhn/arpackpp/tree/long) (dependency to be removed in future)

**Note: using a modified version of arpack++ to be compatible with MKL.**

## Restrictions on lattice:
When using translational symmetry, at least one of the dimensions (Lx, Ly, Lz, or number of sublattices) has to be an even number (current implementation of the generalized Lin Table).

## Parallel scheme: 
MKL + OpenMP

## Compilation
Currently the known working compilers are g++, icpc, and clang++ (c++11 required).
(openmp in clang++ not tested yet)

The code can be compiled in two modes:
- 32-bit integer
- 64-bit integer (if the unrestricted Hilbert space reaches 10^9, it is necessary to use 64-bit mode)

**In the 64-bit mode**:
- arpack has to be compiled with "-fdefault-integer-8"
- arpack++ has to use the branch "long"
- linking to MKL has to use the ILP64 mode
