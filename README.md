# quantum_basis
Basis of condensed matter quantum lattice problems, for usage in exact diagonalization (ED). The code is designed for any general bosonic or fermionic problem (or a mix of both), as long as the user can provide the matrix form (explained in the manual) of the elementary operators of the Hamiltonian.

To learn how to use this library to design ED code for your own models, please refer to the folder "examples" (each complete example is about 100 lines):

. Fermi-Hubbard Model on square lattice

. Bose-Hubbard Model on square lattice (to be filled up)

. tJ model on triangular lattice (to be filled up)

. Kondo Lattice model on a chain (need more benchmark, currently the gap has been verified, but E0 not checked yet)

. xxz spin-1/2 model on a chain (to be filled up)

. xxz spin-1 model on kagome lattice (to be filled up)

Dependencies:
1. boost(>=1.56) (Download: http://www.boost.org/)
2. MKL (Download: https://software.intel.com/en-us/intel-mkl)
3. arpack (dependency to be removed in a future release)
4. arpack++ (dependency to be removed in a future release)

Note: using a modified version of arpack++ to be compatible with MKL. To download, go to 
https://github.com/wztzjhn/arpackpp

Currently the known working compilers are clang and gcc (c++11 required). (Intel icc not working yet)
