# quantum_basis
Basis of condensed matter quantum lattice problems, for usage in exact diagonalization (ED) and beyond. The code is designed for any general bosonic or fermionic problem (or a mix of both), as long as the user can provide the matrix form of the elementary operators of the Hamiltonian.

The users are encouraged to use this library to design ED code for any intercting quantum lattice model. 

Some examples of writing ED code based on this library are provided in the folder "examples":

. Fermi-Hubbard Model on square lattice

. Bose-Hubbard Model on square lattice (to be filled up)

. tJ model on triangular lattice (to be filled up)

. Kondo lattice model on a chain (to be filled up)

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
