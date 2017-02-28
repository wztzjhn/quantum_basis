# quantum_basis
Basis of condensed matter quantum lattice problems, for usage in exact diagonalization (ED) and beyond. The code is designed for any general bosonic or fermionic problem (or a mix of both), as long as the user can provide the matrix form of the elementary operators of the Hamiltonian.

Dependencies:

1. boost(>=1.56)

2. MKL

3. arpack (dependency to be removed in future)

4. arpack++ (dependency to be removed in future)

Note: using a modified version of arpack++ to be compatible with MKL. To download:

git clone git@github.com:wztzjhn/arpackpp.git

Currently the known working compilers are clang and gcc (c++11 required). (Intel icc not tested yet)
