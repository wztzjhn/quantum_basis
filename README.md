# quantum_basis
Under development, not ready for practical use yet.

Basis of condensed matter quantum lattice problems, for usage in exact diagonalization (ED). The code is designed to be used for any bosonic or fermionic problem, as long as the user can provide the matrix form of the elementary operators of the Hamiltonian.

Dependencies:

1. boost(>=1.56)

2. MKL

3. arpack (dependency to be removed in future)

4. arpack++ (dependency to be removed in future)

Note: using a modified version of arpack++ to be compatible with MKL. To download:

git clone git@github.com:wztzjhn/arpackpp.git

Currently the known working compilers are clang and gcc (c++11 required). (Intel icc not tested yet)
