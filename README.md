# quantum_basis
basis of condensed matter quantum lattice problems, for usage in exact diagonalization (ED)

under development, not ready for practical use yet

Prerequisites:

1. boost(>=1.56)

2. MKL

3. arpack (dependency to be removed in future)

4. arpack++ (dependency to be removed in future)

Note: using a modified version of arpack++ to be compatible with MKL. To download:

git clone git@github.com:wztzjhn/arpackpp.git

Currently the known working compilers are clang and gcc (c++11 required). (Intel icc not working yet)
