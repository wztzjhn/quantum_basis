[![C/C++ CI](https://github.com/wztzjhn/quantum_basis/actions/workflows/c-cpp.yml/badge.svg?branch=master)](https://github.com/wztzjhn/quantum_basis/actions/workflows/c-cpp.yml)
# Quantum Basis
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
- More featuers (good quantum numbers, translation symmetry, dynamical response, etc.)? Explore the folder *examples*!
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

## Install from source (Linux):

0. Create a local folder to keep the manually installed libraries (optional for experts): 

    ```cd ${HOME}; mkdir ~/installs; mkdir ~/installs/lib; mkdir ~/installs/include```

    Then add the following to the end of your *~/.bashrc* file and restart the terminal:

    ```
    export LD_LIBRARY_PATH=${HOME}/installs/lib:${HOME}/installs/lib64:$LD_LIBRARY_PATH
    if [ -d ${HOME}/installs/lib/pkgconfig ]; then
        export PKG_CONFIG_PATH=${HOME}/installs/lib/pkgconfig:$PKG_CONFIG_PATH
    fi
    if [ -d ${HOME}/installs/lib64/pkgconfig ]; then
        export PKG_CONFIG_PATH=${HOME}/installs/lib64/pkgconfig:$PKG_CONFIG_PATH
    fi
    export OMP_NUM_THREADS=1
    ```

1. Install MKL. There are two options: 

    - For some linux distros, MKL is already included in the package repository. For instance, in Ubuntu, you can simply type
    
        ```sudo apt install libmkl-dev```
    
    - Otherwise, you can download from [*Intel*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html), where the **Offline Installer** option is recommended.
    
    Either way, after installaltion, you should add the following to the end of your *~/.bashrc* file and restart the terminal:

    ```
    if [ -f /opt/intel/oneapi/setvars.sh ]; then
        source /opt/intel/oneapi/setvars.sh intel64
        export MKL_INC_DIR=${MKLROOT}/include
        export MKL_LIB_DIR=${MKLROOT}/lib/intel64
    elif [ -d /usr/include/mkl ] && [ -d /usr/lib/x86_64-linux-gnu ]; then
        export MKL_INC_DIR=/usr/include/mkl
        export MKL_LIB_DIR=/usr/lib/x86_64-linux-gnu
    fi
    unset PYTHONPATH
    ```

    Note: The variable *intel64* and the destination folders may vary depending on the system, should change the above lines accordingly.
    
2. Install the following dependencies (the names may appear slightly different for different Linux distros):

    - gfortran
    - g++
    - pkg-config
    - boost-dev

    A single command to get them on **Ubuntu**:
    
    ```
    sudo apt install gfortran g++ pkg-config libboost-dev
    ```

3. Install ARPACK-NG:

    ```
    cd /tmp
    wget https://github.com/opencollab/arpack-ng/archive/refs/tags/3.9.0.tar.gz
    tar xf 3.9.0.tar.gz
    cd arpack-ng-3.9.0
    sh bootstrap
    FFLAGS="-m64 -I${MKL_INC_DIR}" FCFLAGS="-m64 -I$MKL_INC_DIR" CFLAGS="-DMKL_ILP64 -m64 -I${MKL_INC_DIR}" CXXFLAGS="-DMKL_ILP64 -m64 -I${MKL_INC_DIR}" LIBS="-L${MKL_LIB_DIR} -Wl,--no-as-needed -lmkl_gf_ilp64 -lmkl_tbb_thread -lmkl_core -lpthread -ltbb -lstdc++ -lm -ldl" LIBSUFFIX="ILP64" INTERFACE64="1" ./configure --with-blas=mkl_gf_ilp64 --with-lapack=mkl_gf_ilp64 --enable-icb --prefix=$HOME/installs
    make check
    make install
    ```

4. Test *Quantum Basis*:

    Create a directory "bin" that sits at the same level of "src": `mkdir bin`, then go inside "src": `cd src`, then type

    ```make clean; make test```
    
    Then execute the file `./test.x` in folder bin. :beer: if you see "All tests passed!" :beer:

5. Install *Quantum Basis* as a library:

    ```make clean; make install```

6. To try out the examples, go to folders:
    - *examples/trans_absent/platform_linux* (without using translational symmetry)
    - *examples/trans_symmetric/platform_linux* (using translational symmetry)

    and use the makefiles therein.

## Restrictions on lattice:
When using translational symmetry, at least one of the dimensions (Lx, Ly, Lz, or number of sublattices) has to be an even number (current implementation of the generalized Lin Table).

