#include <complex>
#include <iostream>
#include <cassert>
#include "lanczos.h"


template <typename T>
void lanczos(const csr_mat<T> &mat, std::vector<std::vector<T>> &v, double hessenberg[])
{
    // alpha[j] = hessenberg[j+k], diagonal of hessenberg matrix
    // beta[j]  = hessenberg[j]
    //  a[0]  b[1]      -> note: beta[0] not used
    //  b[1]  a[1]  b[2]
    //        b[2]  a[2]  b[3]
    //              b[3]  a[3] b[4]
    //                    ..  ..  ..    b[k-1]
    //                          b[k-1]  a[k-1]
    assert(v[0].size() == v[1].size());
    assert(mat.dimension() == v[0].size());
    assert(std::abs(nrm2(v[0].size(), v[0].data(), 1) - 1.0) < lanczos_precision); // normalized
    MKL_INT k   = v.size() - 1;
    MKL_INT dim = v[0].size();
    assert(k < dim); // # of orthogonal vectors: at most dim
    hessenberg[0] = 0.0; // beta[0] not used
    mat.MultMv(v[0], v[1]);                                                       // w_0, not orthogonal to v[0] yet
    hessenberg[k] = std::real(dotc(dim, v[0].data(), 1, v[1].data(), 1));         // alpha[0]
    //std::cout << "alpha[0]=" << dotc(dim, v[0].data(), 1, v[1].data(), 1) << std::endl;
    axpy(dim, -hessenberg[k], v[0].data(), 1, v[1].data(), 1);                    // w_0, orthogonal but not normalized yet
    hessenberg[1] = nrm2(dim, v[1].data(), 1);                                    // beta[1]
    scal(dim, 1.0 / hessenberg[1], v[1].data(), 1);                               // v[1]
    for (MKL_INT j = 1; j < k-1; j++) {
        mat.MultMv(v[j], v[j+1]);
        axpy(dim, -hessenberg[j], v[j-1].data(), 1, v[j+1].data(), 1);            // w_j
        hessenberg[k+j] = std::real(dotc(dim, v[j].data(), 1, v[j+1].data(), 1)); // alpha[j]
        //std::cout << "alpha[" << j << "]=" << dotc(dim, v[j].data(), 1, v[j+1].data(), 1) << std::endl;
        axpy(dim, -hessenberg[k+j], v[j].data(), 1, v[j+1].data(), 1);            // w_j
        hessenberg[j+1] = nrm2(dim, v[j+1].data(), 1);                            // beta[j+1]
        scal(dim, 1.0 / hessenberg[j+1], v[j+1].data(), 1);                       // v[j+1]
    }
    mat.MultMv(v[k-1], v[k]);
    axpy(dim, -hessenberg[k-1], v[k-2].data(), 1, v[k].data(), 1);
    hessenberg[k+k-1] = std::real(dotc(dim, v[k-1].data(), 1, v[k].data(), 1));   // alpha[k-1]
    //std::cout << "alpha[" << k-1 << "]=" << dotc(dim, v[k-1].data(), 1, v[k].data(), 1) << std::endl;
    axpy(dim, -hessenberg[k+k-1], v[k-1].data(), 1, v[k].data(), 1);              // w_{k-1}
    double betak = nrm2(dim, v[k].data(), 1);                                     // beta[k] (not stored)
    scal(dim, 1.0 / betak, v[k].data(), 1);                                       // v[k]
}



// Explicit instantiation
template void lanczos(const csr_mat<double> &mat, std::vector<std::vector<double>> &v, double hessenberg[]);
template void lanczos(const csr_mat<std::complex<double>> &mat, std::vector<std::vector<std::complex<double>>> &v, double hessenberg[]);
