#ifndef MKL_INTERFACE_H
#define MKL_INTERFACE_H

#define MKL_Complex16 std::complex<double>
#include "mkl.h"

// blas level 1, Euclidean norm of vector
inline // double
double nrm2(const MKL_INT n, const double *x, const MKL_INT incx) {
    return dnrm2(&n, x, &incx);
}
inline // complex double
double nrm2(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx) {
    return dznrm2(&n, x, &incx);
}

// blas level 3, matrix matrix product
inline // double
void gemm(const char transa, const char transb, const MKL_INT m, const MKL_INT n, const MKL_INT k,
          const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb,
          const double beta, double *c, const MKL_INT ldc) {
    dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}
inline // complex double
void gemm(const char transa, const char transb, const MKL_INT m, const MKL_INT n, const MKL_INT k,
          const std::complex<double> alpha, const std::complex<double> *a, const MKL_INT lda,
          const std::complex<double> *b, const MKL_INT ldb,
          const std::complex<double> beta, std::complex<double> *c, const MKL_INT ldc) {
    zgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

#endif
