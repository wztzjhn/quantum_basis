#ifndef MKL_INTERFACE_H
#define MKL_INTERFACE_H

#define MKL_INT size_t
#define MKL_Complex16 std::complex<double>
#include "mkl.h"

inline // double
void gemm(const char transa, const char transb, const size_t m, const size_t n, const size_t k,
          const double alpha, const double *a, const size_t lda, const double *b, const size_t ldb,
          const double beta, double *c, const size_t ldc) {
    return dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}
inline // complex double
void gemm(const char transa, const char transb, const size_t m, const size_t n, const size_t k,
          const std::complex<double> alpha, const std::complex<double> *a, const size_t lda, const std::complex<double> *b, const size_t ldb,
          const std::complex<double> beta, std::complex<double> *c, const size_t ldc) {
    return zgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

#endif
