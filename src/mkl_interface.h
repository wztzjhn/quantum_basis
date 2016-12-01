#ifndef MKL_INTERFACE_H
#define MKL_INTERFACE_H

#define MKL_Complex16 std::complex<double>
#include "mkl.h"

// blas level 1, y = a*x + y
inline // double
void axpy(const MKL_INT n, const double alpha, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
    daxpy(&n, &alpha, x, &incx, y, &incy);
}
inline // complex double
void axpy(const MKL_INT n, const std::complex<double> alpha, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy) {
    zaxpy(&n, &alpha, x, &incx, y, &incy);
}

// blas level 1, Euclidean norm of vector
inline // double
double nrm2(const MKL_INT n, const double *x, const MKL_INT incx) {
    return dnrm2(&n, x, &incx);
}
inline // complex double
double nrm2(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx) {
    return dznrm2(&n, x, &incx);
}

// blas level 1, rescale: x = a*x
inline // double * double vector
void scal(const MKL_INT n, const double a, double *x, const MKL_INT incx) {
    dscal(&n, &a, x, &incx);
}
inline // double complex * double complex vector
void scal(const MKL_INT n, const std::complex<double> a, std::complex<double> *x, const MKL_INT incx) {
    zscal(&n, &a, x, &incx);
}
inline // double * double complex vector
void scal(const MKL_INT n, const double a, std::complex<double> *x, const MKL_INT incx) {
    zdscal(&n, &a, x, &incx);
}


// blas level 1, conjugated vector dot vector
inline // double
double dotc(const MKL_INT n, const double *x, const MKL_INT incx, const double *y, const MKL_INT incy) {
    return ddot(&n, x, &incx, y, &incy);
}

inline // complex double
std::complex<double> dotc(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx, const std::complex<double> *y, const MKL_INT incy) {
    std::complex<double> result;
    zdotc(&result, &n, x, &incx, y, &incy);
    return result;
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


// sparse blas routines
inline // double
void csrgemv(const char transa, const MKL_INT m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y) {
    mkl_cspblas_dcsrgemv(&transa, &m, a, ia, ja, x, y);
}
inline // complex double
void csrgemv(const char transa, const MKL_INT m, const std::complex<double> *a, const MKL_INT *ia, const MKL_INT *ja, const std::complex<double> *x, std::complex<double> *y) {
    mkl_cspblas_zcsrgemv(&transa, &m, a, ia, ja, x, y);
}

// for symmetric matrix (NOT Hermitian matrix)
inline // double
void csrsymv(const char uplo, const MKL_INT m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y) {
    mkl_cspblas_dcsrsymv(&uplo, &m, a, ia, ja, x, y);
}
inline // complex double
void csrsymv(const char uplo, const MKL_INT m, const std::complex<double> *a, const MKL_INT *ia, const MKL_INT *ja, const std::complex<double> *x, std::complex<double> *y) {
    mkl_cspblas_zcsrsymv(&uplo, &m, a, ia, ja, x, y);
}

// more general function to perform matrix vector product in mkl
inline // double
void mkl_csrmv(const char transa, const MKL_INT m, const MKL_INT k, const double alpha, const char *matdescra,
               const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
               const double *x, const double beta, double *y) {
    mkl_dcsrmv(&transa, &m, &k, &alpha, matdescra, val, indx, pntrb, pntre, x, &beta, y);
}
inline // complex double
void mkl_csrmv(const char transa, const MKL_INT m, const MKL_INT k, const std::complex<double> alpha, const char *matdescra,
               const std::complex<double> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
               const std::complex<double> *x, const std::complex<double> beta, std::complex<double> *y) {
    mkl_zcsrmv(&transa, &m, &k, &alpha, matdescra, val, indx, pntrb, pntre, x, &beta, y);
}

#endif
