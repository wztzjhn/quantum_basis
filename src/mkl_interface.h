#ifndef MKL_INTERFACE_H
#define MKL_INTERFACE_H

#define MKL_Complex16 std::complex<double>

#ifndef lapack_int
#define lapack_int MKL_INT
#endif

#ifndef lapack_complex_double
#define lapack_complex_double   MKL_Complex16
#endif

#include "mkl.h"

namespace qbasis {
    // blas level 1, y = a*x + y
    inline // double
    void axpy(const MKL_INT n, const double alpha, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
        daxpy(&n, &alpha, x, &incx, y, &incy);
    }
    inline // complex double
    void axpy(const MKL_INT n, const std::complex<double> alpha, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy) {
        zaxpy(&n, &alpha, x, &incx, y, &incy);
    }


    // blas level 1, y = x
    inline // double
    void copy(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
        dcopy(&n, x, &incx, y, &incy);
    }
    inline // complex double
    void copy(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy) {
        zcopy(&n, x, &incx, y, &incy);
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

    inline // double
    void mkl_csrmm(const char transa, const MKL_INT m, const MKL_INT n, const MKL_INT k, const double alpha, const char *matdescra,
                   const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
                   const double *b, const MKL_INT ldb, const double beta, double *c, const MKL_INT ldc) {
        mkl_dcsrmm(&transa, &m, &n, &k, &alpha, matdescra, val, indx, pntrb, pntre, b, &ldb, &beta, c, &ldc);
    }
    inline // complex double
    void mkl_csrmm(const char transa, const MKL_INT m, const MKL_INT n, const MKL_INT k, const std::complex<double> alpha, const char *matdescra,
                   const std::complex<double> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
                   const std::complex<double> *b, const MKL_INT ldb, const std::complex<double> beta, std::complex<double> *c, const MKL_INT ldc) {
        mkl_zcsrmm(&transa, &m, &n, &k, &alpha, matdescra, val, indx, pntrb, pntre, b, &ldb, &beta, c, &ldc);
    }

    // sparse blas, convert csr to csc
    inline // double
    void mkl_csrcsc(const MKL_INT *job, const MKL_INT n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0,
                    double *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info) {
        mkl_dcsrcsc(job, &n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info);
    }
    inline // complex double
    void mkl_csrcsc(const MKL_INT *job, const MKL_INT n, std::complex<double> *Acsr, MKL_INT *AJ0, MKL_INT *AI0,
                    std::complex<double> *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info) {
        mkl_zcsrcsc(job, &n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info);
    }

    // lapack computational routine, computes all eigenvalues of a real symmetric tridiagonal matrix using QR algorithm.
    inline // double
    lapack_int sterf(const lapack_int &n, double *d, double *e) {
        return LAPACKE_dsterf(n, d, e);
    }

    // lapack computational routine, computes all eigenvalues and (optionally) eigenvectors of a symmetric/hermitian tridiagonal matrix using the divide and conquer method.
    inline // double
    lapack_int stedc(const int &matrix_layout, const char &compz, const lapack_int &n, double *d, double *e, double *z, const lapack_int &ldz) {
        return LAPACKE_dstedc(matrix_layout, compz, n, d, e, z, ldz);
    }
    inline // complex double (for the unitary matrix which brings the original matrix to tridiagonal form)
    lapack_int stedc(const int &matrix_layout, const char &compz, const lapack_int &n, double *d, double *e, std::complex<double> *z, const lapack_int &ldz) {
        return LAPACKE_zstedc(matrix_layout, compz, n, d, e, z, ldz);
    }



    //// lapack symmetric eigenvalue driver routine, using divide and conquer, for band matrix
    //inline // double
    //lapack_int bevd(const int &matrix_layout, const char &jobz, const char &uplo, const lapack_int &n, const lapack_int &kd,
    //                double *ab, const lapack_int &ldab, double *w, double *z, const lapack_int &ldz) {
    //    return LAPACKE_dsbevd(matrix_layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz);
    //}
    //lapack_int bevd(const int &matrix_layout, const char &jobz, const char &uplo, const lapack_int &n, const lapack_int &kd,
    //                std::complex<double> *ab, const lapack_int &ldab, double *w, std::complex<double> *z, const lapack_int &ldz) {
    //    return LAPACKE_zhbevd(matrix_layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz);
    //}


    // lapack, Computes the QR factorization of a general m-by-n matrix.
    inline // double
    lapack_int geqrf(const int &matrix_layout, const lapack_int &m, const lapack_int &n, double *a, const lapack_int &lda, double *tau) {
        return LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);
    }
    inline // complex double
    lapack_int geqrf(const int &matrix_layout, const lapack_int &m, const lapack_int &n, std::complex<double> *a, const lapack_int &lda, std::complex<double> *tau) {
        return LAPACKE_zgeqrf(matrix_layout, m, n, a, lda, tau);
    }

    // lapack, Multiplies a real matrix by the orthogonal matrix Q of the QR factorization formed by ?geqrf or ?geqpf.
    inline // double
    lapack_int ormqr(const int &matrix_layout, const char &side, const char &trans,
                     const lapack_int &m, const lapack_int &n, const lapack_int &k,
                     const double *a, const lapack_int &lda, const double *tau, double *c, const lapack_int &ldc) {
        return LAPACKE_dormqr(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }

}

#endif
