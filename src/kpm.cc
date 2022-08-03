#include <iostream>

#include "qbasis.h"

/** @file
     *  \fn void axpy(const MKL_INT n, const double alpha, const double *x, const MKL_INT incx, double *y, const MKL_INT incy)
     *  \brief blas level 1, y = a*x + y (double)
     */
inline
void axpy(const MKL_INT n, const double alpha, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
    daxpy(&n, &alpha, x, &incx, y, &incy);
}
/** @file
 *  \fn void axpy(const MKL_INT n, const std::complex<double> alpha, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy)
 *  \brief blas level 1, y = a*x + y (complex double)
 */
inline
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
// comment: zdotc is a problematic function in lapack, when returning std::complex. So using cblas here.
inline // complex double
std::complex<double> dotc(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx,
                          const std::complex<double> *y, const MKL_INT incy) {
    std::complex<double> result(0.0, 0.0);
    cblas_zdotc_sub(n, x, incx, y, incy, &result);
    return result;
}

namespace qbasis {
    
    template <typename T, typename MAT>
    void energy_scale(const MKL_INT &dim, const MAT &mat, T v[], double &lo, double &hi,
                      const double &extend, const MKL_INT &iters)
    {
        std::cout << "Locating upper/lower energy bounary..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        MKL_INT mm = iters - 1;
        std::vector<T*> vpt(mm+1);
        for (MKL_INT j = 0; j <= mm; j++) vpt[j] = &v[(j%2)*dim];
        
        std::vector<double> Hessenberg(2*iters,0.0);
        std::vector<double> ritz(mm), s(mm * mm);                                // Ritz values and eigenvecs of Hess
        
        vec_randomize(dim, vpt[0]);
        for (MKL_INT j = 0; j < dim; j++) vpt[1][j] = static_cast<T>(0.0);       // v[1] = 0
        mat.MultMv2(vpt[0], vpt[1]);                                             // v[1] = H * v[0] + v[1]
        Hessenberg[iters] = std::real(dotc(dim, vpt[0], 1, vpt[1], 1));          // a[0] = (v[0], v[1])
        axpy(dim, -Hessenberg[iters], vpt[0], 1, vpt[1], 1);                     // v[1] = v[1] - a[0] * v[0]
        Hessenberg[1] = nrm2(dim, vpt[1], 1);                                    // b[1] = || v[1] ||
        scal(dim, 1.0 / Hessenberg[1], vpt[1], 1);                               // v[1] = v[1] / b[1]
        
        for (MKL_INT m = 2; m <= mm; m++) {
            for (MKL_INT l = 0; l < dim; l++)
                vpt[m][l] = -Hessenberg[m-1] * vpt[m-2][l];                      // v[m] = -b[m-1] * v[m-2]
            mat.MultMv2(vpt[m-1], vpt[m]);                                       // v[m] = H * v[m-1] + v[m]
            Hessenberg[iters+m-1] = std::real(dotc(dim, vpt[m-1], 1, vpt[m], 1));// a[m-1] = (v[m-1], v[m])
            axpy(dim, -Hessenberg[iters+m-1], vpt[m-1], 1, vpt[m], 1);           // v[m] = v[m] - a[m-1] * v[m-1]
            Hessenberg[m] = nrm2(dim, vpt[m], 1);                                // b[m] = || v[m] ||
            scal(dim, 1.0 / Hessenberg[m], vpt[m], 1);                           // v[m] = v[m] / b[m]
        }
        hess_eigen(Hessenberg.data(), iters, mm, "sr", ritz, s);                 // calculate {theta, s}
        
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << std::endl << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        
        lo = ritz[0];
        hi = ritz[mm-1];
        double slack = extend * (hi - lo);
        lo -= slack;
        hi += slack;
        std::cout << "[lo, hi]: [" << lo << ", " << hi << "] (extented by " << extend << ")." << std::endl;
    }
    template void energy_scale(const MKL_INT &dim, const csr_mat<double> &mat,
                               double v[], double &lo, double &hi,
                               const double &extend, const MKL_INT &iters);
    template void energy_scale(const MKL_INT &dim, const csr_mat<std::complex<double>> &mat,
                               std::complex<double> v[], double &lo, double &hi,
                               const double &extend, const MKL_INT &iters);
    template void energy_scale(const MKL_INT &dim, const model<std::complex<double>> &mat,
                               std::complex<double> v[], double &lo, double &hi,
                               const double &extend, const MKL_INT &iters);
    
}
