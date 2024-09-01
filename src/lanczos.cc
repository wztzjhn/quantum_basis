#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "arpack-ng/arpack.hpp"

#include "qbasis.h"

// y := a * x + y
inline void blas_axpy(const MKL_INT &n, const double &alpha, const double *x, const MKL_INT &incx,
                      double *y, const MKL_INT &incy) {
    cblas_daxpy(n, alpha, x, incx, y, incy);
}
inline void blas_axpy(const MKL_INT &n, const std::complex<double> &alpha, const std::complex<double> *x, const MKL_INT &incx,
                      std::complex<double> *y, const MKL_INT &incy) {
    cblas_zaxpy(n, &alpha, x, incx, y, incy);
}

// y := x
inline void blas_copy(const MKL_INT &n, const double *x, const MKL_INT &incx, double *y, const MKL_INT &incy) {
    cblas_dcopy(n, x, incx, y, incy);
}
inline void blas_copy(const MKL_INT &n, const std::complex<double> *x, const MKL_INT &incx, std::complex<double> *y, const MKL_INT &incy) {
    cblas_zcopy(n, x, incx, y, incy);
}

// Euclidean norm
inline double blas_nrm2(const MKL_INT &n, const double *x, const MKL_INT &incx) {
    return cblas_dnrm2(n, x, incx);
}
inline double blas_nrm2(const MKL_INT &n, const std::complex<double> *x, const MKL_INT &incx) {
    return cblas_dznrm2(n, x, incx);
}

//  x := a * x
inline void blas_scal(const MKL_INT &n, const double &a, double *x, const MKL_INT &incx) {
    cblas_dscal(n, a, x, incx);
}
inline void blas_scal(const MKL_INT &n, const std::complex<double> &a, std::complex<double> *x, const MKL_INT &incx) {
    cblas_zscal(n, &a, x, incx);
}

// conj(x) . y
inline double blas_dotc(const MKL_INT &n, const double *x, const MKL_INT &incx, const double *y, const MKL_INT &incy) {
    return cblas_ddot(n, x, incx, y, incy);
}
inline std::complex<double> blas_dotc(const MKL_INT &n, const std::complex<double> *x, const MKL_INT &incx,
                                      const std::complex<double> *y, const MKL_INT &incy) {
    std::complex<double> result(0.0, 0.0);
    cblas_zdotc_sub(n, x, incx, y, incy, &result);
    return result;
}

// C := alpha * op(A) * op(B) + beta * C
// A: mxk, B: kxn, C: mxn
inline void blas_gemm(const CBLAS_LAYOUT &Layout, const CBLAS_TRANSPOSE &transA, const CBLAS_TRANSPOSE &transB,
                      const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const double &alpha,
                      const double *a, const MKL_INT &lda, const double *b, const MKL_INT &ldb,
                      const double &beta, double *c, const MKL_INT &ldc) {
    cblas_dgemm(Layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void blas_gemm(const CBLAS_LAYOUT &Layout, const CBLAS_TRANSPOSE &transA, const CBLAS_TRANSPOSE &transB,
                      const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const std::complex<double> &alpha,
                      const std::complex<double> *a, const MKL_INT &lda, const std::complex<double> *b, const MKL_INT &ldb,
                      const std::complex<double> &beta, std::complex<double> *c, const MKL_INT &ldc) {
    cblas_zgemm(Layout, transA, transB, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}

// Computes all eigenvalues and, optionally, all eigenvectors of a real symmetric / complex Hermitian matrix
// using divide and conquer algorithm.
inline lapack_int lapack_syevd_heevd(const int &matrix_layout, const char &jobz, const char &uplo,
                                     const lapack_int &n, double* a, const lapack_int &lda, double *w) {
    return LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, a, lda, w);
}
inline lapack_int lapack_syevd_heevd(const int &matrix_layout, const char &jobz, const char &uplo,
                                     const lapack_int &n, lapack_complex_double* a, const lapack_int &lda, double *w) {
    return LAPACKE_zheevd(matrix_layout, jobz, uplo, n, a, lda, w);
}

namespace qbasis {

    // the ckpt functions are defined in ckpt.cc
    template <typename T>
    void ckpt_lanczos_init(MKL_INT &k, const MKL_INT &maxit, const MKL_INT &dim,
                           int &cnt_accuE0, double &accuracy, double &theta0_prev, double &theta1_prev,
                           T v[], double hessenberg[], const std::string &purpose);

    template <typename T>
    void ckpt_lanczos_update(const MKL_INT &m, const MKL_INT &maxit, const MKL_INT &dim,
                             int &cnt_accuE0, double &accuracy, double &theta0_prev, double &theta1_prev,
                             T v[], double hessenberg[], const std::string &purpose);



    template <typename T>
    void ckpt_CG_init(MKL_INT &m, const MKL_INT &maxit, const MKL_INT &dim, T v[], T r[], T p[]);

    template <typename T>
    void ckpt_CG_update(const MKL_INT &m, const MKL_INT &dim, T v[], T r[], T p[]);

    void log_Lanczos_srval(const MKL_INT &k, const std::vector<double> &ritz,
                           const double hessenberg[], const MKL_INT &maxit,
                           const double &accuracy,
                           const double &accu_E0, const double &accu_E1,
                           const std::string &filename)
    {
        assert(k > 2);
        std::ofstream fout(filename, std::ios::out | std::ios::app);
        fout << std::setprecision(10);
        fout << std::setw(20) << "#(1)" << std::setw(20) << "(2)" << std::setw(20) << "(3)"
             << std::setw(20) << "(4)"  << std::setw(20) << "(5)" << std::setw(20) << "(6)"
             << std::setw(20) << "(7)"  << std::setw(20) << "(8)" << std::setw(20) << "(9)"
             << std::setw(20) << "(10)" << std::endl;
        fout << std::setw(20) << "Iter(k)"
             << std::setw(20) << "Ritz[0]"  << std::setw(20) << "Ritz[1]"
             << std::setw(20) << "Ritz[2]"  << std::setw(20) << "Ritz[3]"
             << std::setw(20) << "a[k-1]"     << std::setw(20) << "b[k]"
             << std::setw(20) << "accuracy" << std::setw(20) << "accu_E0"
             << std::setw(20) << "accu_E1"  << std::endl;
        fout << std::setw(20) << k
             << std::setw(20) << ritz[0] << std::setw(20) << ritz[1]
             << std::setw(20) << ritz[2] << std::setw(20) << ritz[3]
             << std::setw(20) << hessenberg[maxit+k-1] << std::setw(20) << hessenberg[k]
             << std::setw(20) << accuracy << std::setw(20) << accu_E0
             << std::setw(20) << accu_E1 << std::endl;
        fout.close();
    }

    // need further classification:
    // 1. ask lanczos to restart with a new linearly independent vector when v_m+1 = 0
    // 2. add DGKS re-orthogonalization (when purpose == iram)
    // 3. add partial and selective re-orthogonalization
    template <typename T, typename MAT>
    void lanczos(MKL_INT k, MKL_INT np, const MKL_INT &maxit, MKL_INT &m, const MKL_INT &dim,
                 const MAT &mat, T v[], double hessenberg[], const std::string &purpose)
    {
        auto &npos = std::string::npos;
        MKL_INT mm = k + np;
        double theta0_prev, theta1_prev;                                         // record Ritz values from last step
        int cnt_accuE0 = 0;
        double accuracy;

        ckpt_lanczos_init(k, maxit, dim, cnt_accuE0, accuracy, theta0_prev, theta1_prev, v, hessenberg, purpose);
        m = k;
        np = mm - k;
        assert(mm < maxit && k >= 0 && np >= 0);
        assert(purpose != "iram" || mm < dim);                                   // # of orthogonal vectors: at most dim
        if ( cnt_accuE0 > 15 && accuracy < lanczos_precision) return;
        if (np == 0) return;

        T zero = static_cast<T>(0.0);
        std::vector<T*> vpt(mm+1);                                               // pointers of v[0],v[1],...,v[m]
        T* phipt = &v[2*dim];                                                    // for ground state eigenvector
        T* ypt = (purpose.find("vec1") != npos) ? &v[3*dim] : &v[2*dim];         // for eigenvector y

        if (purpose == "iram") {                                                 // v has m+1 cols
            for (MKL_INT j = 0; j <= mm; j++) vpt[j] = &v[j*dim];
        } else {                                                                 // v has only 2 or 3 cols
            for (MKL_INT j = 0; j <= mm; j++) vpt[j] = &v[(j%2)*dim];
        }

        std::vector<double> ritz(mm), s(mm * mm);                                // Ritz values and eigenvecs of Hess
        if (purpose.find("vec") != npos) hess_eigen(hessenberg, maxit, mm, "sr", ritz, s);

        assert(std::abs(blas_nrm2(dim, vpt[k], 1) - 1.0) < lanczos_precision);        // v[k] should be normalized
        if (k == 0) {                                                            // prepare 2 vectors to start
            hessenberg[0] = 0.0;
            for (MKL_INT l = 0; l < dim; l++) vpt[1][l] = zero;                  // v[1] = 0
            mat.MultMv2(vpt[0], vpt[1]);                                         // v[1] = H * v[0] + v[1]
            if (purpose == "iram" || purpose.find("val") != npos || purpose == "dnmcs") {
                hessenberg[maxit] = std::real(blas_dotc(dim, vpt[0], 1, vpt[1], 1));  // a[0] = (v[0], v[1])
            } else if (purpose.find("vec") != npos) {
                assert(std::abs(hessenberg[maxit] - std::real(blas_dotc(dim, vpt[0], 1, vpt[1], 1))) < lanczos_precision);
            } else {
                assert(false);
            }
            blas_axpy(dim, -hessenberg[maxit], vpt[0], 1, vpt[1], 1);                 // v[1] = v[1] - a[0] * v[0]
            if (purpose == "iram" || purpose.find("val") != npos || purpose == "dnmcs") {
                hessenberg[1] = blas_nrm2(dim, vpt[1], 1);                            // b[1] = || v[1] ||
            } else if (purpose.find("vec") != npos) {
                assert(std::abs(hessenberg[1] - blas_nrm2(dim, vpt[1], 1)) < lanczos_precision);
            } else {
                assert(false);
            }
            blas_scal(dim, 1.0 / hessenberg[1], vpt[1], 1);                           // v[1] = v[1] / b[1]
            m = ++k;
            --np;
            if (purpose.find("vec") != npos) blas_axpy(dim, s[m], vpt[m], 1, ypt, 1); // y += s[m] * v[m]
            ckpt_lanczos_update(m, maxit, dim, cnt_accuE0, accuracy, theta0_prev, theta1_prev, v, hessenberg, purpose);
        }

        do {                                                                     // while m < mm
            m++;
            for (MKL_INT l = 0; l < dim; l++)
                vpt[m][l] = -hessenberg[m-1] * vpt[m-2][l];                      // v[m] = -b[m-1] * v[m-2]
            mat.MultMv2(vpt[m-1], vpt[m]);                                       // v[m] = H * v[m-1] + v[m]

            if (purpose == "iram" || purpose.find("val") != npos || purpose == "dnmcs") {
                hessenberg[maxit+m-1] = std::real(blas_dotc(dim, vpt[m-1], 1, vpt[m], 1)); // a[m-1] = (v[m-1], v[m])
            } else if (purpose.find("vec") != npos) {
                assert(std::abs(hessenberg[maxit+m-1] - std::real(blas_dotc(dim, vpt[m-1], 1, vpt[m], 1))) < lanczos_precision);
            } else {
                assert(false);
            }
            blas_axpy(dim, -hessenberg[maxit+m-1], vpt[m-1], 1, vpt[m], 1);           // v[m] = v[m] - a[m-1] * v[m-1]
            if (purpose == "iram" || purpose.find("val") != npos || purpose == "dnmcs") {
                hessenberg[m] = blas_nrm2(dim, vpt[m], 1);                            // b[m] = || v[m] ||
            } else if (purpose.find("vec") != npos) {
                assert(std::abs(hessenberg[m] - blas_nrm2(dim, vpt[m], 1)) < lanczos_precision);
            } else {
                assert(false);
            }
            blas_scal(dim, 1.0 / hessenberg[m], vpt[m], 1);                           // v[m] = v[m] / b[m]

            if (std::abs(hessenberg[m]) < lanczos_precision) break;

            if (purpose.find("val1") != npos || purpose.find("vec1") != npos) {  // re-orthogonalization again phi0
                auto temp = blas_dotc(dim, phipt, 1, vpt[m], 1);
                if (std::abs(temp) > lanczos_precision) {
                    std::cout << "-" << std::flush;
                    blas_axpy(dim, -temp, phipt, 1, vpt[m], 1);
                    double rnorm = blas_nrm2(dim, vpt[m], 1);
                    blas_scal(dim, 1.0 / rnorm, vpt[m], 1);
                }
            }

            if (purpose.find("val") != npos) {
                hess_eigen(hessenberg, maxit, m, "sr", ritz, s);                  // calculate {theta, s}
                if (m > 3) {
                    accuracy = std::abs(hessenberg[m] * s[m-1]);
                    double accu_E0  = std::abs((ritz[0] - theta0_prev) / ritz[0]);
                    double accu_E1  = std::abs((ritz[1] - theta1_prev) / ritz[1]);
                    log_Lanczos_srval(m, ritz, hessenberg, maxit, accuracy, accu_E0, accu_E1, "log_Lanczos_"+purpose+".txt");
                    if (accu_E0 < lanczos_precision) {
                        cnt_accuE0++;
                    } else {
                        cnt_accuE0 = 0;
                    }
                    if ( cnt_accuE0 > 15 && accuracy < lanczos_precision)
                    {
                        ckpt_lanczos_update(m, maxit, dim, cnt_accuE0, accuracy, theta0_prev, theta1_prev, v, hessenberg, purpose);
                        break;
                    }
                }
                theta0_prev = ritz[0];
                theta1_prev = ritz[1];
            }
            if (purpose.find("vec") != npos) blas_axpy(dim, s[m], vpt[m], 1, ypt, 1); // y += s[m] * v[m]

            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // naive re-orthogonalization, change checking criteria and replace with DGKS later
            if (purpose == "iram") {
                for (MKL_INT l = 0; l < m-1; l++) {
                    auto q = blas_dotc(dim, vpt[l], 1, vpt[m], 1);
                    double qabs = std::abs(q);
                    if (qabs > lanczos_precision) {
                        blas_axpy(dim, -q, vpt[l], 1, vpt[m], 1);
                        blas_scal(dim, 1.0 / std::sqrt(1.0 - qabs * qabs), vpt[m], 1);
                    }
                }
            }
            ckpt_lanczos_update(m, maxit, dim, cnt_accuE0, accuracy, theta0_prev, theta1_prev, v, hessenberg, purpose);
        } while (m < mm);
        std::cout << std::endl;
    }
    template void lanczos(MKL_INT k, MKL_INT np, const MKL_INT &maxit, MKL_INT &m, const MKL_INT &dim,
                          const csr_mat<double> &mat, double v[],
                          double hessenberg[], const std::string &purpose);
    template void lanczos(MKL_INT k, MKL_INT np, const MKL_INT &maxit, MKL_INT &m, const MKL_INT &dim,
                          const csr_mat<std::complex<double>> &mat, std::complex<double> v[],
                          double hessenberg[], const std::string &purpose);
    template void lanczos(MKL_INT k, MKL_INT np, const MKL_INT &maxit, MKL_INT &m, const MKL_INT &dim,
                          const model<std::complex<double>> &mat, std::complex<double> v[],
                          double hessenberg[], const std::string &purpose);
//    template void lanczos(MKL_INT k, MKL_INT np, MKL_INT &mm, const MKL_INT &dim,
//                          const model<double> &mat, double v[],
//                          double hessenberg[], const MKL_INT &ldh, const std::string &purpose);


    template <typename T, typename MAT>
    void eigenvec_CG(const MKL_INT &dim, const MKL_INT &maxit, MKL_INT &m,
                     const MAT &mat, const T &E0, double &accu,
                     T v[], T r[], T p[], T pp[])
    {
        ckpt_CG_init(m, maxit, dim, v, r, p);
        assert(m >= 0 && m < maxit);
        if (m == 0) {
            accu = 0.0;
        } else {
            accu = blas_nrm2(dim, r, 1);
        }

        while (m < maxit) {
            if (accu < lanczos_precision) {
                double rnorm = blas_nrm2(dim, v, 1);
                if (m == 0 || std::abs(rnorm - 1.0) > lanczos_precision) {       // re-normalize and restart
                    std::cout << "1" << std::flush;
                    blas_scal(dim, 1.0/rnorm, v, 1);
                    for (MKL_INT j = 0; j < dim; j++) r[j] = 0.0;
                    mat.MultMv2(v,r);
                    blas_scal(dim, -static_cast<T>(1.0), r, 1);
                    blas_axpy(dim, E0, v, 1, r, 1);                                   // r = (E0 - H) * v
                    blas_copy(dim, r, 1, p, 1);                                       // p = r
                    accu = blas_nrm2(dim, r, 1);
                    m++;

                    std::ofstream fout("log_CG.txt", std::ios::out | std::ios::app);
                    fout << std::setprecision(10);
                    fout << std::setw(20) << m << std::setw(20) << accu << std::endl;
                    fout.close();
                    ckpt_CG_update(m, dim, v, r, p);


                    if (accu < lanczos_precision) break;
                } else {
                    break;
                }
            } else {
                blas_copy(dim, p, 1, pp, 1);
                blas_scal(dim, machine_prec - E0, pp, 1);
                mat.MultMv2(p,pp);                                               // pp[m]    = (H - E0) * p[m]
                T delta = blas_dotc(dim, p, 1, pp, 1);                                // delta[m] = (p[m], pp[m])
                T alpha = accu * accu / delta;                                   // alpha[m] = gamma[m]^2 / delta[m]
                blas_axpy(dim,  alpha,  p, 1, v, 1);                                  // v[m+1]   = v[m] + alpha[m] * p[m]
                blas_axpy(dim, -alpha, pp, 1, r, 1);                                  // r[m+1]   = r[m] - alpha[m] * pp[m]
                double beta = blas_nrm2(dim, r, 1) / accu;                            // beta[m]  = gamma[m+1] / gamma[m]
                blas_scal(dim, static_cast<T>(beta*beta), p, 1);
                blas_axpy(dim, static_cast<T>(1.0), r, 1, p, 1);                      // p[m+1]   = r[m+1] + beta^2 * p[m]
                accu *= beta;                                                    // gamma[m+1] = beta[m] * gamma[m]
                m++;

                ckpt_CG_update(m, dim, v, r, p);
                std::ofstream fout("log_CG.txt", std::ios::out | std::ios::app);
                fout << std::setprecision(10);
                fout << std::setw(20) << m << std::setw(20) << accu << std::endl;
                fout.close();
            }
        }
        std::cout << std::endl;
    }
    template void eigenvec_CG(const MKL_INT &dim, const MKL_INT &maxit, MKL_INT &m,
                              const csr_mat<double> &mat, const double &E0, double &accu,
                              double v[], double r[], double p[], double pp[]);
    template void eigenvec_CG(const MKL_INT &dim, const MKL_INT &maxit, MKL_INT &m,
                              const csr_mat<std::complex<double>> &mat, const std::complex<double> &E0, double &accu,
                              std::complex<double> v[], std::complex<double> r[],
                              std::complex<double> p[], std::complex<double> pp[]);
    template void eigenvec_CG(const MKL_INT &dim, const MKL_INT &maxit, MKL_INT &m,
                              const model<std::complex<double>> &mat, const std::complex<double> &E0, double &accu,
                              std::complex<double> v[], std::complex<double> r[],
                              std::complex<double> p[], std::complex<double> pp[]);


    void hess_eigen(const double hessenberg[], const MKL_INT &maxit, const MKL_INT &m,
                    const std::string &order, std::vector<double> &ritz, std::vector<double> &s)
    {
        assert(m > 0 && m < maxit);
        ritz.resize(m);
        s.resize(m*m);
        std::vector<double> b(m);
        std::vector<double> eigenvecs(m*m);

        blas_copy(m, hessenberg + maxit, 1, ritz.data(), 1);                          // ritz = a
        blas_copy(m, hessenberg,         1, b.data(),    1);                          // b
        // ritz to be rewritten by eigenvalues in ascending order
        auto info = LAPACKE_dstedc(LAPACK_COL_MAJOR, 'I', m, ritz.data(), b.data() + 1, eigenvecs.data(), m);
        assert(info == 0);

        std::vector<std::pair<double, MKL_INT>> eigenvals(m);
        for (MKL_INT j = 0; j < m; j++) {
            eigenvals[j].first = ritz[j];
            eigenvals[j].second = j;
        }
        if (order == "SR" || order == "sr" || order == "SA" || order == "sa") {  // smallest real part
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return a.first < b.first; });
        } else if (order == "LR" || order == "lr" || order == "LA" || order == "la") { // largest real part
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return b.first < a.first; });
        } else if (order == "SM" || order == "sm") {                             // smallest magnitude
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return std::abs(a.first) < std::abs(b.first); });
        } else if (order == "LM" || order == "lm") {                             // largest magnitude
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return std::abs(b.first) < std::abs(a.first); });
        }
        for (MKL_INT j = 0; j < m; j++) ritz[j] = eigenvals[j].first;
        for (MKL_INT j = 0; j < m; j++) blas_copy(m, eigenvecs.data() + m * eigenvals[j].second, 1, s.data() + m * j, 1);
    }

    // interface to arpack-ng
    template <typename MAT>
    void call_arpack(const a_int &N, const MAT &mat, double resid[],
                     const a_int &nev, const a_int &ncv, const a_int &maxit, a_int &nconv, a_int &niter,
                     const arpack::which &ritz_option, double eigenvals[], double eigenvecs[])
    {
        std::cout << "ARPACK info:" << std::endl;
        std::cout << "(nev, ncv)    = (" << nev << "," << ncv << ")" << std::endl;
        std::cout << "Max iteration = " << maxit << std::endl;
        std::cout << "Max Mat*Vec   = " << maxit * (ncv - nev) << std::endl;

        const auto bmat_option   = arpack::bmat::identity;
        const auto howmny_option = arpack::howmny::ritz_vectors;
        const double tol         = 0.0;  // when tol <= 0, it is reset to machine precision
        const double sigma       = 0.0;  // not referenced
        a_int const rvec         = 1;    // need eigenvectors

        a_int iparam[11], ipntr[11];
        iparam[0] = 1;      // ishift
        iparam[2] = maxit;  // on input: maxit; on output: actual iteration
        iparam[3] = 1;      // NB, only 1 allowed
        iparam[6] = 1;      // mode

        a_int lworkl = ncv * (ncv + 8);
        std::vector<double> V(ncv * N);
        std::vector<double> workd(3 * N);
        std::vector<double> workl(lworkl);
        std::vector<a_int> select(ncv);

        a_int info = 0;     // use random initial residual vector
        a_int ido = 0;
        do {
            arpack::saupd(ido, bmat_option, N, ritz_option, nev, tol, resid, ncv,
                          V.data(), N, iparam, ipntr, workd.data(), workl.data(), lworkl, info);
            mat.MultMv(&(workd[ipntr[0] - 1]), &(workd[ipntr[1] - 1]));
        } while (ido == 1 || ido == -1);
        if (info < 0) throw std::runtime_error("Error with saupd, info = " + std::to_string(info));
        niter = iparam[2];
        nconv = iparam[4];

        arpack::seupd(rvec, howmny_option, select.data(), eigenvals,
                      eigenvecs, N, sigma, bmat_option, N, ritz_option, nev, tol, resid, ncv,
                      V.data(), N, iparam, ipntr, workd.data(), workl.data(), lworkl, info);
        if (info < 0) throw std::runtime_error("Error with seupd, info = " + std::to_string(info));
    }

    template <typename MAT>
    void call_arpack(const a_int &N, const MAT &mat, std::complex<double> resid[],
                     const a_int &nev, const a_int &ncv, const a_int &maxit, a_int &nconv, a_int &niter,
                     const arpack::which &ritz_option, double eigenvals[], std::complex<double> eigenvecs[])
    {
        auto *eigenvals_copy = new std::complex<double>[nev];

        std::cout << "ARPACK info:" << std::endl;
        std::cout << "(nev, ncv)    = (" << nev << "," << ncv << ")" << std::endl;
        std::cout << "Max iteration = " << maxit << std::endl;
        std::cout << "Max Mat*Vec   = " << maxit * (ncv - nev) << std::endl;

        const auto bmat_option           = arpack::bmat::identity;
        const auto howmny_option         = arpack::howmny::ritz_vectors;
        const double tol                 = 0.0;  // when tol <= 0, it is reset to machine precision
        const std::complex<double> sigma = 0.0;  // not referenced
        a_int const rvec                 = 1;    // need eigenvectors

        a_int iparam[11], ipntr[14];
        iparam[0] = 1;      // ishift
        iparam[2] = maxit;  // on input: maxit; on output: actual iteration
        iparam[3] = 1;      // NB, only 1 allowed
        iparam[6] = 1;      // mode

        a_int lworkl = ncv * (3 * ncv + 5);
        std::vector<std::complex<double>> V(ncv * N);
        std::vector<std::complex<double>> workd(3 * N);
        std::vector<std::complex<double>> workl(lworkl);
        std::vector<std::complex<double>> workev(3 * ncv);
        std::vector<double> rwork(ncv);
        std::vector<a_int> select(ncv);

        a_int info = 0;     // use random initial residual vector
        a_int ido = 0;
        do {
            arpack::naupd(ido, bmat_option, N, ritz_option, nev, tol, resid, ncv,
                          V.data(), N, iparam, ipntr,
                          workd.data(), workl.data(), lworkl, rwork.data(), info);
            mat.MultMv(&(workd[ipntr[0] - 1]), &(workd[ipntr[1] - 1]));
        } while (ido == 1 || ido == -1);
        if (info < 0) throw std::runtime_error("Error with naupd, info = " + std::to_string(info));
        niter = iparam[2];
        nconv = iparam[4];

        arpack::neupd(rvec, howmny_option, select.data(), eigenvals_copy,
                      eigenvecs, N, sigma, workev.data(), bmat_option, N, ritz_option, nev, tol, resid, ncv,
                      V.data(), N, iparam, ipntr, workd.data(), workl.data(), lworkl, rwork.data(), info);
        if (info < 0) throw std::runtime_error("Error with neupd, info = " + std::to_string(info));

        for (a_int j = 0; j < nconv; j++) {
            eigenvals[j] = eigenvals_copy[j].real();
            if (std::abs(eigenvals_copy[j].imag()) > lanczos_precision) {
                std::cout << "Eigenvalue[" << j << "]: " << eigenvals_copy[j] << std::endl;
                throw std::runtime_error("eigenvalue should be real.");
            }
        }
        delete [] eigenvals_copy;
    }

    template <typename T, typename MAT>
    void iram(const MKL_INT &dim, MAT &mat, T v0[], const MKL_INT &nev, const MKL_INT &ncv,
              const MKL_INT &maxit, const std::string &order,
              MKL_INT &nconv, double eigenvals[], T eigenvecs[])
    {
        if (nev <= 0 || nev >= dim - 1) throw std::invalid_argument("0 < nev < N-1 should be satisfied.");

        if (maxit < 20) throw std::invalid_argument("maxit should not be smaller than 20!");
        std::string orderC(order);
        std::transform(orderC.begin(), orderC.end(), orderC.begin(), ::toupper);

        if (dim <= 30) {              // fall back to full diagonalization
            auto mat_dense = mat.to_dense();
            std::vector<double> eigenvals_all(dim);
            auto info = lapack_syevd_heevd(LAPACK_COL_MAJOR, 'V', 'U', dim, mat_dense.data(), dim, eigenvals_all.data());
            if (info != 0) throw std::runtime_error("heevd failed!");
            nconv = nev;
            std::vector<std::pair<double, MKL_INT>> eigenvals_copy(dim);
            for (MKL_INT j = 0; j < dim; j++) {
                eigenvals_copy[j].first = eigenvals_all[j];
                eigenvals_copy[j].second = j;
            }
            if (orderC == "SR" || orderC == "SA") {                               // smallest real part
                std::sort(eigenvals_copy.begin(), eigenvals_copy.end(),
                          [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return a.first < b.first; });
            } else if (orderC == "LR" || orderC == "LA") {                        // largest real part
                std::sort(eigenvals_copy.begin(), eigenvals_copy.end(),
                          [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return b.first < a.first; });
            } else if (orderC == "SM") {                                         // smallest magnitude
                std::sort(eigenvals_copy.begin(), eigenvals_copy.end(),
                          [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return std::abs(a.first) < std::abs(b.first); });
            } else if (orderC == "LM") {                                         // largest magnitude
                std::sort(eigenvals_copy.begin(), eigenvals_copy.end(),
                          [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return std::abs(b.first) < std::abs(a.first); });
            } else {
                throw std::invalid_argument("Invalid argument orderC.");
            }
            for (MKL_INT j = 0; j < nconv; j++)
                eigenvals[j] = eigenvals_copy[j].first;
            for (MKL_INT j = 0; j < nconv; j++) {
                std::cout << "j = " << j << ", E_j = " << eigenvals[j] << std::endl;
            }
            // sort the eigenvecs
            for (MKL_INT j = 0; j < nconv; j++) {
                blas_copy(dim, mat_dense.data() + dim * eigenvals_copy[j].second, 1, eigenvecs + dim * j, 1);
            }
        } else {
            arpack::which ritz_option;
            if (orderC == "LA") {
                ritz_option = arpack::which::largest_algebraic;
            } else if (orderC == "SA") {
                ritz_option = arpack::which::smallest_algebraic;
            } else if (orderC == "LM") {
                ritz_option = arpack::which::largest_magnitude;
            } else if (orderC == "SM") {
                ritz_option = arpack::which::smallest_magnitude;
            } else if (orderC == "LR") {
                ritz_option = arpack::which::largest_real;
            } else if (orderC == "SR") {
                ritz_option = arpack::which::smallest_real;
            } else if (orderC == "BE") {
                ritz_option = arpack::which::both_ends;
            } else {
                throw std::invalid_argument("Invalid argument orderC.");
            }

            a_int nconv_arpack, niter;
            call_arpack(static_cast<a_int>(dim), mat, v0, static_cast<a_int>(nev), static_cast<a_int>(ncv),
                        static_cast<a_int>(maxit), nconv_arpack, niter, ritz_option, eigenvals, eigenvecs);
            if (nconv_arpack <= 0) throw std::runtime_error("nconv == 0...");
            nconv = static_cast<MKL_INT>(nconv_arpack);
            std::cout << std::endl << "(nev,ncv,nconv) = (" << nev << "," << ncv << "," << nconv << ")" << std::endl;
            std::cout << "Number of implicit restarts: " << niter << std::endl;
            auto comp = [&orderC](const double &a, const double &b)
            {
                if (orderC == "SR" || orderC == "SA") {
                    return a < b;
                } else if (orderC == "LR" || orderC == "LA") {
                    return b < a;
                } else if (orderC == "SM") {
                    return std::abs(a) < std::abs(b);
                } else {
                    assert(orderC == "LM");
                    return std::abs(b) < std::abs(a);
                }
            };
            // bubble sort the eigenvalues and eigenvecs
            using std::swap;
            for (MKL_INT j = 1; j < nconv; j++) {
                bool sorted = true;
                for (MKL_INT i = 0; i < nconv - j; i++) {
                    if (comp(eigenvals[i + 1], eigenvals[i])) {
                        swap(eigenvals[i], eigenvals[i + 1]);
                        vec_swap(dim, eigenvecs + i * dim, eigenvecs + (i + 1) * dim);
                        sorted = false;
                    }
                }
                if (sorted) break;
            }
            for (MKL_INT j = 0; j < nconv; j++) {
                std::cout << "E_" << j << " = " << eigenvals[j] << std::endl;
            }
            std::cout << "Caution: IRAM may miss a few degenerate eigenstates!" << std::endl;
            std::cout << "(in these cases, try a different set of {nev, ncv} may help finding the missing eigenstates)"
                      << std::endl;
        }
    }

    // Explicit instantiation
/*
    template void iram(const MKL_INT &dim, csr_mat<double> &mat, double v0[], const MKL_INT &nev, const MKL_INT &ncv,
                       const MKL_INT &maxit, const std::string &order,
                       MKL_INT &nconv, double eigenvals[], double eigenvecs[]);
    template void iram(const MKL_INT &dim, model<double> &mat, double v0[], const MKL_INT &nev, const MKL_INT &ncv,
                       const MKL_INT &maxit, const std::string &order,
                       MKL_INT &nconv, double eigenvals[], double eigenvecs[]);
*/
    template void iram(const MKL_INT &dim, csr_mat<std::complex<double>> &mat, std::complex<double> v0[],
                       const MKL_INT &nev, const MKL_INT &ncv, const MKL_INT &maxit, const std::string &order,
                       MKL_INT &nconv, double eigenvals[], std::complex<double> eigenvecs[]);
    template void iram(const MKL_INT &dim, model<std::complex<double>> &mat, std::complex<double> v0[],
                       const MKL_INT &nev, const MKL_INT &ncv, const MKL_INT &maxit, const std::string &order,
                       MKL_INT &nconv, double eigenvals[], std::complex<double> eigenvecs[]);

}
