#ifndef LANCZOS_H
#define LANCZOS_H
#include <vector>
#include <string>
#include <complex>
#include <iostream>
#include <cassert>
#include <utility>
#include <algorithm>
#include "mkl_interface.h"
#include "sparse.h"


namespace qbasis {
    // Note: sparse matrices in this code are using zero-based convention
    
    // By default, all diagonal elements are stored, even if they are zero (to be compatible with pardiso, if used in future)
    
    static const double lanczos_precision = 1e-12;
    
    
    // m = k + np step of Lanczos
    // v of length m+1, hessenberg matrix of size m*m (m-step Lanczos)
    // after decomposition, mat * v[0:m-1] = v[0:m-1] * hessenberg + rnorm * resid * e_m^T,
    // where e_m has only one nonzero element: e[0:m-2] == 0, e[m-1] = 1
    
    // on entry, assuming k steps of Lanczos already performed:
    // v_0, ..., v_{k-1} stored in v, v{k} stored in resid
    // alpha_0, ..., alpha_{k-1} in hessenberg matrix
    // beta_1,  ..., beta_{k-1} in hessenberg matrix, beta_k as rnorm
    // if on entry k==0, then beta_k=rnorm=0, v_0=resid
    
    // ldh: leading dimension of hessenberg
    // alpha[j] = hessenberg[j+ldh], diagonal of hessenberg matrix
    // beta[j]  = hessenberg[j]
    //  a[0]  b[1]      -> note: beta[0] not used
    //  b[1]  a[1]  b[2]
    //        b[2]  a[2]  b[3]
    //              b[3]  a[3] b[4]
    //                    ..  ..  ..    b[k-1]
    //                          b[k-1]  a[k-1]
    template <typename T>
    void lanczos(MKL_INT k, MKL_INT np, const csr_mat<T> &mat, double &rnorm, T resid[],
                 T v[], double hessenberg[], const MKL_INT &ldh, const bool &MemoSteps = true);
    
    // if possible, add a block Arnoldi version here
    
    // transform from band storage to general storage
    template <typename T>
    void hess2matform(const double hessenberg[], T mat[], const MKL_INT &m, const MKL_INT &ldh);
    
    // compute eigenvalues (and optionally eigenvectors, stored in s) of hessenberg matrix
    // on entry, hessenberg and s should have the same leading dimension: ldh
    // order = "sm", "lm", "sr", "lr", where 's': small, 'l': large, 'm': magnitude, 'r': real part
    void select_shifts(const double hessenberg[], const MKL_INT &ldh, const MKL_INT &m,
                       const std::string &order, double ritz[], double s[] = nullptr);
    
    // --------------------------
    // ideally, here we should use the bulge-chasing algorithm; for this moment, we simply use the less efficient brute force QR factorization
    // --------------------------
    // QR factorization of hessenberg matrix, using np selected eigenvalues from ritz
    // [H, Q] = QR(H, shift1, shift2, ..., shift_np)
    // \tilde{H} = Q_np^T ... Q_1^T H Q_1 ... Q_np
    // \tilde{V} = V Q
    template <typename T>
    void perform_shifts(const MKL_INT &dim, const MKL_INT &m, const MKL_INT &np, const double shift[],
                        double &rnorm, T resid[], T v[], double hessenberg[], const MKL_INT &ldh,
                        double Q[], const MKL_INT &ldq);
    
    // implicitly restarted Arnoldi method
    // nev: number of eigenvalues needed
    // ncv: length of each individual lanczos process
    // 2 < nev + 2 <= ncv
    // when not using arpack++, we can modify the property of mat to be const
    template <typename T>
    void iram(csr_mat<T> &mat, T v0[], const MKL_INT &nev, const MKL_INT &ncv, MKL_INT &nconv,
              const std::string &order, double eigenvals[], T eigenvecs[], const bool &use_arpack = true);
}

#endif
