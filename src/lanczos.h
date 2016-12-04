#ifndef LANCZOS_H
#define LANCZOS_H
#include <vector>
#include <string>
#include <complex>
#include <iostream>
#include <cassert>
#include <utility>
#include "mkl_interface.h"
#include "sparse.h"

// Note: sparse matrices in this code are using zero-based convention

// By default, all diagonal elements are stored, even if they are zero (to be compatible with pardiso, if used in future)

static const double lanczos_precision = 1e-14;

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
void lanczos(MKL_INT k, MKL_INT np, csr_mat<T> &mat, double &rnorm, T resid[],
             T v[], double hessenberg[], const MKL_INT &ldh);

// if possible, add a block Arnoldi version here

// transform from band storage to general storage
template <typename T>
void hess2matform(const double hessenberg[], T mat[], const MKL_INT &m, const MKL_INT &ldh);

// compute eigenvalues (and optionally eigenvectors, stored in s) of hessenberg matrix
// on entry, hessenberg and s should have the same leading dimension: ldh
// order = "sm", "lm", "sr", "lr", where 's': small, 'l': large, 'm': magnitude, 'r': real part
void select_shifts(const double hessenberg[], const MKL_INT &ldh, const MKL_INT &m,
                   const std::string &order, double ritz[], double s[] = nullptr);

#endif
