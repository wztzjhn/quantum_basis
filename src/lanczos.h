#ifndef LANCZOS_H
#define LANCZOS_H
#include <vector>
#include "mkl_interface.h"
#include "sparse.h"

// Note: sparse matrices in this code are using zero-based convention

// By default, all diagonal elements are stored, even if they are zero (to be compatible with pardiso, if used in future)

static const double lanczos_precision = 1e-14;

// v of length k+1, hessenberg matrix of size k*k (k-step Lanczos)
// after decomposition, mat * v[0:k-1] = v[0:k-1] * hessenberg + beta_k * v[k] * e_k^T,
// where e_k has only one nonzero element: e[0:k-2] == 0, e[k-1] = 1
template <typename T>
void lanczos(const csr_mat<T> &mat, std::vector<std::vector<T>> &v, double hessenberg[], double &beta_k);
//void lanczos(void (*MultMv)(const std::vector<T> &, std::vector<T> &), std::vector<std::vector<T>> &v, double hessenberg[]);


// transform from band storage to general storage
void hess2matform(double hessenberg[], double mat[], const MKL_INT &k);
void hess2matform(double hessenberg[], std::complex<double> mat[], const MKL_INT &k);
#endif
