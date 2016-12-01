#ifndef LANCZOS_H
#define LANCZOS_H
#include <vector>
#include "mkl_interface.h"
#include "sparse.h"

// Note: sparse matrices in this code are using zero-based convention

// By default, all diagonal elements are stored, even if they are zero (to be compatible with pardiso)

static const double lanczos_precision = 1e-14;

// v of length k+1, hessenberg matrix of dim k (k-step Lanczos)
template <typename T>
void lanczos(const csr_mat<T> &mat, std::vector<std::vector<T>> &v, double hessenberg[]);
//void lanczos(void (*MultMv)(const std::vector<T> &, std::vector<T> &), std::vector<std::vector<T>> &v, double hessenberg[]);


#endif
