#ifndef SPARSE_H
#define SPARSE_H
#include <vector>
#include <forward_list>
#include "mkl_interface.h"

// Note: sparse matrices in this code are using zero-based convention

// By default, all diagonal elements are stored, even if they are zero (to be compatible with pardiso)

static const double sparse_precision = 1e-15;

template <typename T> struct lil_mat_elem {
    T val;
    MKL_INT col;
};

template <typename T> class lil_mat {
public:
    // default constructor
    lil_mat() = default;
    
    // constructor with the Hilbert space dimension
    lil_mat(const MKL_INT &n) : dim(n), nnz(n),
                                mat(std::vector<std::forward_list<lil_mat_elem<T>>>(n, std::forward_list<lil_mat_elem<T>>(1)))
    {
        mat.shrink_to_fit();
        for (MKL_INT i = 0; i < n; i++) {
            mat[i].front().col = i;
            mat[i].front().val = 0.0;
        }
    }
    
    // add one element
    void add(const MKL_INT &row, const MKL_INT &col, const T &val);

    
    // explicitly destroy, free space
    void destroy()
    {
        mat.clear();
        mat.shrink_to_fit();
    }
    
    // destructor
    ~lil_mat() {};
    
    MKL_INT dimension() const { return dim; }
    
    MKL_INT num_nonzero() const { return nnz; }
    
    // print
    void prt();
    
private:
    MKL_INT dim;    // dimension of the matrix
    MKL_INT nnz;    // number of non-zero entries
    std::vector<std::forward_list<lil_mat_elem<T>>> mat;
};


// 3-array form of csr sparse matrix format, zero baseds
template <typename T> class csr_mat {
public:
    // default constructor
    csr_mat() = default;
    
    // construcotr from a lil_mat, and if sym_ == true, use only the upper triangle
    csr_mat(const lil_mat<T> &old, const bool &sym_ = false);
    
    // destructor
    ~csr_mat()
    {
        if(val != nullptr) delete [] val;
        if(col != nullptr) delete [] col;
        if(row_ptr != nullptr) delete [] row_ptr;
    }
    
private:
    MKL_INT dim;
    MKL_INT nnz;    // number of non-zero entries
    bool sym;       // if storing only upper triangle
    T *val;
    MKL_INT *col;
    MKL_INT *row_ptr;
};


#endif
