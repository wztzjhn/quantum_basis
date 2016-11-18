#ifndef SPARSE_H
#define SPARSE_H
#include <vector>
#include <list>

// Note: sparse matrices in this code are using zero-based convention

// By default, all diagonal elements are stored, even if they are zero (to be compatible with pardiso)

static const double sparse_precision = 1e-20;

template <typename T> struct lil_mat_elem {
    int col;
    T val;
};

template <typename T> class lil_mat {
public:
    // default constructor
    lil_mat() = default;
    
    // constructor with the Hilbert space dimension
    lil_mat(const size_t &n) : mat(std::vector<std::list<lil_mat_elem<T>>>(n, std::list<lil_mat_elem<T>>(1)))
    {
        for (decltype(mat.size()) i = 0; i < n; i++) {
            mat[i].front().col = i;
            mat[i].front().val = 0.0;
        }
    }
    
    // add one element
    void add(const int &row, const int &col, const T &val);

    
    // explicitly destroy, free space
    void destroy()
    {
        mat.clear();
        mat.shrink_to_fit();
    }
    
    
    // destructor
    ~lil_mat() {};
    
    // print
    
private:
    size_t nnz;    // number of non-zero entries
    std::vector<std::list<lil_mat_elem<T>>> mat;
};


template <typename T> class csr_mat {
public:
    
private:
    size_t nnz;    // number of non-zero entries
    bool sym;      // if storing only upper triangle
    T *val;
    int *col;
    int *row_prt;
    
};


#endif
