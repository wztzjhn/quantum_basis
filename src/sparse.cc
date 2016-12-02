#include <complex>
#include <iostream>
#include <iomanip>
#include <cassert>
#include "sparse.h"



// --------- implementation of class lil (list of list sparse matrix data structure) ----------
template <typename T>
void lil_mat<T>::add(const MKL_INT &row, const MKL_INT &col, const T &val)
{
    assert(row >= 0 && row < mat.size() && col >=0 && col < mat.size());
    assert(sym ? row <= col : true);
    if (std::abs(val) < sparse_precision) return;
    auto it_prev = mat[row].before_begin();
    auto it_curr = mat[row].begin();
    while (it_curr != mat[row].end() && it_curr->col < col) {
        it_prev = it_curr;
        ++it_curr;
    }
    if (it_curr != mat[row].end() && it_curr->col == col) {
        it_curr->val += val;
        if (row != col && std::abs(it_curr->val) < sparse_precision) {
            mat[row].erase_after(it_prev);
            --nnz;
        }
    } else {
        ++nnz;
        mat[row].insert_after(it_prev,{val,col});
    }
}

template <typename T>
void lil_mat<T>::prt()
{
    std::cout << "dim = " << dim << std::endl;
    std::cout << "nnz = " << nnz << std::endl;
    std::cout << (sym?"Upper triangle":"Full matrix") << std::endl;
    for (decltype(mat.size()) i = 0; i < mat.size(); i++) {
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Row " << i << ": " << std::endl;
        for (auto &ele : mat[i]) {
            std::cout << "col " << ele.col << ", val = " << ele.val << std::endl;
        }
    }
}


template <typename T>
csr_mat<T>::csr_mat(const lil_mat<T> &old) : dim(old.dim), nnz(old.nnz), sym(old.sym)
{
    assert(old.nnz>0);
    std::cout << "Converting LIL to CSR: " << std::endl;
    std::cout << "# of Row and col:      " << dim << std::endl;
    std::cout << "# of nonzero elements: " << nnz << std::endl;
    auto capacity= sym ? static_cast<long long>(dim+1) * static_cast<long long>(dim) / 2 : static_cast<long long>(dim) * static_cast<long long>(dim);
    std::cout << "# of all elements:     " << capacity << std::endl;
    std::cout << "Sparsity:              " << static_cast<double>(nnz) / capacity << std::endl;
    std::cout << "Matrix usage:          " << (sym?"Upper triangle":"Full") << std::endl;
    val = new T[nnz];
    ja = new MKL_INT[nnz];
    ia = new MKL_INT[dim+1];
    MKL_INT counts = 0;
    for (decltype(dim) i = 0; i < dim; i++) {
        assert(! old.mat[i].empty()); // at least storing the diagonal element
        ia[i] = counts;
        auto it = old.mat[i].begin();
        while (it != old.mat[i].end()) {
            assert(old.sym ? i <= it->col : true);
            ja[counts] = it->col;
            val[counts] = it->val;
            ++it;
            ++counts;
        }
    }
    assert(counts == nnz);
    ia[dim] = counts;
}

template <typename T>
void csr_mat<T>::MultMv(const T *x, T *y) const
{
    if (sym) {
        char matdescar[7] = "HUNC";
        T zero = static_cast<T>(0.0);
        T one  = static_cast<T>(1.0);
        for (MKL_INT j = 0; j < dim; j++) y[j] = zero; // required by mkl_csrmv
        mkl_csrmv('n', dim, dim, one, matdescar,
                  val, ja, ia, ia + 1, x, one, y);
    } else {
        csrgemv('n', dim, val, ia, ja, x, y);
    }
}

// need fix ldb and ldc to make this subroutine work
//template <typename T>
//void csr_mat<T>::MultMm(const T *x, T *y, MKL_INT n) const
//{
//    T zero = static_cast<T>(0.0);
//    T one  = static_cast<T>(1.0);
//    for (MKL_INT j = 0; j < dim * n; j++) y[j] = zero; // required by mkl_csrmm
//    char matdescar[7] = "HUNC";
//    matdescar[0] = sym ? 'H' : 'G';
//    mkl_csrmm('n', dim, n, dim, one, matdescar,
//              val, ja, ia, ia + 1, x, n, zero, y, n);
//}

template <typename T>
void csr_mat<T>::prt()
{
    std::cout << "dim = " << dim << std::endl;
    std::cout << "nnz = " << nnz << std::endl;
    std::cout << (sym?"Upper triangle":"Full matrix") << std::endl;
    for (MKL_INT i = 0; i < nnz; i++) std::cout << std::setw(8) << val[i];
    std::cout << std::endl;
    for (MKL_INT i = 0; i < nnz; i++) std::cout << std::setw(8) << ja[i];
    std::cout << std::endl;
    for (MKL_INT i = 0; i <= dim; i++) std::cout << std::setw(8) << ia[i];
    std::cout << std::endl;
}

//template <typename T>
//void csrXvec(const csr_mat<T> &mat, const std::vector<T> &x, std::vector<T> &y)
//{
//    assert(mat.dim == x.size() && x.size() == y.size());
//    if (mat.sym) {
//        char matdescar[7] = "HUNC";
//        // later can be optimized
//        for (MKL_INT j = 0; j < y.size(); j++) y[j] = 0.0; // required by mkl_csrmv
//        mkl_csrmv('n', mat.dim, mat.dim, static_cast<T>(1.0), matdescar,
//                  mat.val, mat.ja, mat.ia, mat.ia + 1, x.data(), static_cast<T>(1.0), y.data());
//    } else {
//        csrgemv('n', mat.dim, mat.val, mat.ia, mat.ja, x.data(), y.data());
//    }
//}


// Explicit instantiation
template struct lil_mat_elem<double>;
template struct lil_mat_elem<std::complex<double>>;

template class lil_mat<double>;
template class lil_mat<std::complex<double>>;

template class csr_mat<double>;
template class csr_mat<std::complex<double>>;

//template void csrXvec(const csr_mat<double>&, const std::vector<double>&, std::vector<double>&);
//template void csrXvec(const csr_mat<std::complex<double>>&, const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);

