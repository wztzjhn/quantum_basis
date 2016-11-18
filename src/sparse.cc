#include <complex>
#include <iostream>
#include <cassert>
#include "sparse.h"



// --------- implementation of class lil (list of list sparse matrix data structure) ----------
template <typename T>
void lil_mat<T>::add(const MKL_INT &row, const MKL_INT &col, const T &val)
{
    assert(row >= 0 && row < mat.size() && col >=0 && col < mat.size());
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
    std::cout << "nnz = " << nnz << std::endl;
//    for (decltype(mat.size()) i = 0; i < mat.size(); i++) {
//        std::cout << "--------------------------------" << std::endl;
//        std::cout << "Row " << i << ": " << std::endl;
//        for (auto &ele : mat[i]) {
//            std::cout << "col " << ele.col << ", val = " << ele.val << std::endl;
//        }
//    }
}


template <typename T>
csr_mat<T>::csr_mat(const lil_mat<T> &old, const bool &sym_) : dim(old.dimension()), nnz(old.num_nonzero()), sym(sym_)
{
    assert(old.num_nonzero()>0);
    std::cout << "Converting LIL to CSR: " << std::endl;
    std::cout << "# of Row and col:      " << dim << std::endl;
    std::cout << "# of nonzero elements: " << nnz << std::endl;
    std::cout << "# of all elements:     " << static_cast<long long>(dim) * static_cast<long long>(dim) << std::endl;
    std::cout << "Sparsity:              " << static_cast<double>(nnz) / static_cast<double>(dim) / static_cast<double>(dim) << std::endl;
    std::cout << "Matrix usage:          " << (sym?"Upper triangle":"Full") << std::endl;
    //val = new T[nnz];
}

// Explicit instantiation
template struct lil_mat_elem<double>;
template struct lil_mat_elem<std::complex<double>>;

template class lil_mat<double>;
template class lil_mat<std::complex<double>>;

template class csr_mat<double>;
template class csr_mat<std::complex<double>>;
