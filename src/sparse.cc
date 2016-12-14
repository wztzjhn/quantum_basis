#include <iostream>
#include <iomanip>
#include "qbasis.h"

namespace qbasis {
    // --------- implementation of class lil (list of list sparse matrix data structure) ----------
    template <typename T>
    void lil_mat<T>::add(const MKL_INT &row, const MKL_INT &col, const T &val)
    {
        assert(row >= 0 && row < static_cast<MKL_INT>(mat.size()) &&
               col >=0 && col < static_cast<MKL_INT>(mat.size()));
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
    void lil_mat<T>::prt() const
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
            assert(! old.mat[i].empty());                                        // at least storing the diagonal element
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
        
        if (! sym) {                                                             // check if matrix is hermitian
            for (decltype(dim) row = 0; row < dim; row++) {
                for (decltype(dim) j = ia[row]; j < ia[row+1]; j++) {            // check for each element in current row
                    auto col = ja[j];
                    if(row == col) continue;
                    auto i = ia[col];
                    while (i < ia[col+1] && ja[i] != row) i++;                   // until it's conjugate found
                    if (i == ia[col+1] || std::abs(val[j] - std::conj(val[i])) > sparse_precision) {
                        std::cout << "Hermitian check failed!!!" << std::endl;
                        std::cout << "(row, col)    = (" << row << ", " << col << ")" << std::endl;
                        std::cout << "mat(row, col) = " << val[j] << std::endl;
                        if (i == ia[col+1]) {
                            std::cout << "mat(col, row) NOT found!" << std::endl;
                        } else {
                            std::cout << "mat(col, row) = " << val[i] << std::endl;
                        }
                        std::exit(99);
                    }
                }
            }
        }
    }

    
    template <typename T>
    void csr_mat<T>::MultMv(const T *x, T *y) const
    {
        assert(val != nullptr && ja != nullptr && ia != nullptr);
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
    template <typename T>
    void csr_mat<T>::MultMv(T *x, T *y)
    {
        assert(val != nullptr && ja != nullptr && ia != nullptr);
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
    void csr_mat<T>::prt() const
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

    
    // Explicit instantiation
    template struct lil_mat_elem<double>;
    template struct lil_mat_elem<std::complex<double>>;

    template class lil_mat<double>;
    template class lil_mat<std::complex<double>>;

    template class csr_mat<double>;
    template class csr_mat<std::complex<double>>;

    //template void csrXvec(const csr_mat<double>&, const std::vector<double>&, std::vector<double>&);
    //template void csrXvec(const csr_mat<std::complex<double>>&, const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);


}
