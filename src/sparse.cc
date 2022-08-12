#include <iostream>
#include <iomanip>

#include "qbasis.h"

// general function to perform matrix vector product in mkl, deprecated since MKL 2018.3
inline // double
void mkl_csrmv(const char transa, const MKL_INT m, const MKL_INT k, const double alpha, const char *matdescra,
               const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
               const double *x, const double beta, double *y) {
    mkl_dcsrmv(&transa, &m, &k, &alpha, matdescra, val, indx, pntrb, pntre, x, &beta, y);
}
inline // complex double
void mkl_csrmv(const char transa, const MKL_INT m, const MKL_INT k, const std::complex<double> alpha, const char *matdescra,
               const std::complex<double> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
               const std::complex<double> *x, const std::complex<double> beta, std::complex<double> *y) {
    mkl_zcsrmv(&transa, &m, &k, &alpha, matdescra, val, indx, pntrb, pntre, x, &beta, y);
}

namespace qbasis {
    // --------- implementation of class lil (list of list sparse matrix data structure) ----------
    template <typename T>
    lil_mat<T>::lil_mat(const MKL_INT &n, bool sym_) :
        dim(n), nnz(n), sym(sym_),
        mat(std::vector<std::forward_list<lil_mat_elem<T>>>(n, std::forward_list<lil_mat_elem<T>>(1)))
    {
        mat.shrink_to_fit();
        for (MKL_INT i = 0; i < n; i++) {
            mat[i].front().col = i;
            mat[i].front().val = 0.0;
        }
    }


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
                #pragma omp atomic
                --nnz;
            }
        } else {
            #pragma omp atomic
            ++nnz;
            mat[row].insert_after(it_prev,{val,col});
        }
    }

    template <typename T>
    void lil_mat<T>::destroy(const MKL_INT &row)
    {
        assert(row >= 0 && row < dim);
        mat[row].clear();
    }

    template <typename T>
    void lil_mat<T>::destroy()
    {
        mat.clear();
        mat.shrink_to_fit();
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

    // --------- implementation of class csr (compressed row sparse matrix data structure) ----------
    template <typename T>
    csr_mat<T>::csr_mat(const csr_mat<T> &old) :
        dim(old.dim), nnz(old.nnz), sym(old.sym)
    {
        if (nnz > 0) {
            val = new T[nnz];
            ja  = new MKL_INT[nnz];
            ia  = new MKL_INT[dim+1];
            for (MKL_INT j = 0; j < nnz; j++) {
                val[j] = old.val[j];
                ja[j]  = old.ja[j];
            }
            for (MKL_INT j = 0; j < dim + 1; j++) {
                ia[j]  = old.ia[j];
            }
        } else {
            val = nullptr;
            ja  = nullptr;
            ia  = nullptr;
        }
    }

    template <typename T>
    csr_mat<T>::csr_mat(csr_mat<T> &&old) noexcept :
        dim(old.dim), nnz(old.nnz), sym(old.sym),
        val(old.val), ja(old.ja), ia(old.ia)
    {
        old.val = nullptr;
        old.ja  = nullptr;
        old.ia  = nullptr;
    }

    template <typename T>
    void csr_mat<T>::destroy()
    {
        if(val != nullptr) {
            delete [] val;
            val = nullptr;
        }
        if(ja != nullptr){
            delete [] ja;
            ja = nullptr;
        }
        if(ia != nullptr){
            delete [] ia;
            ia = nullptr;
        }
    }

    template <typename T>
    csr_mat<T>::~csr_mat()
    {
        if(val != nullptr) {
            delete [] val;
            val = nullptr;
        }
        if(ja != nullptr) {
            delete [] ja;
            ja = nullptr;
        }
        if(ia != nullptr) {
            delete [] ia;
            ia = nullptr;
        }
    }

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

    template <typename T>
    csr_mat<T>::csr_mat(lil_mat<T> &old) : dim(old.dim), nnz(old.nnz), sym(old.sym)
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
            old.destroy(i);
        }
        assert(counts == nnz);
        ia[dim] = counts;
        old.destroy();

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
    void csr_mat<T>::MultMv2(const T *x, T *y) const
    {
        std::cout << "*" << std::flush;
        assert(val != nullptr && ja != nullptr && ia != nullptr);
        char matdescra[7] = "HUNC";
        if (! sym) matdescra[0] = 'G';
        T one  = static_cast<T>(1.0);
        mkl_csrmv('n', dim, dim, one, matdescra,
                  val, ja, ia, ia + 1, x, one, y);
    }

    template <typename T>
    void csr_mat<T>::MultMv(const T *x, T *y) const
    {
        T zero = static_cast<T>(0.0);
        for (MKL_INT j = 0; j < dim; j++) y[j] = zero;
        MultMv2(x, y);
    }

    template <typename T>
    std::vector<T> csr_mat<T>::to_dense() const
    {
        std::cout << "Converting CSR to " << dim << "x" << dim << " dense matrix!" << std::endl;
        if (dim > 500) std::cout << "Warning: Dense matrix large!!!" << std::endl;
        std::vector<T> res(dim*dim,static_cast<T>(0.0));
        for (MKL_INT row = 0; row < dim; row++) {
            MKL_INT pt_row_curr = ia[row];
            MKL_INT pt_row_next = ia[row+1];
            for (MKL_INT pt = pt_row_curr; pt < pt_row_next; pt++) {
                MKL_INT col = ja[pt];
                res[row + col * dim] = val[pt];
                if (sym && row != col) res[col + row * dim] = conjugate(val[pt]);
            }
        }
        return res;
    }

    template <typename T>
    void swap(csr_mat<T> &lhs, csr_mat<T> &rhs)
    {
        using std::swap;
        swap(lhs.dim, rhs.dim);
        swap(lhs.nnz, rhs.nnz);
        swap(lhs.sym, rhs.sym);
        swap(lhs.val, rhs.val);
        swap(lhs.ja,  rhs.ja);
        swap(lhs.ia,  rhs.ia);
    }

    // Explicit instantiation
    template struct lil_mat_elem<double>;
    template struct lil_mat_elem<std::complex<double>>;

    template class lil_mat<double>;
    template class lil_mat<std::complex<double>>;

    template class csr_mat<double>;
    template class csr_mat<std::complex<double>>;

    template void swap(csr_mat<double>&, csr_mat<double>&);
    template void swap(csr_mat<std::complex<double>>&, csr_mat<std::complex<double>>&);

}
