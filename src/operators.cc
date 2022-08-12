#include <algorithm>
#include <cassert>
#include <iostream>

#include "qbasis.h"

// Euclidean norm
inline double blas_nrm2(const MKL_INT &n, const double *x, const MKL_INT &incx) {
    return cblas_dnrm2(n, x, incx);
}
inline double blas_nrm2(const MKL_INT &n, const std::complex<double> *x, const MKL_INT &incx) {
    return cblas_dznrm2(n, x, incx);
}

// C := alpha * op(A) * op(B) + beta * C
// A: mxk, B: kxn, C: mxn
inline void blas_gemm(const CBLAS_LAYOUT &Layout, const CBLAS_TRANSPOSE &transA, const CBLAS_TRANSPOSE &transB,
                      const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const double &alpha,
                      const double *a, const MKL_INT &lda, const double *b, const MKL_INT &ldb,
                      const double &beta, double *c, const MKL_INT &ldc) {
    cblas_dgemm(Layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline void blas_gemm(const CBLAS_LAYOUT &Layout, const CBLAS_TRANSPOSE &transA, const CBLAS_TRANSPOSE &transB,
                      const MKL_INT &m, const MKL_INT &n, const MKL_INT &k, const std::complex<double> &alpha,
                      const std::complex<double> *a, const MKL_INT &lda, const std::complex<double> *b, const MKL_INT &ldb,
                      const std::complex<double> &beta, std::complex<double> *c, const MKL_INT &ldc) {
    cblas_zgemm(Layout, transA, transB, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}

namespace qbasis {
    // ----------------- implementation of class opr (operator) ------------------
    template <typename T>
    opr<T>::opr(const uint32_t &site_, const uint32_t &orbital_, const bool &fermion_, const std::vector<T> &mat_):
        site(site_), orbital(orbital_), dim(static_cast<uint32_t>(mat_.size())), fermion(fermion_), diagonal(true)
    {
        if (mat_.empty() ||
            std::all_of(mat_.begin(), mat_.end(), [](const T &a){ return std::abs(a) < opr_precision; })) {
            mat = nullptr;
        } else {
            mat = new T[mat_.size()];
            for (decltype(mat_.size()) i=0; i<mat_.size(); i++) mat[i] = mat_[i];
        }
    }

    template <typename T>
    opr<T>::opr(const uint32_t &site_, const uint32_t &orbital_, const bool &fermion_, const std::vector<std::vector<T>> &mat_):
        site(site_), orbital(orbital_), dim(static_cast<uint32_t>(mat_.size())), fermion(fermion_), diagonal(false)
    {
        assert(mat_.empty() || (mat_.size() == mat_[0].size()));
        if (mat_.empty()) {
            mat = nullptr;
        } else {
            bool qempty = true;
            for (auto &ele : mat_) {
                if (std::any_of(ele.begin(), ele.end(), [](const T &a){ return std::abs(a) >= opr_precision; })) {
                    qempty = 0;
                    break;
                }
            }
            if (qempty) {
                mat = nullptr;
            } else {
                mat = new T[dim * dim];
                for (uint32_t j = 0; j < dim; j++) {
                    for (uint32_t i = 0; i < dim; i++) {
                        mat[i + j * dim] = mat_[i][j]; // column major order
                    }
                }
            }
        }
    }

    template <typename T>
    opr<T>::opr(const opr<T> &old):
        site(old.site), orbital(old.orbital), dim(old.dim), fermion(old.fermion), diagonal(old.diagonal)
    {
        decltype(dim) sz = (old.mat == nullptr ? 0 : (old.diagonal ? dim : dim * dim));
        if (sz > 0) {
            mat = new T[sz];
            for (decltype(dim) i=0; i < sz; i++) mat[i] = old.mat[i];
        } else {
            mat = nullptr;
        }
    }

    template <typename T>
    opr<T>::opr(opr<T> &&old) noexcept :
    site(old.site), orbital(old.orbital), dim(old.dim), fermion(old.fermion), diagonal(old.diagonal)
    {
        mat = old.mat;
        old.mat = nullptr;
    }

    template <typename T>
    opr<T>::~opr()
    {
        if(mat != nullptr)
        {
            delete [] mat;
            mat = nullptr;
        }
    }

    template <typename T>
    void opr<T>::prt() const
    {
        std::cout << "operator (" << site << "," << orbital << "), "
                  << dim << " x " << dim << ":" << std::endl;
        if (fermion) std::cout << "fermion" << std::endl;
        if (mat != nullptr) {
            if (diagonal) {
                for (decltype(dim) i = 0; i < dim; i++) std::cout << mat[i] << " ";
                std::cout << std::endl;
            } else {
                for (decltype(dim) i = 0; i < dim; i++) {
                    for (decltype(dim) j = 0; j < dim; j++) {
                        std::cout << mat[i + j * dim] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        } else {
            std::cout << "zero operator!" << std::endl;
        }
    }

    template <typename T>
    bool opr<T>::q_zero() const
    {
        auto temp = *this;
        temp.simplify();
        if (temp.mat == nullptr) {
            return true;
        } else {
            return false;
        }
    }

    template <typename T>
    bool opr<T>::q_identity() const
    {
        T prefactor;
        auto temp = normalize(*this, prefactor);
        if (temp.mat == nullptr || ! temp.diagonal || temp.fermion || std::abs(prefactor - static_cast<T>(1.0)) >= opr_precision) {
            return false;
        } else {
            for (decltype(temp.dim) j = 0; j < temp.dim; j++) {
                if (std::abs(temp.mat[j] - static_cast<T>(1.0)) >= opr_precision) return false;
            }
            return true;
        }
    }

    template <typename T>
    double opr<T>::norm() const
    {
        if (mat == nullptr) {
            return 0.0;
        } else {
            decltype(dim) n = diagonal ? dim : dim * dim;
            return blas_nrm2(static_cast<MKL_INT>(n), mat, 1);
        }
    }

    template <typename T>
    opr<T> &opr<T>::simplify()
    {
        if (mat == nullptr) return *this;
        if (! diagonal) {
            diagonal = true; // until found false
            for (decltype(dim) j = 0; j < dim; j++) {
                for (decltype(dim) i = 0; i < dim; i++) {
                    if (i != j && std::abs(mat[j + i * dim]) >= opr_precision) {
                        diagonal = false;
                        break;
                    }
                }
            }
            if(! diagonal) return *this; // nothing to simplify
            bool zero = true; // until found false
            for (decltype(dim) j = 0; j < dim; j++) {
                if (std::abs(mat[j + j * dim]) >= opr_precision) {
                    zero = false;
                    break;
                }
            }
            if (zero) { // found zero operator
                delete [] mat;
                mat = nullptr;
            } else {    // found diagonal operator
                T* mat_new = new T[dim];
                for (decltype(dim) j = 0; j < dim; j++) mat_new[j] = mat[j + j * dim];
                using std::swap;
                swap(mat, mat_new);
                delete [] mat_new;
            }
        } else {
            bool zero = true; // until found false
            for (decltype(dim) j = 0; j < dim; j++) {
                if (std::abs(mat[j]) >= opr_precision) {
                    zero = false;
                    break;
                }
            }
            if (zero) {
                delete [] mat;
                mat = nullptr;
            }
        }
        return *this;
    }

    template <typename T>
    opr<T> &opr<T>::negative()
    {
        decltype(dim) sz = (mat == nullptr ? 0 : (diagonal ? dim : dim * dim));
        for (decltype(sz) i = 0; i < sz; i++) mat[i] = -mat[i];
        return *this;
    }

    template <typename T>
    opr<T> &opr<T>::dagger()
    {
        if ((mat == nullptr) || diagonal) return *this;
        T* mat_new = new T[dim*dim];
        for (decltype(dim) j = 0; j < dim; j++) {
            for (decltype(dim) i = 0; i < dim; i++) {
                mat_new[i + j * dim] = conjugate(mat[j + i * dim]);
            }
        }
        using std::swap;
        swap(mat, mat_new);
        delete [] mat_new;
        return *this;
    }

    template <typename T>
    opr<T> &opr<T>::change_site(const uint32_t &site_)
    {
        site = site_;
        return *this;
    }

    template <typename T>
    opr<T> &opr<T>::transform(const std::vector<uint32_t> &plan)
    {
        site = plan[site];
        return *this;
    }

    template <typename T>
    opr<T> &opr<T>::operator+=(const opr<T> &rhs)
    {
        if (rhs.mat == nullptr) return *this;
        if (mat == nullptr) {
            *this = rhs;
            return *this;
        }
        assert(site == rhs.site && orbital == rhs.orbital && dim == rhs.dim && fermion == rhs.fermion);
        if (diagonal == rhs.diagonal) {
            auto sz = (diagonal ? dim : dim * dim);
            for (decltype(sz) i = 0; i < sz; i++) mat[i] += rhs.mat[i];
        } else if (rhs.diagonal) {
            for (decltype(dim) i = 0; i < dim; i++) mat[i + i * dim] += rhs.mat[i];
        } else {
            diagonal = false;
            T *mat_new = new T[dim * dim];
            for (decltype(dim) i = 0; i < dim * dim; i++) mat_new[i] = rhs.mat[i];  // copy from rhs
            for (decltype(dim) i = 0; i < dim; i++) mat_new[i + i * dim] += mat[i]; // add *this
            using std::swap;
            swap(mat, mat_new); // now mat points to the correct memory
            delete [] mat_new;  // release original memroy from mat
        }
        return *this;
    }

    template <typename T>
    opr<T> &opr<T>::operator-=(const opr<T> &rhs)
    {
        if (rhs.mat == nullptr) return *this;
        if (mat == nullptr) {
            *this = rhs;
            this->negative();
            return *this;
        }
        assert(site == rhs.site && orbital == rhs.orbital && dim == rhs.dim && fermion == rhs.fermion);
        if (diagonal == rhs.diagonal) {
            auto sz = (diagonal ? dim : dim * dim);
            for (decltype(sz) i = 0; i < sz; i++) mat[i] -= rhs.mat[i];
        } else if (rhs.diagonal) {
            for (decltype(dim) i = 0; i < dim; i++) mat[i + i * dim] -= rhs.mat[i];
        } else {
            diagonal = false;
            T *mat_new = new T[dim * dim];
            for (decltype(dim) i = 0; i < dim * dim; i++) mat_new[i] = -rhs.mat[i]; // copy from rhs
            for (decltype(dim) i = 0; i < dim; i++) mat_new[i + i * dim] += mat[i]; // add *this
            using std::swap;
            swap(mat, mat_new); // now mat points to the correct memory
            delete [] mat_new;  // release original memroy from mat
        }
        return *this;
    }

    template <typename T>
    opr<T> &opr<T>::operator*=(const opr<T> &rhs)
    {
        if (rhs.mat == nullptr || mat == nullptr) {
            if (mat != nullptr) {
                delete [] mat;
                mat = nullptr;
            }
            return *this;
        }
        assert(site == rhs.site && orbital == rhs.orbital && dim == rhs.dim);
        fermion = (fermion == rhs.fermion ? false : true);
        if (diagonal && rhs.diagonal) {
            for (decltype(dim) i = 0; i < dim; i++) mat[i] *= rhs.mat[i];
        } else if ((! diagonal) && rhs.diagonal) {
            for (decltype(dim) j = 0; j < dim; j++) {
                for (decltype(dim) i = 0; i < dim; i++) {
                    mat[i + j * dim] *= rhs.mat[j];
                }
            }
        } else {
            T *mat_new = new T[dim * dim];
            for (decltype(dim) i = 0; i < dim * dim; i++) mat_new[i] = rhs.mat[i];  // copy from rhs
            if (diagonal && (! rhs.diagonal)) {
                diagonal = false;
                for (decltype(dim) j = 0; j < dim; j++) {
                    for (decltype(dim) i = 0; i < dim; i++) mat_new[i + j * dim] *= mat[i];
                }
            } else {
                MKL_INT diml = static_cast<MKL_INT>(dim);
                blas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, diml, diml, diml, 1.0, mat, diml, rhs.mat, diml, 0.0, mat_new, diml);
            }
            using std::swap;
            swap(mat, mat_new); // now mat points to the correct memory
            delete [] mat_new;  // release original memroy from mat
        }
        this->simplify();
        return *this;
    }

    template <typename T>
    opr<T> &opr<T>::operator*=(const T &rhs)
    {
        if (mat == nullptr || std::abs(rhs) < opr_precision) {
            if (mat != nullptr) {
                delete [] mat;
                mat = nullptr;
            }
            return *this;
        }
        auto sz = (diagonal ? dim : dim * dim);
        for (decltype(sz) i = 0; i < sz; i++) mat[i] *= rhs;
        return *this;
    }

    template <typename T>
    void swap(opr<T> &lhs, opr<T> &rhs)
    {
        using std::swap;
        swap(lhs.site, rhs.site);
        swap(lhs.orbital, rhs.orbital);
        swap(lhs.dim, rhs.dim);
        swap(lhs.fermion, rhs.fermion);
        swap(lhs.diagonal, rhs.diagonal);
        swap(lhs.mat, rhs.mat);
    }

    template <typename T>
    bool operator==(const opr<T> &lhs, const opr<T> &rhs)
    {
        if (lhs.mat == nullptr && rhs.mat == nullptr) {
            return true;
        } else if (lhs.mat == nullptr || rhs.mat == nullptr) {
            return false;
        }
        if (lhs.site == rhs.site && lhs.orbital == rhs.orbital &&
            lhs.dim == rhs.dim && lhs.fermion == rhs.fermion) {
            auto m = lhs.dim;
            if (m == 0) return true;
            if (lhs.diagonal == rhs.diagonal) { // both diagonal or both not
                auto sz = (lhs.diagonal ? m : m * m);
                for (decltype(sz) i = 0; i < sz; i++)
                    if (std::abs(lhs.mat[i] - rhs.mat[i]) >= opr_precision) return false;
            } else if (lhs.diagonal){ // need all rhs non-diagonal elements to be zero
                for (decltype(m) j = 0; j < m; j++) {
                    for (decltype(m) i = 0; i < m; i++) {
                        if (i == j) {
                            if (std::abs(lhs.mat[j] - rhs.mat[i + j * m]) >= opr_precision) return false;
                        } else {
                            if (std::abs(rhs.mat[i + j * m]) >= opr_precision) return false;
                        }
                    }
                }
            } else { // need all lhs non-diagonal elements to be zero
                for (decltype(m) j = 0; j < m; j++) {
                    for (decltype(m) i = 0; i < m; i++) {
                        if (i == j) {
                            if (std::abs(lhs.mat[i + j * m] - rhs.mat[j]) >= opr_precision) return false;
                        } else {
                            if (std::abs(lhs.mat[i + j * m]) >= opr_precision) return false;
                        }
                    }
                }
            }
            return true;
        } else {
            return false;
        }
    }

    template <typename T>
    bool operator!=(const opr<T> &lhs, const opr<T> &rhs)
    {
        return !(lhs == rhs);
    }

    template <typename T>
    bool operator<(const opr<T> &lhs, const opr<T> &rhs)
    {
        if (rhs.q_zero()) {
            return false;
        } else if (lhs.q_zero()) {
            return true;
        }
        if (lhs.site < rhs.site) {
            return true;
        } else if (lhs.site > rhs.site) {
            return false;
        }
        if (lhs.orbital < rhs.orbital) {
            return true;
        } else if (lhs.orbital > rhs.orbital) {
            return false;
        }
        return (lhs.fermion < rhs.fermion ? true : false);
    }

    template <typename T>
    opr<T> operator*(const T &lhs, const opr<T> &rhs)
    {
        opr<T> prod = rhs;
        prod *= lhs;
        return prod;
    }

    template <typename T>
    opr<T> operator*(const opr<T> &lhs, const T &rhs)
    {
        opr<T> prod = lhs;
        prod *= rhs;
        return prod;
    }

    template <typename T>
    opr<T> normalize(const opr<T> &old, T &prefactor)
    {
        opr<T> temp = old;
        temp.simplify();
        if (temp.mat == nullptr) {
            prefactor = 0.0;
        } else {
            double norm = temp.norm();
            assert(norm >= opr_precision);
            decltype(temp.dim) n = temp.diagonal ? temp.dim : temp.dim * temp.dim;
            T phase = 0.0;
            for (decltype(n) j = 0; j < n; j++) {
                if (std::abs(temp.mat[j]) > opr_precision) {
                    phase = temp.mat[j] / std::abs(temp.mat[j]);
                    break;
                }
            }
            assert(std::abs(std::abs(phase) - 1.0) < opr_precision);
            prefactor = norm / sqrt(static_cast<double>(temp.dim)) * phase;
            T preinv = 1.0 / prefactor;
            for (decltype(n) j = 0; j < n; j++) temp.mat[j] *= preinv;
        }
        return temp;
    }


    // ----------------- implementation of class opr_prod --------------------
    template <typename T>
    opr_prod<T>::opr_prod(const opr<T> &ele)
    {
        T prefactor;
        auto ele_new = normalize(ele, prefactor);
        coeff = prefactor;
        if (std::abs(prefactor) >= opr_precision) {
            mat_prod = std::list<opr<T>>(1, ele_new);
        } else {
            mat_prod = std::list<opr<T>>();
        }
    }

    template <typename T>
    void opr_prod<T>::prt() const
    {
        std::cout << mat_prod.size() << " products: "<< std::endl;
        std::cout << "prefactor: " << coeff << std::endl;
        for(auto &product : mat_prod){
            product.prt();
            std::cout << "xxxxxxxx" << std::endl;
        }
    }

    template <typename T>
    bool opr_prod<T>::q_zero() const
    {
        return std::abs(coeff) < opr_precision ? true : false;
    }

    template <typename T>
    bool opr_prod<T>::q_diagonal() const
    {
        if (q_zero()) return true;
        for (auto it = mat_prod.begin(); it != mat_prod.end(); it++) {
            if (! it->q_diagonal()) return false;
        }
        return true;
    }

    template <typename T>
    bool opr_prod<T>::q_prop_identity() const // ask this question without simplifying the operator products
    {
        return (std::abs(coeff) >= opr_precision && mat_prod.empty()) ? true : false;
    }

    template <typename T>
    uint32_t opr_prod<T>::len() const
    {
        return static_cast<uint32_t>(mat_prod.size());
    }

    template <typename T>
    opr<T> &opr_prod<T>::operator[](uint32_t n)
    {
        assert(n < len());
        auto it = mat_prod.begin();
        for (decltype(n) i = 0; i < n; i++) ++it;
        return *it;
    }

    template <typename T>
    const opr<T> &opr_prod<T>::operator[](uint32_t n) const
    {
        assert(n < len());
        auto it = mat_prod.begin();
        for (decltype(n) i = 0; i < n; i++) ++it;
        return *it;
    }

    template <typename T>
    opr_prod<T> &opr_prod<T>::negative()
    {
        coeff = -coeff;
        return *this;
    }

    template <typename T>
    opr_prod<T> &opr_prod<T>::dagger()
    {
        coeff = conjugate(coeff);
        mat_prod.reverse();
        for (auto it = mat_prod.begin(); it != mat_prod.end(); it++) it->dagger();
        return *this;
    }

    template <typename T>
    opr_prod<T> &opr_prod<T>::bubble_sort()
    {
        if (q_zero() || q_prop_identity() || len() <= 1) return *this;
        uint32_t length = len();

        auto it_end = mat_prod.end();
        using std::swap;
        for (uint32_t j = 1; j < length; j++) {
            it_end--;
            int cnt0 = 0;
            for (auto it = mat_prod.begin(); it != it_end; it++) {
                auto it_next = it;
                it_next++;
                std::vector<uint32_t> a = {it->site, it->orbital};
                std::vector<uint32_t> b = {it_next->site, it_next->orbital};
                if (b < a) {
                    cnt0++;
                    if (it->fermion && it_next->fermion) coeff = -coeff;
                    swap(*it,*it_next);
                }

            }
            if (cnt0 == 0) break;                                                 // already sorted
        }
        return *this;
    }

    template <typename T>
    opr_prod<T> &opr_prod<T>::transform(const std::vector<uint32_t> &plan)
    {
        if (q_zero() || q_prop_identity()) return *this;

        for (auto it = mat_prod.begin(); it != mat_prod.end(); it++) it->site = plan[it->site];
        bubble_sort();
        return *this;
    }

    template <typename T>
    opr_prod<T> &opr_prod<T>::operator*=(opr<T> rhs)
    {
        if (this->q_zero()) return *this;                  // zero operator self
        T prefactor;
        rhs = normalize(rhs, prefactor);
        if (this->q_prop_identity()) {                     // itself proportional to identity operator
            coeff *= prefactor;
            if (std::abs(prefactor) >= opr_precision) mat_prod.push_back(std::move(rhs));
            return *this;
        }
        if (std::abs(prefactor) < opr_precision) {         // zero operator
            coeff = static_cast<T>(0.0);
            mat_prod.clear();
            return *this;
        }
        if (rhs.q_identity()) {                           // identity operator
            coeff *= prefactor;
            return *this;
        }
        auto j = mat_prod.rbegin();
        std::vector<uint32_t> val_rhs = {rhs.site, rhs.orbital};
        std::vector<uint32_t> val_j   = {j->site, j->orbital};
        while (j != mat_prod.rend() && val_rhs < val_j) {
            if(rhs.fermion && j->fermion) prefactor = -prefactor;
            j++;
            if (j != mat_prod.rend()) {
                val_j[0] = j->site;
                val_j[1] = j->orbital;
            }
        }
        if (j == mat_prod.rend() || val_j < val_rhs) { // opr not found
            coeff *= prefactor;
            mat_prod.insert(j.base(), std::move(rhs));
        } else { // found an operator on same site and same orbital
            auto opr_prod(*j);
            opr_prod *= rhs;
            T prefactor_new;
            (*j) = normalize(opr_prod, prefactor_new);
            coeff *= (prefactor * prefactor_new);
            if (std::abs(coeff) < opr_precision) { // zero operator
                mat_prod.clear();
            } else if (mat_prod.size() > 1 && j->q_identity()) {          // identity operator
                mat_prod.erase(--(j.base()));
            }
        }
        return *this;
    }

    template <typename T>
    opr_prod<T> &opr_prod<T>::operator*=(opr_prod<T> rhs)
    {
        if (this->q_zero()) return *this;             // zero operator self
        if (this->q_prop_identity()) {                // itself proportional to identity operator
            (*this) = rhs;
            return *this;
        }
        if (rhs.q_zero()) {                           // zero operator
            coeff = static_cast<T>(0.0);
            mat_prod.clear();
            return *this;
        }
        if (rhs.q_prop_identity()) {                  // proportional to identity operator
            coeff *= rhs.coeff;
            return *this;
        }
        coeff *= rhs.coeff;
        for (auto &ele : rhs.mat_prod) {
            (*this) *= ele;                           // speed slower in this way, but safer; optimize later if needed
            if (this->q_zero()) break;
        }
        return *this;
    }

    template <typename T>
    opr_prod<T> &opr_prod<T>::operator*=(const T &rhs)
    {
        coeff *= rhs;
        return *this;
    }


    template <typename T>
    void swap(opr_prod<T> &lhs, opr_prod<T> &rhs)
    {
        using std::swap;
        swap(lhs.coeff, rhs.coeff);
        swap(lhs.mat_prod, rhs.mat_prod);
    }

    template <typename T>
    bool operator==(const opr_prod<T> &lhs, const opr_prod<T> &rhs)
    {
        if (std::abs(lhs.coeff - rhs.coeff) >= opr_precision || lhs.mat_prod.size() != rhs.mat_prod.size()) {
            return false;
        } else {
            return lhs.mat_prod == rhs.mat_prod;
        }
    }

    template <typename T>
    bool operator!=(const opr_prod<T> &lhs, const opr_prod<T> &rhs)
    {
        return !(lhs == rhs);
    }

    template <typename T>
    bool operator<(const opr_prod<T> &lhs, const opr_prod<T> &rhs)
    {
        if (rhs.q_zero()) {
            return false;
        } else if (lhs.q_zero()) {
            return true;
        }
        if (lhs.len() < rhs.len()) {
            return true;
        } else {
            return lhs.mat_prod < rhs.mat_prod;
        }
    }

    template <typename T>
    opr_prod<T> operator*(const opr_prod<T> &lhs, const opr_prod<T> &rhs)
    {
        opr_prod<T> prod = lhs;
        prod *= rhs;
        return prod;
    }

    template <typename T>
    opr_prod<T> operator*(const opr_prod<T> &lhs, const opr<T> &rhs)
    {
        opr_prod<T> prod = lhs;
        prod *= rhs;
        return prod;
    }

    template <typename T>
    opr_prod<T> operator*(const opr<T> &lhs, const opr_prod<T> &rhs)
    {
        opr_prod<T> prod(lhs);
        prod *= rhs;
        return prod;
    }

    template <typename T>
    opr_prod<T> operator*(const opr_prod<T> &lhs, const T &rhs)
    {
        opr_prod<T> prod = lhs;
        prod *= rhs;
        return prod;
    }

    template <typename T>
    opr_prod<T> operator*(const T &lhs, const opr_prod<T> &rhs)
    {
        opr_prod<T> prod = rhs;
        prod *= lhs;
        return prod;
    }

    template <typename T>
    opr_prod<T> operator*(const opr<T> &lhs, const opr<T> &rhs)
    {
        opr_prod<T> prod(lhs);
        prod *= rhs;
        return prod;
    }

    // ----------------- implementation of class mopr ------------------------
    template <typename T>
    mopr<T>::mopr(const opr<T> &ele)
    {
        if (ele.q_zero()) {
            mats = std::list<opr_prod<T>>();
        } else {
            mats.emplace_back(ele);
        }
    }

    template <typename T>
    mopr<T>::mopr(const opr_prod<T> &ele)
    {
        if (ele.q_zero()) {
            mats = std::list<opr_prod<T>>();
        } else {
            mats = std::list<opr_prod<T>>(1, ele);
        }
    }

    template <typename T>
    void mopr<T>::prt() const
    {
        std::cout << "terms: " << mats.size() << std::endl;
        for(auto &product : mats){
            std::cout << "----------------------" << std::endl;
            std::cout << "prefactor: " << product.coeff << std::endl;
            for (auto &individual : product.mat_prod) {
                individual.prt();
                std::cout << "xxxxxxxx" << std::endl;
            }
            std::cout << std::endl;
        }
    }

    template <typename T>
    bool mopr<T>::q_diagonal() const
    {
        if (q_zero()) return true;
        for (auto it = mats.begin(); it != mats.end(); it++) {
            if (! it->q_diagonal()) return false;
        }
        return true;
    }

    template <typename T>
    opr_prod<T> &mopr<T>::operator[](uint32_t n)
    {
        assert(n < size());
        assert(! mats.empty());
        auto it = mats.begin();
        for (decltype(n) i = 0; i < n; i++) ++it;
        return *it;
    }

    template <typename T>
    const opr_prod<T> &mopr<T>::operator[](uint32_t n) const
    {
        assert(n < size());
        assert(! mats.empty());
        auto it = mats.begin();
        for (decltype(n) i = 0; i < n; i++) ++it;
        return *it;
    }

    template <typename T>
    mopr<T> &mopr<T>::simplify()
    {
        mats.sort();
        auto it = mats.begin();
        auto it_prev = it++;
        while (it != mats.end()) {  // combine all terms
            if(it->len() == it_prev->len() && it->mat_prod == it_prev->mat_prod) {
                it_prev->coeff += it->coeff;
                it = mats.erase(it);
            } else {
                it_prev = it++;
            }
        }
        it = mats.begin();
        while (it != mats.end()) { // remove all zero terms
            if (std::abs(it->coeff) < opr_precision) {
                it = mats.erase(it);
            } else {
                it++;
            }
        }
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::negative()
    {
        for (auto &ele : mats) ele.negative();
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::dagger()
    {
        for (auto &ele : mats) ele.dagger();
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::transform(const std::vector<uint32_t> &plan)
    {
        for (auto &ele: mats) ele.transform(plan);
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::operator+=(opr<T> rhs)
    {
        T prefactor;
        rhs = normalize(rhs, prefactor);
        if (std::abs(prefactor) < opr_precision) return *this; // adding zero
        (*this) += opr_prod<T>(prefactor * rhs);
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::operator+=(opr_prod<T> rhs)
    {
        if (rhs.q_zero()) return *this; // adding zero
        if (mats.empty()) {             // itself zero
            mats.push_back(std::move(rhs));
            return *this;
        }
        auto it = mats.begin();
        while (it != mats.end() && *it < rhs) it++;
        if (rhs.len() == 1) {
            if (it == mats.end()) {         // rhs not found
                mats.insert(it, std::move(rhs));
            } else if(rhs < *it) {          // rhs not found
                mats.insert(it, std::move(rhs));
            } else {                        // found something on same site, orbital and fermionic property
                auto opr_sum = it->coeff * (it->mat_prod).front();
                opr_sum += rhs.coeff * rhs.mat_prod.front();
                *it = opr_prod<T>(std::move(opr_sum));
                if (it->q_zero()) mats.erase(it);
            }
        } else {
            mats.insert(it, std::move(rhs));  // leave simplification separately
        }
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::operator+=(mopr<T> rhs)
    {
        if (rhs.mats.empty()) return *this;  // adding zero
        if (mats.empty()) {                  // itself zero
            swap(*this, rhs);
            return *this;
        }
        for (auto &ele : rhs.mats) (*this) += ele;
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::operator-=(opr<T> rhs)
    {
        (*this) += rhs.negative();
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::operator-=(opr_prod<T> rhs)
    {
        (*this) += rhs.negative();
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::operator-=(mopr<T> rhs)
    {
        (*this) += rhs.negative();
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::operator*=(opr<T> rhs)
    {
        auto it = mats.begin();
        while (it != mats.end()) {
            (*it) *= rhs;
            if (it->q_zero()) {
                it = mats.erase(it);
            } else {
                ++it;
            }
        }
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::operator*=(opr_prod<T> rhs)
    {
        auto it = mats.begin();
        while (it != mats.end()) {
            (*it) *= rhs;
            if (it->q_zero()) {
                it = mats.erase(it);
            } else {
                ++it;
            }
        }
        return *this;
    }

    template <typename T>
    mopr<T> &mopr<T>::operator*=(mopr<T> rhs)
    {
        mopr<T> sum;
        for (auto &ele : rhs.mats) {
            sum += (*this) * ele;
        }
        swap(*this, sum);
        return *this;
    }


    template <typename T>
    mopr<T> &mopr<T>::operator*=(const T &rhs)
    {
        if (std::abs(rhs) < opr_precision) {
            mats.clear();
            return *this;
        }
        for (auto &ele : mats) ele *= rhs;
        return *this;
    }

    template <typename T>
    void swap(mopr<T> &lhs, mopr<T> &rhs)
    {
        using std::swap;
        swap(lhs.mats, rhs.mats);
    }

    template <typename T>
    bool operator==(const mopr<T> &lhs, const mopr<T> &rhs)
    {
        return lhs.mats == rhs.mats;
    }

    template <typename T>
    bool operator!=(const mopr<T> &lhs, const mopr<T> &rhs)
    {
        return lhs.mats != rhs.mats;
    }

    template <typename T>
    mopr<T> operator+(const mopr<T> &lhs, const mopr<T> &rhs)
    {
        mopr<T> sum(lhs);
        sum += rhs;
        return sum;
    }

    template <typename T>
    mopr<T> operator+(const mopr<T> &lhs, const opr_prod<T> &rhs)
    {
        mopr<T> sum(lhs);
        sum += rhs;
        return sum;
    }

    template <typename T>
    mopr<T> operator+(const opr_prod<T> &lhs, const mopr<T> &rhs)
    {
        mopr<T> sum(rhs);
        sum += lhs;
        return sum;
    }

    template <typename T>
    mopr<T> operator+(const mopr<T> &lhs, const opr<T> &rhs)
    {
        mopr<T> sum(lhs);
        sum += rhs;
        return sum;
    }

    template <typename T>
    mopr<T> operator+(const opr<T> &lhs, const mopr<T> &rhs)
    {
        mopr<T> sum(rhs);
        sum += lhs;
        return sum;
    }

    template <typename T>
    mopr<T> operator+(const opr_prod<T> &lhs, const opr_prod<T> &rhs)
    {
        mopr<T> sum(lhs);
        sum += rhs;
        return sum;
    }

    template <typename T>
    mopr<T> operator+(const opr_prod<T> &lhs, const opr<T> &rhs)
    {
        mopr<T> sum(lhs);
        sum += rhs;
        return sum;
    }

    template <typename T>
    mopr<T> operator+(const opr<T> &lhs, const opr_prod<T> &rhs)
    {
        mopr<T> sum(rhs);
        sum += lhs;
        return sum;
    }

    template <typename T>
    mopr<T> operator+(const opr<T> &lhs, const opr<T> &rhs)
    {
        mopr<T> sum(lhs);
        sum += rhs;
        return sum;
    }

    template <typename T>
    mopr<T> operator-(const mopr<T> &lhs, const mopr<T> &rhs)
    {
        mopr<T> dif(lhs);
        dif -= rhs;
        return dif;
    }

    template <typename T>
    mopr<T> operator-(const mopr<T> &lhs, const opr_prod<T> &rhs)
    {
        mopr<T> dif(lhs);
        dif -= rhs;
        return dif;
    }

    template <typename T>
    mopr<T> operator-(const opr_prod<T> &lhs, const mopr<T> &rhs)
    {
        mopr<T> dif(rhs);
        dif.negative();
        dif += lhs;
        return dif;
    }

    template <typename T>
    mopr<T> operator-(const mopr<T> &lhs, const opr<T> &rhs)
    {
        mopr<T> dif(lhs);
        dif -= rhs;
        return dif;
    }

    template <typename T>
    mopr<T> operator-(const opr<T> &lhs, const mopr<T> &rhs)
    {
        mopr<T> dif(rhs);
        dif.negative();
        dif += lhs;
        return dif;
    }

    template <typename T>
    mopr<T> operator-(const opr_prod<T> &lhs, const opr_prod<T> &rhs)
    {
        mopr<T> dif(lhs);
        dif -= rhs;
        return dif;
    }

    template <typename T>
    mopr<T> operator-(const opr_prod<T> &lhs, const opr<T> &rhs)
    {
        mopr<T> dif(lhs);
        dif -= rhs;
        return dif;
    }

    template <typename T>
    mopr<T> operator-(const opr<T> &lhs, const opr_prod<T> &rhs)
    {
        mopr<T> dif(rhs);
        dif.negative();
        dif += lhs;
        return dif;
    }

    template <typename T>
    mopr<T> operator-(const opr<T> &lhs, const opr<T> &rhs)
    {
        mopr<T> dif(lhs);
        dif -= rhs;
        return dif;
    }

    template <typename T>
    mopr<T> operator*(const mopr<T> &lhs, const mopr<T> &rhs)
    {
        mopr<T> prod(lhs);
        prod *= rhs;
        return prod;
    }

    template <typename T>
    mopr<T> operator*(const mopr<T> &lhs, const opr_prod<T> &rhs)
    {
        mopr<T> prod(lhs);
        prod *= rhs;
        return prod;
    }

    template <typename T>
    mopr<T> operator*(const opr_prod<T> &lhs, const mopr<T> &rhs)
    {
        mopr<T> prod(lhs);
        prod *= rhs;
        return prod;
    }

    template <typename T>
    mopr<T> operator*(const mopr<T> &lhs, const opr<T> &rhs)
    {
        mopr<T> prod(lhs);
        prod *= rhs;
        return prod;
    }

    template <typename T>
    mopr<T> operator*(const opr<T> &lhs, const mopr<T> &rhs)
    {
        mopr<T> prod(lhs);
        prod *= rhs;
        return prod;
    }

    template <typename T>
    mopr<T> operator*(const T &lhs, const mopr<T> &rhs)
    {
        mopr<T> prod(rhs);
        prod *= lhs;
        return prod;
    }

    template <typename T>
    mopr<T> operator*(const mopr<T> &lhs, const T &rhs)
    {
        mopr<T> prod(lhs);
        prod *= rhs;
        return prod;
    }


    // Explicit instantiation, so the class definition can be put in this file
    template class opr<double>;
    template class opr<std::complex<double>>;

    template void swap(opr<double>&, opr<double>&);
    template void swap(opr<std::complex<double>>&, opr<std::complex<double>>&);

    template bool operator==(const opr<double>&, const opr<double>&);
    template bool operator==(const opr<std::complex<double>>&, const opr<std::complex<double>>&);

    template bool operator!=(const opr<double>&, const opr<double>&);
    template bool operator!=(const opr<std::complex<double>>&, const opr<std::complex<double>>&);

    template bool operator<(const opr<double>&, const opr<double>&);
    template bool operator<(const opr<std::complex<double>>&, const opr<std::complex<double>>&);

    template opr<double> operator*(const double&, const opr<double>&);
    template opr<std::complex<double>> operator*(const std::complex<double>&, const opr<std::complex<double>>&);

    template opr<double> operator*(const opr<double>&, const double&);
    template opr<std::complex<double>> operator*(const opr<std::complex<double>>&, const std::complex<double>&);

    template opr<double> normalize(const opr<double>&, double&);
    template opr<std::complex<double>> normalize(const opr<std::complex<double>>&, std::complex<double>&);

    // explicit instantiation of opr_prod
    template class opr_prod<double>;
    template class opr_prod<std::complex<double>>;

    template void swap(opr_prod<double>&, opr_prod<double>&);
    template void swap(opr_prod<std::complex<double>>&, opr_prod<std::complex<double>>&);

    template bool operator==(const opr_prod<double>&, const opr_prod<double>&);
    template bool operator==(const opr_prod<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template bool operator!=(const opr_prod<double>&, const opr_prod<double>&);
    template bool operator!=(const opr_prod<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template bool operator<(const opr_prod<double>&, const opr_prod<double>&);
    template bool operator<(const opr_prod<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template opr_prod<double> operator*(const opr_prod<double>&, const opr_prod<double>&);
    template opr_prod<std::complex<double>> operator*(const opr_prod<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template opr_prod<double> operator*(const opr_prod<double>&, const opr<double>&);
    template opr_prod<std::complex<double>> operator*(const opr_prod<std::complex<double>>&, const opr<std::complex<double>>&);

    template opr_prod<double> operator*(const opr<double>&, const opr_prod<double>&);
    template opr_prod<std::complex<double>> operator*(const opr<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template opr_prod<double> operator*(const opr_prod<double>&, const double&);
    template opr_prod<std::complex<double>> operator*(const opr_prod<std::complex<double>>&, const std::complex<double>&);

    template opr_prod<double> operator*(const double&, const opr_prod<double>&);
    template opr_prod<std::complex<double>> operator*(const std::complex<double>&, const opr_prod<std::complex<double>>&);

    template opr_prod<double> operator*(const opr<double>&, const opr<double>&);
    template opr_prod<std::complex<double>> operator*(const opr<std::complex<double>>&, const opr<std::complex<double>>&);

    // explicit instantiation of mopr
    template class mopr<double>;
    template class mopr<std::complex<double>>;

    template void swap(mopr<double>&, mopr<double>&);
    template void swap(mopr<std::complex<double>>&, mopr<std::complex<double>>&);

    template bool operator==(const mopr<double>&, const mopr<double>&);
    template bool operator==(const mopr<std::complex<double>>&, const mopr<std::complex<double>>&);

    template mopr<double> operator+(const mopr<double>&, const mopr<double>&);
    template mopr<std::complex<double>> operator+(const mopr<std::complex<double>>&, const mopr<std::complex<double>>&);

    template mopr<double> operator+(const mopr<double>&, const opr_prod<double>&);
    template mopr<std::complex<double>> operator+(const mopr<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template mopr<double> operator+(const opr_prod<double>&, const mopr<double>&);
    template mopr<std::complex<double>> operator+(const opr_prod<std::complex<double>>&, const mopr<std::complex<double>>&);

    template mopr<double> operator+(const mopr<double>&, const opr<double>&);
    template mopr<std::complex<double>> operator+(const mopr<std::complex<double>>&, const opr<std::complex<double>>&);

    template mopr<double> operator+(const opr<double>&, const mopr<double>&);
    template mopr<std::complex<double>> operator+(const opr<std::complex<double>>&, const mopr<std::complex<double>>&);

    template mopr<double> operator+(const opr_prod<double>&, const opr_prod<double>&);
    template mopr<std::complex<double>> operator+(const opr_prod<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template mopr<double> operator+(const opr_prod<double>&, const opr<double>&);
    template mopr<std::complex<double>> operator+(const opr_prod<std::complex<double>>&, const opr<std::complex<double>>&);

    template mopr<double> operator+(const opr<double>&, const opr_prod<double>&);
    template mopr<std::complex<double>> operator+(const opr<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template mopr<double> operator+(const opr<double>&, const opr<double>&);
    template mopr<std::complex<double>> operator+(const opr<std::complex<double>>&, const opr<std::complex<double>>&);

    template mopr<double> operator-(const mopr<double>&, const mopr<double>&);
    template mopr<std::complex<double>> operator-(const mopr<std::complex<double>>&, const mopr<std::complex<double>>&);

    template mopr<double> operator-(const mopr<double>&, const opr_prod<double>&);
    template mopr<std::complex<double>> operator-(const mopr<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template mopr<double> operator-(const opr_prod<double>&, const mopr<double>&);
    template mopr<std::complex<double>> operator-(const opr_prod<std::complex<double>>&, const mopr<std::complex<double>>&);

    template mopr<double> operator-(const mopr<double>&, const opr<double>&);
    template mopr<std::complex<double>> operator-(const mopr<std::complex<double>>&, const opr<std::complex<double>>&);

    template mopr<double> operator-(const opr<double>&, const mopr<double>&);
    template mopr<std::complex<double>> operator-(const opr<std::complex<double>>&, const mopr<std::complex<double>>&);

    template mopr<double> operator-(const opr_prod<double>&, const opr_prod<double>&);
    template mopr<std::complex<double>> operator-(const opr_prod<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template mopr<double> operator-(const opr_prod<double>&, const opr<double>&);
    template mopr<std::complex<double>> operator-(const opr_prod<std::complex<double>>&, const opr<std::complex<double>>&);

    template mopr<double> operator-(const opr<double>&, const opr_prod<double>&);
    template mopr<std::complex<double>> operator-(const opr<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template mopr<double> operator-(const opr<double>&, const opr<double>&);
    template mopr<std::complex<double>> operator-(const opr<std::complex<double>>&, const opr<std::complex<double>>&);

    template mopr<double> operator*(const mopr<double>&, const mopr<double>&);
    template mopr<std::complex<double>> operator*(const mopr<std::complex<double>>&, const mopr<std::complex<double>>&);

    template mopr<double> operator*(const mopr<double>&, const opr_prod<double>&);
    template mopr<std::complex<double>> operator*(const mopr<std::complex<double>>&, const opr_prod<std::complex<double>>&);

    template mopr<double> operator*(const opr_prod<double>&, const mopr<double>&);
    template mopr<std::complex<double>> operator*(const opr_prod<std::complex<double>>&, const mopr<std::complex<double>>&);

    template mopr<double> operator*(const mopr<double>&, const opr<double>&);
    template mopr<std::complex<double>> operator*(const mopr<std::complex<double>>&, const opr<std::complex<double>>&);

    template mopr<double> operator*(const opr<double>&, const mopr<double>&);
    template mopr<std::complex<double>> operator*(const opr<std::complex<double>>&, const mopr<std::complex<double>>&);

    template mopr<double> operator*(const double&, const mopr<double>&);
    template mopr<std::complex<double>> operator*(const std::complex<double>&, const mopr<std::complex<double>>&);

    template mopr<double> operator*(const mopr<double>&, const double&);
    template mopr<std::complex<double>> operator*(const mopr<std::complex<double>>&, const std::complex<double>&);

}
