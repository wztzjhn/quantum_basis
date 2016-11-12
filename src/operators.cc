#include <iostream>
#include "basis.h"
#include "operators.h"



// ----------------- implementation of class opr (operator) ------------------
template <typename T>
opr<T>::opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<T> &mat_):
            site(site_), orbital(orbital_), dim(mat_.size()), fermion(fermion_), diagonal(true)
{
    if (mat_.empty()) {
        mat = nullptr;
    } else {
        mat = new T[mat_.size()];
        for (decltype(mat_.size()) i=0; i<mat_.size(); i++) mat[i] = mat_[i];
    }
}

template <typename T>
opr<T>::opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<std::vector<T>> &mat_):
            site(site_), orbital(orbital_), dim(mat_.size()), fermion(fermion_), diagonal(false)
{
    assert(mat_.empty() || (mat_.size() == mat_[0].size()));
    if (mat_.empty()) {
        mat = nullptr;
    } else {
        mat = new T[dim * dim];
        for (decltype(mat_.size()) j = 0; j < dim; j++) {
            for (decltype(mat_.size()) i = 0; i < dim; i++) {
                // column major order
                mat[i + j * dim] = mat_[i][j];
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
    site(old.site), orbital(old.orbital), dim(old.dim), fermion(old.fermion), diagonal(old.diagonal), mat(old.mat)
{
    old.mat = nullptr;
}

template <typename T>
opr<T> &opr<T>::operator+=(const opr<T> &rhs)
{
    assert(site == rhs.site &&
           orbital == rhs.orbital &&
           dim == rhs.dim &&
           fermion == rhs.fermion);
    if (diagonal == rhs.diagonal) {
        auto sz = (diagonal ? dim : dim * dim);
        for (decltype(sz) i = 0; i < sz; i++) mat[i] += rhs.mat[i];
    } else if (rhs.diagonal) {
        for (decltype(dim) i = 0; i < dim; i++) mat[i + i * dim] += rhs.mat[i];
    } else { // need enlarge the memory usage
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
    assert(site == rhs.site &&
           orbital == rhs.orbital &&
           dim == rhs.dim &&
           fermion == rhs.fermion);
    if (diagonal == rhs.diagonal) {
        auto sz = (diagonal ? dim : dim * dim);
        for (decltype(sz) i = 0; i < sz; i++) mat[i] -= rhs.mat[i];
    } else if (rhs.diagonal) {
        for (decltype(dim) i = 0; i < dim; i++) mat[i + i * dim] -= rhs.mat[i];
    } else { // need enlarge the memory usage
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
    assert(site == rhs.site &&
           orbital == rhs.orbital &&
           dim == rhs.dim);
    if (fermion == rhs.fermion) {
        fermion = false;
    } else {
        fermion = true;
    }
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
            gemm('n', 'n', dim, dim, dim, 1.0, mat, dim, rhs.mat, dim, 0.0, mat_new, dim);
        }
        using std::swap;
        swap(mat, mat_new); // now mat points to the correct memory
        delete [] mat_new;  // release original memroy from mat
    }
    return *this;
}

template <typename T>
void opr<T>::prt() const
{
    std::cout << "operator (" << site << "," << orbital << "), "
              << dim << " x " << dim << ":" << std::endl;
    if (mat != nullptr) {
        if (diagonal) {
            for (decltype(dim) i = 0; i < dim; i++) std::cout << mat[i] << " ";
            std::cout << std::endl;
            size_t xx = 1;
            std::cout << "testing dasum: " << DASUM(&dim, (double*)mat, &xx);
            std::cout << std::endl;
        } else {
            for (decltype(dim) i = 0; i < dim; i++) {
                for (decltype(dim) j = 0; j < dim; j++) {
                    std::cout << mat[i + j * dim] << " ";
                }
                std::cout << std::endl;
            }
            
        }
    }
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
    if (lhs.site == rhs.site &&
        lhs.orbital == rhs.orbital &&
        lhs.dim == rhs.dim &&
        lhs.fermion == rhs.fermion) {
        auto m = lhs.dim;
        if (m == 0) return true;
        assert(lhs.mat != nullptr && rhs.mat != nullptr);
        if (lhs.diagonal == rhs.diagonal) { // both diagonal or both not
            auto sz = (lhs.diagonal ? m : m * m);
            for (decltype(sz) i = 0; i < sz; i++)
                if (std::abs(lhs.mat[i] - rhs.mat[i]) > opr_precision) return false;
        } else if (lhs.diagonal){ // need all rhs non-diagonal elements to be zero
            for (decltype(m) j = 0; j < m; j++) {
                for (decltype(m) i = 0; i < m; i++) {
                    if (i == j) {
                        if (std::abs(lhs.mat[j] - rhs.mat[i + j * m]) > opr_precision) return false;
                    } else {
                        if (std::abs(rhs.mat[i + j * m]) > opr_precision) return false;
                    }
                }
            }
        } else { // need all lhs non-diagonal elements to be zero
            for (decltype(m) j = 0; j < m; j++) {
                for (decltype(m) i = 0; i < m; i++) {
                    if (i == j) {
                        if (std::abs(lhs.mat[i + j * m] - rhs.mat[j]) > opr_precision) return false;
                    } else {
                        if (std::abs(lhs.mat[i + j * m]) > opr_precision) return false;
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

// yes, returning an object is inefficient... how to overcome elegently???
template <typename T>
opr<T> operator+(const opr<T> &lhs, const opr<T> &rhs)
{
    opr<T> sum = lhs;
    sum += rhs;
    return sum;
}

template <typename T>
opr<T> operator-(const opr<T> &lhs, const opr<T> &rhs)
{
    opr<T> diff = lhs;
    diff -= rhs;
    return diff;
}

template <typename T>
opr<T> operator*(const opr<T> &lhs, const opr<T> &rhs)
{
    opr<T> prod = lhs;
    prod *= rhs;
    return prod;
}

//Explicit instantiation, so the class definition can be put in this file
template class opr<double>;
template class opr<std::complex<double>>;

template void swap(opr<double>&, opr<double>&);
template void swap(opr<std::complex<double>>&, opr<std::complex<double>>&);
template bool operator==(const opr<double>&, const opr<double>&);
template bool operator==(const opr<std::complex<double>>&, const opr<std::complex<double>>&);
template bool operator!=(const opr<double>&, const opr<double>&);
template bool operator!=(const opr<std::complex<double>>&, const opr<std::complex<double>>&);
template opr<double> operator+(const opr<double>&, const opr<double>&);
template opr<std::complex<double>> operator+(const opr<std::complex<double>>&, const opr<std::complex<double>>&);
template opr<double> operator-(const opr<double>&, const opr<double>&);
template opr<std::complex<double>> operator-(const opr<std::complex<double>>&, const opr<std::complex<double>>&);
template opr<double> operator*(const opr<double>&, const opr<double>&);
template opr<std::complex<double>> operator*(const opr<std::complex<double>>&, const opr<std::complex<double>>&);

template class mopr<double>;
template class mopr<std::complex<double>>;
