#include <iostream>
#include "basis.h"
#include "operators.h"



// ----------------- implementation of class opr (operator) ------------------
template <typename T>
opr<T>::opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<T> &mat_):
            site(site_), orbital(orbital_), m(mat_.size()), n(mat_.size()), fermion(fermion_), diagonal(true)
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
            site(site_), orbital(orbital_), m(mat_.size()), fermion(fermion_), diagonal(false)
{
    if (mat_.empty() || mat_[0].empty()) {
        n = 0;
        mat = nullptr;
    } else {
        n = mat_[0].size();
        mat = new T[mat_.size() * mat_[0].size()];
        for (decltype(mat_.size()) j = 0; j < n; j++) {
            for (decltype(mat_.size()) i = 0; i < m; i++) {
                // column major order
                mat[i + j * mat_.size()] = mat_[i][j];
            }
        }
    }
}

template <typename T>
opr<T>::opr(const opr<T> &old):
    site(old.site), orbital(old.orbital), m(old.m), n(old.n), fermion(old.fermion), diagonal(old.diagonal)
{
    decltype(old.m) sz = (old.mat == nullptr ? 0 : (old.diagonal ? old.m : old.m * old.n));
    if (sz > 0) {
        mat = new T[sz];
        for (decltype(m) i=0; i < sz; i++) mat[i] = old.mat[i];
    } else {
        mat = nullptr;
    }
    
}

template <typename T>
opr<T>::opr(opr<T> &&old) noexcept :
    site(old.site), orbital(old.orbital), m(old.m), n(old.n), fermion(old.fermion), diagonal(old.diagonal), mat(old.mat)
{
    old.mat = nullptr;
}

template <typename T>
void opr<T>::prt() const
{
    std::cout << "operator (" << site << "," << orbital << "), "
              << m << " x " << n << ":" << std::endl;
    if (mat != nullptr) {
        if (diagonal) {
            for (decltype(m) i = 0; i < m; i++) std::cout << mat[i] << " ";
            std::cout << std::endl;
            size_t xx = 1;
            std::cout << "testing dasum: " << DASUM(&m, (double*)mat, &xx);
            std::cout << std::endl;
        } else {
            for (decltype(m) i = 0; i < m; i++) {
                for (decltype(n) j = 0; j < n; j++) {
                    std::cout << mat[i + j * m] << " ";
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
    swap(lhs.m, rhs.m);
    swap(lhs.n, rhs.n);
    swap(lhs.fermion, rhs.fermion);
    swap(lhs.diagonal, rhs.diagonal);
    swap(lhs.mat, rhs.mat);
}


//Explicit instantiation, so the class definition can be put in this file
template class opr<double>;
template class opr<std::complex<double>>;

template void swap(opr<double>&, opr<double>&);
template void swap(opr<std::complex<double>>&, opr<std::complex<double>>&);

template class mopr<double>;
template class mopr<std::complex<double>>;
