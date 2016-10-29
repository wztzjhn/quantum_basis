#include <iostream>
#include "basis.h"
#include "operators.h"


// ----------------- implementation of class opr (operator) ------------------
template <typename T>
opr<T>::opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<T> &mat_):
            site(site_), orbital(orbital_), fermion(fermion_), diagonal(true)
{
    mat = nullptr;
}

template <typename T>
opr<T>::opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<std::vector<T>> &mat_):
            site(site_), orbital(orbital_), fermion(fermion_), diagonal(false)
{
    mat = nullptr;
}


template <typename T>
void opr<T>::test() const
{
    std::cout << std::endl << "operator test:" << std::endl
    << "site: " << site << std::endl
    << "orbital: " << orbital << std::endl
    << "fermion: " << fermion << std::endl
    << "diagonal: " << diagonal << std::endl;
}

//Explicit instantiation, so the class definition can be put in this file
template class opr<double>;
template class opr<std::complex<double>>;
