#include "sparse.h"


// --------- implementation of class lil (list of list sparse matrix data structure) ----------
template <typename T>
void lil_mat<T>::add(const int &row, const int &col, const T &val)
{
    mat.clear();
    mat.shrink_to_fit();
}

