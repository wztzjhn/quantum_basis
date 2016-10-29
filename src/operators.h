#ifndef OPERATORS_H
#define OPERATORS_H
#include <complex>
#include <vector>
#include "basis.h"

template <typename T> class opr {
public:
    opr() = default;
    opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<T> &mat_);
    opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<std::vector<T>> &mat_);
    
    void test() const;
    
private:
    int site;      // site No.
    int orbital;   // orbital No.
    bool fermion;  // fermion or not
    bool diagonal; // diagonal in matrix form
    T *mat;        // matrix form, or diagonal elements if diagonal
    
    
    
    
};




#endif
