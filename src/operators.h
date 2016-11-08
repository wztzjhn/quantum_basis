#ifndef OPERATORS_H
#define OPERATORS_H
#include <complex>
#include <vector>
#include <list>
#include "basis.h"

// ---------------- fundamental class for operators ------------------
// an operator on a given site and orbital
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

// -------------- class for a combination of operators ----------------
// a linear combination of products of operators
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// I sacrificed the efficiency by assuming all matrices in this class have the same type, think later how we can improve
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
template <typename T> class mopr {
public:
    
private:
    // the outer list represents the sum of operators
    // the innter list represents the product of operators
    std::list<std::list<opr<T>>> mats;
};



#endif
