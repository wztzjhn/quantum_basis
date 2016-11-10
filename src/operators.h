#ifndef OPERATORS_H
#define OPERATORS_H
#include <complex>
#include <vector>
#include <list>
#include "basis.h"
#define MKL_INT size_t
#define MKL_Complex16 std::complex<double>
#include "mkl.h"

// ---------------- fundamental class for operators ------------------
// an operator on a given site and orbital
template <typename T> class opr {
public:
    // default constructor
    opr() = default;
    
    // constructor from diagonal elements
    opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<T> &mat_);
    
    // constructor from a matrix
    opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<std::vector<T>> &mat_);
    
    
    // destructor
    ~opr() {delete [] mat; mat = nullptr;}
    
    void prt() const;
    
private:
    int site;      // site No.
    int orbital;   // orbital No.
    size_t m;      // number of rows of the matrix
    size_t n;      // number of cols of the matrix
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
