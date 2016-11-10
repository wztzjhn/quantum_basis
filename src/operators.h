#ifndef OPERATORS_H
#define OPERATORS_H
#include <complex>
#include <vector>
#include <list>
#include "basis.h"
#include "mkl_interface.h"


// ---------------- fundamental class for operators ------------------
// an operator on a given site and orbital

// declare friends before use
template <typename> class opr;
template <typename T> void swap(opr<T>&, opr<T>&); // which by itself is just a template function

template <typename T> class opr {
    friend void swap <> (opr<T>&, opr<T>&);
public:
    // default constructor
    opr() = default;
    
    // constructor from diagonal elements
    opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<T> &mat_);
    
    // constructor from a matrix
    opr(const int &site_, const int &orbital_, const bool &fermion_, const std::vector<std::vector<T>> &mat_);
    
    // copy constructor
    opr(const opr<T> &old);
    
    // move constructor
    opr(opr<T> &&old) noexcept;
    
    // copy assignment constructor and move assignment constructor, using "swap and copy"
    opr& operator=(opr<T> old)
    {
        swap(*this, old);
        return *this;
    }
    
    // destructor
    ~opr() {if(mat != nullptr) delete [] mat;}
    
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
