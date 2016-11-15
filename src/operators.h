#ifndef OPERATORS_H
#define OPERATORS_H
#include <complex>
#include <vector>
#include <list>
#include "basis.h"
#include "mkl_interface.h"


static const double opr_precision = 1e-12; // used as the threshold value in comparison

// forward declarations
template <typename> class opr;
template <typename T> void swap(opr<T>&, opr<T>&); // which by itself is just a template function
template <typename T> bool operator==(const opr<T>&, const opr<T>&);
template <typename T> bool operator!=(const opr<T>&, const opr<T>&);
template <typename T> bool operator<(const opr<T>&, const opr<T>&); // only compares site, orbital and fermion, for sorting purpose
template <typename T> opr<T> operator+(const opr<T>&, const opr<T>&);
template <typename T> opr<T> operator-(const opr<T>&, const opr<T>&);
template <typename T> opr<T> operator*(const opr<T>&, const opr<T>&);
template <typename T> opr<T> operator*(const T&, const opr<T>&);
template <typename T> opr<T> operator*(const opr<T>&, const T&);
// sum_{i,j} mat[i,j]^2 == dim; and also require the 1st nonzero element (in memory) to be real positive
template <typename T> opr<T> normalize(const opr<T>&, T&);

template <typename> class opr_prod;
template <typename T> void swap(opr_prod<T>&, opr_prod<T>&);

template <typename> class mopr;
template <typename T> void swap(mopr<T>&, mopr<T>&);


// note:
// zero operator: mat==nullptr

// ---------------- fundamental class for operators ------------------
// an operator on a given site and orbital
template <typename T> class opr {
    friend void swap <> (opr<T>&, opr<T>&);
    friend bool operator== <> (const opr<T>&, const opr<T>&);
    friend bool operator!= <> (const opr<T>&, const opr<T>&);
    friend bool operator< <> (const opr<T>&, const opr<T>&);
    friend opr<T> operator+ <> (const opr<T>&, const opr<T>&);
    friend opr<T> operator- <> (const opr<T>&, const opr<T>&);
    friend opr<T> operator* <> (const opr<T>&, const opr<T>&);
    friend opr<T> operator* <> (const T&, const opr<T>&);
    friend opr<T> operator* <> (const opr<T>&, const T&);
    friend opr<T> normalize <> (const opr<T>&, T&);
    friend class opr_prod<T>;
    
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
    
    // \sqrt { sum_{i,j} |mat[i,j]|^2 }
    double norm() const;
    
    // invert the sign
    opr& negative();
    
    // question if it is identity operator
    bool q_identity() const;
    
    // simplify the structure if possible
    opr& simplify();
    

    // compound assignment operators
    opr& operator+=(const opr<T> &rhs);
    opr& operator-=(const opr<T> &rhs);
    opr& operator*=(const opr<T> &rhs);
    opr& operator*=(const T &rhs);
    
    // destructor
    ~opr() {if(mat != nullptr) delete [] mat;}
    
    void prt() const;
    
private:
    int site;      // site No.
    int orbital;   // orbital No.
    size_t dim;    // number of rows(columns) of the matrix
    bool fermion;  // fermion or not
    bool diagonal; // diagonal in matrix form
    T *mat;        // matrix form, or diagonal elements if diagonal
};


// -------------- class for operator products ----------------
// note: when mat_prod is empty, it represents identity operator, with coefficient coeff
template <typename T> class opr_prod {
    friend void swap <> (opr_prod<T>&, opr_prod<T>&);
    friend class mopr<T>;
public:
    // default constructor
    opr_prod() = default;
    
    // constructor from one fundamental operator
    opr_prod(const opr<T> &ele);
    
    // copy constructor
    opr_prod(const opr_prod<T> &old): mat_prod(old.mat_prod) {}
    
    // move constructor
    opr_prod(opr_prod<T> &&old) noexcept : mat_prod(std::move(old.mat_prod)) {}
    
    // copy assignment constructor and move assignment constructor, using "swap and copy"
    opr_prod& operator=(opr_prod<T> old)
    {
        swap(*this, old);
        return *this;
    }
    
    // compound assignment operators
    opr_prod& operator*=(const opr<T> &rhs);
    opr_prod& operator*=(const opr_prod<T> &rhs);
    
    // destructor
    ~opr_prod() {}
    
    void prt() const;
    
private:
    T coeff;
    std::list<opr<T>> mat_prod; // each opr<T> should be normalized
};


// -------------- class for a combination of operators ----------------
// a linear combination of products of operators
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// I sacrificed the efficiency by assuming all matrices in this class have the same type, think later how we can improve
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
template <typename T> class mopr {
    friend void swap <> (mopr<T>&, mopr<T>&);
public:
    // default constructor
    mopr() = default;
    
    // constructor from one fundamental operator
    mopr(const opr<T> &ele) { mats.emplace_back(ele); }
    
    // copy constructor
    mopr(const mopr<T> &old): mats(old.mats) {}
    
    // move constructor
    mopr(mopr<T> &&old) noexcept : mats(std::move(old.mats)) {}
    
    // copy assignment constructor and move assignment constructor, using "swap and copy"
    mopr& operator=(mopr<T> old)
    {
        swap(*this, old);
        return *this;
    }
    
    // compound assignment operators
    mopr& operator+=(const opr<T> &rhs);
    mopr& operator*=(const opr<T> &rhs);
    mopr& operator+=(const mopr<T> &rhs);
    
    // destructor
    ~mopr() {}
    
    void prt() const;
    
    
private:
    // the outer list represents the sum of operators, inner data structure taken care by operator products
    std::list<opr_prod<T>> mats;
};



#endif
