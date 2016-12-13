#ifndef OPERATORS_H
#define OPERATORS_H
#include <complex>
#include <vector>
#include <list>
#include "basis.h"
#include "mkl_interface.h"

namespace qbasis {
    static const double opr_precision = 1e-12; // used as the threshold value in comparison
    
    // forward declarations
    class basis_elem;
    class mbasis_elem;
    template <typename> class wavefunction;
    template <typename> class opr;
    template <typename> class opr_prod;
    template <typename> class mopr;
    
    template <typename T> void swap(opr<T>&, opr<T>&);
    template <typename T> bool operator==(const opr<T>&, const opr<T>&);
    template <typename T> bool operator!=(const opr<T>&, const opr<T>&);
    template <typename T> bool operator<(const opr<T>&, const opr<T>&); // only compares site, orbital and fermion, for sorting purpose
    template <typename T> opr<T> operator*(const T&, const opr<T>&);
    template <typename T> opr<T> operator*(const opr<T>&, const T&);
    template <typename T> opr<T> normalize(const opr<T>&, T&); // sum_{i,j} mat[i,j]^2 == dim; the 1st nonzero element (in memory) be real positive
    
    template <typename T> void swap(opr_prod<T>&, opr_prod<T>&);
    template <typename T> bool operator==(const opr_prod<T>&, const opr_prod<T>&);
    template <typename T> bool operator!=(const opr_prod<T>&, const opr_prod<T>&);
    template <typename T> bool operator<(const opr_prod<T>&, const opr_prod<T>&); // compare only length, and if each lhs.mat_prod < rhs.mat_prod
    template <typename T> opr_prod<T> operator*(const opr_prod<T>&, const opr_prod<T>&);
    template <typename T> opr_prod<T> operator*(const opr_prod<T>&, const opr<T>&);
    template <typename T> opr_prod<T> operator*(const opr<T>&, const opr_prod<T>&);
    template <typename T> opr_prod<T> operator*(const opr_prod<T>&, const T&);
    template <typename T> opr_prod<T> operator*(const T&, const opr_prod<T>&);
    template <typename T> opr_prod<T> operator*(const opr<T>&, const opr<T>&);       // cast up
    
    template <typename T> void swap(mopr<T>&, mopr<T>&);
    template <typename T> bool operator==(const mopr<T>&, const mopr<T>&);
    template <typename T> bool operator!=(const mopr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator+(const mopr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator+(const mopr<T>&, const opr_prod<T>&);
    template <typename T> mopr<T> operator+(const opr_prod<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator+(const mopr<T>&, const opr<T>&);
    template <typename T> mopr<T> operator+(const opr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator+(const opr_prod<T>&, const opr_prod<T>&); // cast up
    template <typename T> mopr<T> operator+(const opr_prod<T>&, const opr<T>&);      // cast up
    template <typename T> mopr<T> operator+(const opr<T>&, const opr_prod<T>&);      // cast up
    template <typename T> mopr<T> operator+(const opr<T>&, const opr<T>&);           // cast up
    template <typename T> mopr<T> operator-(const mopr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator-(const mopr<T>&, const opr_prod<T>&);
    template <typename T> mopr<T> operator-(const opr_prod<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator-(const mopr<T>&, const opr<T>&);
    template <typename T> mopr<T> operator-(const opr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator-(const opr_prod<T>&, const opr_prod<T>&); // cast up
    template <typename T> mopr<T> operator-(const opr_prod<T>&, const opr<T>&);      // cast up
    template <typename T> mopr<T> operator-(const opr<T>&, const opr_prod<T>&);      // cast up
    template <typename T> mopr<T> operator-(const opr<T>&, const opr<T>&);           // cast up
    template <typename T> mopr<T> operator*(const mopr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator*(const mopr<T>&, const opr_prod<T>&);
    template <typename T> mopr<T> operator*(const opr_prod<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator*(const mopr<T>&, const opr<T>&);
    template <typename T> mopr<T> operator*(const opr<T>&, const mopr<T>&);
    
    template <typename T> wavefunction<T> operator*(const opr<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const opr_prod<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const opr_prod<T>&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const mopr<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const mopr<T>&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const mopr<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const mopr<T>&, const wavefunction<T>&);
    // note:
    // zero operator: mat==nullptr
    
    // ---------------- fundamental class for operators ------------------
    // an operator on a given site and orbital
    template <typename T> class opr {
        friend void swap <> (opr<T>&, opr<T>&);
        friend bool operator== <> (const opr<T>&, const opr<T>&);
        friend bool operator!= <> (const opr<T>&, const opr<T>&);
        friend bool operator< <> (const opr<T>&, const opr<T>&);
        friend opr<T> operator* <> (const T&, const opr<T>&);
        friend opr<T> operator* <> (const opr<T>&, const T&);
        friend opr<T> normalize <> (const opr<T>&, T&);
        friend class opr_prod<T>;
        friend class mopr<T>;
        friend class basis_elem;
        friend class mbasis_elem;
        //friend T operator* <> (const opr<T>&, const basis_elem&);
        //friend T operator* <> (const opr<T>&, const mbasis_elem&);
        //friend T operator* <> (const opr_prod<T>&, const mbasis_elem&);
        friend wavefunction<T> operator* <> (const opr<T>&, const mbasis_elem&);
        
    public:
        // default constructor
        opr() = default;
        
        // constructor from diagonal elements
        opr(const MKL_INT &site_, const MKL_INT &orbital_, const bool &fermion_, const std::vector<T> &mat_);
        
        // constructor from a matrix
        opr(const MKL_INT &site_, const MKL_INT &orbital_, const bool &fermion_, const std::vector<std::vector<T>> &mat_);
        
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
        
        // change site index
        opr& change_site(const MKL_INT &site_);
        
        // question if it is identity operator
        bool q_diagonal() const
        {
            return diagonal;
        }
        
        // question if it is identity operator
        bool q_identity() const;
        
        // question if it is zero operator
        bool q_zero() const;
        
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
        MKL_INT site;      // site No.
        MKL_INT orbital;   // orbital No.
        MKL_INT dim;       // number of rows(columns) of the matrix
        bool fermion;      // fermion or not
        bool diagonal;     // diagonal in matrix form
        T *mat;            // matrix form, or diagonal elements if diagonal
    };
    
    
    // -------------- class for operator products ----------------
    // note: when mat_prod is empty, it represents identity operator, with coefficient coeff
    
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // I sacrificed the efficiency by assuming all matrices in this class have the same type, think later how we can improve
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    template <typename T> class opr_prod {
        friend void swap <> (opr_prod<T>&, opr_prod<T>&);
        friend bool operator== <> (const opr_prod<T>&, const opr_prod<T>&);
        friend bool operator!= <> (const opr_prod<T>&, const opr_prod<T>&);
        friend bool operator< <> (const opr_prod<T>&, const opr_prod<T>&);
        friend opr_prod<T> operator* <> (const opr_prod<T>&, const opr_prod<T>&);
        friend opr_prod<T> operator* <> (const opr_prod<T>&, const opr<T>&);
        friend opr_prod<T> operator* <> (const opr<T>&, const opr_prod<T>&);
        friend opr_prod<T> operator* <> (const opr_prod<T>&, const T&);
        friend opr_prod<T> operator* <> (const T&, const opr_prod<T>&);
        friend opr_prod<T> operator* <> (const opr<T>&, const opr<T>&);
        friend class mopr<T>;
        friend class mbasis_elem;
        //friend T operator* <> (const opr_prod<T>&, const mbasis_elem&);
        friend wavefunction<T> operator* <> (const opr_prod<T>&, const mbasis_elem&);
        friend wavefunction<T> operator* <> (const opr_prod<T>&, const wavefunction<T>&);
    public:
        // default constructor
        opr_prod() = default;
        
        // constructor from one fundamental operator
        opr_prod(const opr<T> &ele);
        
        // copy constructor
        opr_prod(const opr_prod<T> &old): coeff(old.coeff), mat_prod(old.mat_prod) {}
        
        // move constructor
        opr_prod(opr_prod<T> &&old) noexcept : coeff(std::move(old.coeff)), mat_prod(std::move(old.mat_prod)) {}
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        opr_prod& operator=(opr_prod<T> old)
        {
            swap(*this, old);
            return *this;
        }
        
        // compound assignment operators
        opr_prod& operator*=(opr<T> rhs);
        opr_prod& operator*=(opr_prod<T> rhs); // in this form to avoid self-assignment
        opr_prod& operator*=(const T &rhs);
        
        // invert the sign
        opr_prod& negative();
        
        // question if it is proportional to identity operator
        bool q_prop_identity() const;
        
        // question if it is zero operator
        bool q_zero() const;
        
        MKL_INT len() const;
        
        // destructor
        ~opr_prod() {}
        
        void prt() const;
        
    private:
        T coeff;
        std::list<opr<T>> mat_prod; // each opr<T> should be normalized
    };
    
    
    // -------------- class for a combination of operators ----------------
    // a linear combination of products of operators
    template <typename T> class mopr {
        friend void swap <> (mopr<T>&, mopr<T>&);
        friend bool operator== <> (const mopr<T>&, const mopr<T>&);
        friend bool operator!= <> (const mopr<T>&, const mopr<T>&);
        friend mopr<T> operator+ <> (const mopr<T>&, const mopr<T>&);
        friend mopr<T> operator+ <> (const mopr<T>&, const opr_prod<T>&);
        friend mopr<T> operator+ <> (const opr_prod<T>&, const mopr<T>&);
        friend mopr<T> operator+ <> (const mopr<T>&, const opr<T>&);
        friend mopr<T> operator+ <> (const opr<T>&, const mopr<T>&);
        friend mopr<T> operator+ <> (const opr_prod<T>&, const opr_prod<T>&);
        friend mopr<T> operator+ <> (const opr_prod<T>&, const opr<T>&);
        friend mopr<T> operator+ <> (const opr<T>&, const opr_prod<T>&);
        friend mopr<T> operator+ <> (const opr<T>&, const opr<T>&);
        friend mopr<T> operator- <> (const mopr<T>&, const mopr<T>&);
        friend mopr<T> operator- <> (const mopr<T>&, const opr_prod<T>&);
        friend mopr<T> operator- <> (const opr_prod<T>&, const mopr<T>&);
        friend mopr<T> operator- <> (const mopr<T>&, const opr<T>&);
        friend mopr<T> operator- <> (const opr<T>&, const mopr<T>&);
        friend mopr<T> operator- <> (const opr_prod<T>&, const opr_prod<T>&);
        friend mopr<T> operator- <> (const opr_prod<T>&, const opr<T>&);
        friend mopr<T> operator- <> (const opr<T>&, const opr_prod<T>&);
        friend mopr<T> operator- <> (const opr<T>&, const opr<T>&);
        friend mopr<T> operator* <> (const mopr<T>&, const mopr<T>&);
        friend mopr<T> operator* <> (const mopr<T>&, const opr_prod<T>&);
        friend mopr<T> operator* <> (const opr_prod<T>&, const mopr<T>&);
        friend mopr<T> operator* <> (const mopr<T>&, const opr<T>&);
        friend mopr<T> operator* <> (const opr<T>&, const mopr<T>&);
        friend wavefunction<T> operator* <> (const mopr<T>&, const mbasis_elem&);
        friend wavefunction<T> operator* <> (const mopr<T>&, const wavefunction<T>&);
    public:
        // default constructor
        mopr() = default;
        
        // constructor from one fundamental operator
        mopr(const opr<T> &ele);
        
        // constructor from operator products
        mopr(const opr_prod<T> &ele);
        
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
        mopr& operator+=(opr<T> rhs);
        mopr& operator+=(opr_prod<T> rhs);
        mopr& operator+=(mopr<T> rhs);
        mopr& operator-=(opr<T> rhs);
        mopr& operator-=(opr_prod<T> rhs);
        mopr& operator-=(mopr<T> rhs);
        mopr& operator*=(opr<T> rhs);
        mopr& operator*=(opr_prod<T> rhs);
        mopr& operator*=(mopr<T> rhs);
        mopr& operator*=(const T &rhs);
        
        opr_prod<T>& operator[](MKL_INT n)
        {
            assert(! mats.empty());
            auto it = mats.begin();
            for (decltype(n) i = 0; i < n; i++) ++it;
            return *it;
        }
        
        const opr_prod<T>& operator[](MKL_INT n) const
        {
            assert(! mats.empty());
            auto it = mats.begin();
            for (decltype(n) i = 0; i < n; i++) ++it;
            return *it;
        }
        
        
        // simplify, need implementation
        mopr& simplify();
        
        // invert the sign
        mopr& negative();
        
        // destructor
        ~mopr() {}
        
        bool q_zero() const {return mats.empty(); }
        
        void prt() const;
        
        
    private:
        // the outer list represents the sum of operators, inner data structure taken care by operator products
        std::list<opr_prod<T>> mats;
    };

}

#endif
