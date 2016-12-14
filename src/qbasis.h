//
//  qbasis.h
//  qbasis
//
//  Created by Zhentao Wang on 11/21/16.
//
//

#ifndef qbasis_h
#define qbasis_h

#define MKL_Complex16 std::complex<double>

#ifndef lapack_int
#define lapack_int MKL_INT
#endif

#ifndef lapack_complex_double
#define lapack_complex_double   MKL_Complex16
#endif

#include <complex>
#include <string>
#include <vector>
#include <list>
#include <forward_list>
#include <utility>
#include <initializer_list>
#include <cassert>

#include <boost/dynamic_bitset.hpp>
#include "mkl.h"


namespace qbasis {

//  -------------part 0: global vals, forward declarations ---------------------
//  ----------------------------------------------------------------------------
    // the startup cost of boost::dynamic_bitset is around 40 Bytes
    using DBitSet = boost::dynamic_bitset<>;
    static const double opr_precision = 1e-12; // used as the threshold value in comparison
    static const double sparse_precision = 1e-14;
    static const double lanczos_precision = 1e-12;
    
    
    class basis_elem;
    class mbasis_elem;
    template <typename> class wavefunction;
    template <typename> class opr;
    template <typename> class opr_prod;
    template <typename> class mopr;
    template <typename> class csr_mat;
    
    bool operator<(const basis_elem&, const basis_elem&);
    bool operator==(const basis_elem&, const basis_elem&);
    
    bool operator<(const mbasis_elem&, const mbasis_elem&);
    bool operator==(const mbasis_elem&, const mbasis_elem&);
    
    template <typename T> void swap(wavefunction<T>&, wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const wavefunction<T>&, const T&);
    template <typename T> wavefunction<T> operator*(const T&, const wavefunction<T>&);
    
    
    
    
    // the following originally from operators.h
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

    // opr * | orb0, orb1, ..., ORB, ... > = | orb0, orb1, ..., opr*ORB, ... >, fermionic sign has to be computed when traversing orbitals
    template <typename T> wavefunction<T> operator*(const opr<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const opr<T>&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const opr_prod<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const opr_prod<T>&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const mopr<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const mopr<T>&, const wavefunction<T>&);
    
    
    
    
    
    
    
    
    


//  --------------------  part 1: basis of the wave functions ------------------
//  ----------------------------------------------------------------------------
    
    
    
    // ------------ fundamental class for basis elements --------------
    // for given number of sites, store the bits for a single orbital
    class basis_elem {
        friend void swap(basis_elem &lhs, basis_elem &rhs);
        friend bool operator<(const basis_elem&, const basis_elem&);
        friend bool operator==(const basis_elem&, const basis_elem&);
        //template <typename T> friend class opr;                                   // all instances of any type is a friend of basis_elem
        //template <typename T> friend T operator* (const opr<T>&, const mbasis_elem&);
        //template <typename T> friend T operator* (const opr_prod<T>&, const mbasis_elem&);
        template <typename T> friend wavefunction<T> operator*(const opr<T>&, const mbasis_elem&);
    public:
        // default constructor
        basis_elem() = default;
        
        // constructor (initializer for all site to be the 0th state)
        // with total number of sites, local dimension of Hilbert space,
        // and Nfermion is the fermion density operator (single site)
        basis_elem(const MKL_INT &n_sites, const MKL_INT &dim_local_);                              // boson
        basis_elem(const MKL_INT &n_sites, const MKL_INT &dim_local_, const opr<double> &Nfermion); // fermion
        
        // constructor from total number of sites and a given name.
        // current choices:
        // "spin-1/2", "spin-1"
        basis_elem(const MKL_INT &n_sites, const std::string &s);
        
        // copy constructor
        basis_elem(const basis_elem& old) :
        dim_local(old.dim_local), bits_per_site(old.bits_per_site),
        Nfermion_map(old.Nfermion_map), bits(old.bits) {}
        
        // move constructor
        basis_elem(basis_elem &&old) noexcept :
        dim_local(old.dim_local), bits_per_site(old.bits_per_site),
        Nfermion_map(std::move(old.Nfermion_map)), bits(std::move(old.bits)) {}
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        basis_elem& operator=(basis_elem old)
        {
            swap(*this, old);
            return *this;
        }
        
        // destructor
        ~basis_elem() {}
        
        MKL_INT total_sites() const;
        
        bool q_fermion() const {return (! Nfermion_map.empty()); }
        
        MKL_INT local_dimension() const { return static_cast<MKL_INT>(dim_local); }
        
        // read out the status of a particular site
        // storage order: site0, site1, site2, ..., site(N-1)
        MKL_INT siteRead(const MKL_INT &site) const;
        
        void siteWrite(const MKL_INT &site, const MKL_INT &val);
        
        void prt() const;
        
        // remember to change odd_fermion[site]
        //basis_elem& flip();
        
    private:
        short dim_local;
        short bits_per_site;
        std::vector<int> Nfermion_map;     // Nfermion_map[i] corresponds to the number of fermions on state i
        DBitSet bits;
    };
    
    
    
    
    // -------------- class for basis with several orbitals---------------
    // for given number of sites, and several orbitals, store the vectors of bits
    class mbasis_elem {
        friend void swap(mbasis_elem&, mbasis_elem&);
        friend bool operator<(const mbasis_elem&, const mbasis_elem&);
        friend bool operator==(const mbasis_elem&, const mbasis_elem&);
        //template <typename T> friend class opr;
        //template <typename T> friend T operator* (const opr<T>&, const mbasis_elem&);
        //template <typename T> friend T operator* (const opr_prod<T>&, const mbasis_elem&);
        template <typename T> friend wavefunction<T> operator*(const opr<T>&, const mbasis_elem&);
    public:
        // default constructor
        mbasis_elem() = default;
        
        // construcutor with total number of sites, and name  of each orbital
        mbasis_elem(const MKL_INT &n_sites, std::initializer_list<std::string> lst);
        
        // copy constructor
        mbasis_elem(const mbasis_elem& old): mbits(old.mbits) {}
        
        // move constructor
        mbasis_elem(mbasis_elem &&old) noexcept : mbits(std::move(old.mbits)) {}
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        mbasis_elem& operator=(mbasis_elem old)
        {
            swap(*this, old);
            return *this;
        }
        
        // destructor
        ~mbasis_elem() {}
        
        double diagonal_operator(const opr<double>& lhs) const;
        std::complex<double> diagonal_operator(const opr<std::complex<double>>& lhs) const;
        
        double diagonal_operator(const opr_prod<double>& lhs) const;
        std::complex<double> diagonal_operator(const opr_prod<std::complex<double>>& lhs) const;
        
        MKL_INT total_sites() const;
        MKL_INT total_orbitals() const;
        void test() const;
        
    private:
        // store an array of basis elements, for multi-orbital site (or unit cell)
        std::vector<basis_elem> mbits;
    };
    
    
    
    //      BETTER USE SMART POINTERS TO POINT TO THE ORIGINAL BASIS
    // -------------- class for wave functions ---------------
    // Use with caution, may hurt speed when not used properly
    template <typename T> class wavefunction {
        friend void swap <> (wavefunction<T> &, wavefunction<T> &);
        friend wavefunction<T> operator* <> (const opr<T>&, const wavefunction<T>&);
        friend wavefunction<T> operator* <> (const opr_prod<T>&, const wavefunction<T>&);
        friend wavefunction<T> operator* <> (const mopr<T>&, const wavefunction<T>&);
    public:
        // default constructor
        wavefunction() = default;
        
        // constructor from an element
        wavefunction(const mbasis_elem &old) : elements(1, std::pair<mbasis_elem, T>(old, static_cast<T>(1.0))) {}
        
        // copy constructor
        wavefunction(const wavefunction<T> &old) : elements(old.elements) {}
        
        // move constructor
        wavefunction(wavefunction<T> &&old) noexcept : elements(std::move(old.elements)) {}
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        wavefunction& operator=(wavefunction<T> old)
        {
            swap(*this, old);
            return *this;
        }
        
        // destructor
        ~wavefunction() {}
        
        // add one element
        wavefunction& operator+=(std::pair<mbasis_elem, T> ele);
        
        // add a wave function
        wavefunction& operator+=(wavefunction<T> rhs);
        
        // multiply by a constant
        wavefunction& operator*=(const T &rhs);
        
        // check if sorted
        bool sorted() const;
        
    private:
        // store an array of basis elements, and their corresponding coefficients
        // note: there should not be any dulplicated elements
        std::list<std::pair<mbasis_elem, T>> elements;
    };
    
    
    
    
//  -----------------------  part 2: basis of the operators --------------------
//  ----------------------------------------------------------------------------
    
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
    
    
    
    
//  --------------------------  part 3: sparse matrices ------------------------
//  ----------------------------------------------------------------------------
    
    template <typename T> struct lil_mat_elem {
        T val;
        MKL_INT col;
    };
    
    template <typename T> class lil_mat {
        friend class csr_mat<T>;
    public:
        // default constructor
        lil_mat() = default;
        
        // constructor with the Hilbert space dimension
        lil_mat(const MKL_INT &n, bool sym_ = false) : dim(n), nnz(n), sym(sym_),
        mat(std::vector<std::forward_list<lil_mat_elem<T>>>(n, std::forward_list<lil_mat_elem<T>>(1)))
        {
            mat.shrink_to_fit();
            for (MKL_INT i = 0; i < n; i++) {
                mat[i].front().col = i;
                mat[i].front().val = 0.0;
            }
        }
        
        // add one element
        void add(const MKL_INT &row, const MKL_INT &col, const T &val);
        
        
        // explicitly destroy, free space
        void destroy()
        {
            mat.clear();
            mat.shrink_to_fit();
        }
        
        // destructor
        ~lil_mat() {};
        
        MKL_INT dimension() const { return dim; }
        
        MKL_INT num_nonzero() const { return nnz; }
        
        // print
        void prt() const;
        
        void use_full_matrix() {sym = false; }
        
    private:
        MKL_INT dim;    // dimension of the matrix
        MKL_INT nnz;    // number of non-zero entries
        bool sym;       // if storing only upper triangle
        std::vector<std::forward_list<lil_mat_elem<T>>> mat;
    };
    
    
    // 3-array form of csr sparse matrix format, zero based
    template <typename T> class csr_mat {
        //    friend void csrXvec <> (const csr_mat<T>&, const std::vector<T>&, std::vector<T>&);
    public:
        // default constructor
        csr_mat() = default;
        
        // construcotr from a lil_mat, and if sym_ == true, use only the upper triangle
        csr_mat(const lil_mat<T> &old);
        
        // explicitly destroy, free space
        void destroy()
        {
            if(val != nullptr) {
                delete [] val;
                val = nullptr;
            }
            if(ja != nullptr){
                delete [] ja;
                ja = nullptr;
            }
            if(ia != nullptr){
                delete [] ia;
                ia = nullptr;
            }
        }
        
        // destructor
        ~csr_mat()
        {
            if(val != nullptr) delete [] val;
            if(ja != nullptr) delete [] ja;
            if(ia != nullptr) delete [] ia;
        }
        
        // matrix vector product
        void MultMv(const T *x, T *y) const;
        void MultMv(T *x, T *y);  // to be compatible with arpack++
        
        // matrix matrix product, x and y of shape dim * n
        //void MultMm(const T *x, T *y, MKL_INT n) const;
        
        
        MKL_INT dimension() const {return dim; }
        
        // print
        void prt() const;
        
    private:
        MKL_INT dim;
        MKL_INT nnz;        // number of non-zero entries
        bool sym;           // if storing only upper triangle
        T *val;
        MKL_INT *ja;
        MKL_INT *ia;
    };
    
    
    
//  ------------------------------part 4: Lanczos ------------------------------
//  ----------------------------------------------------------------------------
    
    // Note: sparse matrices in this code are using zero-based convention
    
    // By default, all diagonal elements are stored, even if they are zero (to be compatible with pardiso, if used in future)
    
    
    // m = k + np step of Lanczos
    // v of length m+1, hessenberg matrix of size m*m (m-step Lanczos)
    // after decomposition, mat * v[0:m-1] = v[0:m-1] * hessenberg + rnorm * resid * e_m^T,
    // where e_m has only one nonzero element: e[0:m-2] == 0, e[m-1] = 1
    
    // on entry, assuming k steps of Lanczos already performed:
    // v_0, ..., v_{k-1} stored in v, v{k} stored in resid
    // alpha_0, ..., alpha_{k-1} in hessenberg matrix
    // beta_1,  ..., beta_{k-1} in hessenberg matrix, beta_k as rnorm
    // if on entry k==0, then beta_k=rnorm=0, v_0=resid
    
    // ldh: leading dimension of hessenberg
    // alpha[j] = hessenberg[j+ldh], diagonal of hessenberg matrix
    // beta[j]  = hessenberg[j]
    //  a[0]  b[1]      -> note: beta[0] not used
    //  b[1]  a[1]  b[2]
    //        b[2]  a[2]  b[3]
    //              b[3]  a[3] b[4]
    //                    ..  ..  ..    b[k-1]
    //                          b[k-1]  a[k-1]
    template <typename T>
    void lanczos(MKL_INT k, MKL_INT np, const csr_mat<T> &mat, double &rnorm, T resid[],
                 T v[], double hessenberg[], const MKL_INT &ldh, const bool &MemoSteps = true);
    
    // if possible, add a block Arnoldi version here
    
    // transform from band storage to general storage
    template <typename T>
    void hess2matform(const double hessenberg[], T mat[], const MKL_INT &m, const MKL_INT &ldh);
    
    // compute eigenvalues (and optionally eigenvectors, stored in s) of hessenberg matrix
    // on entry, hessenberg and s should have the same leading dimension: ldh
    // order = "sm", "lm", "sr", "lr", where 's': small, 'l': large, 'm': magnitude, 'r': real part
    void select_shifts(const double hessenberg[], const MKL_INT &ldh, const MKL_INT &m,
                       const std::string &order, double ritz[], double s[] = nullptr);
    
    // --------------------------
    // ideally, here we should use the bulge-chasing algorithm; for this moment, we simply use the less efficient brute force QR factorization
    // --------------------------
    // QR factorization of hessenberg matrix, using np selected eigenvalues from ritz
    // [H, Q] = QR(H, shift1, shift2, ..., shift_np)
    // \tilde{H} = Q_np^T ... Q_1^T H Q_1 ... Q_np
    // \tilde{V} = V Q
    template <typename T>
    void perform_shifts(const MKL_INT &dim, const MKL_INT &m, const MKL_INT &np, const double shift[],
                        double &rnorm, T resid[], T v[], double hessenberg[], const MKL_INT &ldh,
                        double Q[], const MKL_INT &ldq);
    
    // implicitly restarted Arnoldi method
    // nev: number of eigenvalues needed
    // ncv: length of each individual lanczos process
    // 2 < nev + 2 <= ncv
    // when not using arpack++, we can modify the property of mat to be const
    template <typename T>
    void iram(csr_mat<T> &mat, T v0[], const MKL_INT &nev, const MKL_INT &ncv, MKL_INT &nconv,
              const std::string &order, double eigenvals[], T eigenvecs[], const bool &use_arpack = true);
    
    
    
    
    
//  ---------------------------part 5: Measurements ----------------------------
//  ----------------------------------------------------------------------------
    //             b1
    // a0 +  ---------------
    //                b2
    //       a1 + ----------
    //            a2 + ...
    template <typename T>
    T continued_fraction(T a[], T b[], const MKL_INT &len); // b0 not used
    
    
    
    
    
}





//  -------------------------  interface to mkl library ------------------------
//  ----------------------------------------------------------------------------
namespace qbasis {
    // blas level 1, y = a*x + y
    inline // double
    void axpy(const MKL_INT n, const double alpha, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
        daxpy(&n, &alpha, x, &incx, y, &incy);
    }
    inline // complex double
    void axpy(const MKL_INT n, const std::complex<double> alpha, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy) {
        zaxpy(&n, &alpha, x, &incx, y, &incy);
    }
    
    
    // blas level 1, y = x
    inline // double
    void copy(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
        dcopy(&n, x, &incx, y, &incy);
    }
    inline // complex double
    void copy(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy) {
        zcopy(&n, x, &incx, y, &incy);
    }
    
    // blas level 1, Euclidean norm of vector
    inline // double
    double nrm2(const MKL_INT n, const double *x, const MKL_INT incx) {
        return dnrm2(&n, x, &incx);
    }
    inline // complex double
    double nrm2(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx) {
        return dznrm2(&n, x, &incx);
    }
    
    // blas level 1, rescale: x = a*x
    inline // double * double vector
    void scal(const MKL_INT n, const double a, double *x, const MKL_INT incx) {
        dscal(&n, &a, x, &incx);
    }
    inline // double complex * double complex vector
    void scal(const MKL_INT n, const std::complex<double> a, std::complex<double> *x, const MKL_INT incx) {
        zscal(&n, &a, x, &incx);
    }
    inline // double * double complex vector
    void scal(const MKL_INT n, const double a, std::complex<double> *x, const MKL_INT incx) {
        zdscal(&n, &a, x, &incx);
    }
    
    
    // blas level 1, conjugated vector dot vector
    inline // double
    double dotc(const MKL_INT n, const double *x, const MKL_INT incx, const double *y, const MKL_INT incy) {
        return ddot(&n, x, &incx, y, &incy);
    }
    inline // complex double
    std::complex<double> dotc(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx, const std::complex<double> *y, const MKL_INT incy) {
        std::complex<double> result;
        zdotc(&result, &n, x, &incx, y, &incy);
        return result;
    }
    
    // blas level 3, matrix matrix product
    inline // double
    void gemm(const char transa, const char transb, const MKL_INT m, const MKL_INT n, const MKL_INT k,
              const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb,
              const double beta, double *c, const MKL_INT ldc) {
        dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
    inline // complex double
    void gemm(const char transa, const char transb, const MKL_INT m, const MKL_INT n, const MKL_INT k,
              const std::complex<double> alpha, const std::complex<double> *a, const MKL_INT lda,
              const std::complex<double> *b, const MKL_INT ldb,
              const std::complex<double> beta, std::complex<double> *c, const MKL_INT ldc) {
        zgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
    
    
    // sparse blas routines
    inline // double
    void csrgemv(const char transa, const MKL_INT m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y) {
        mkl_cspblas_dcsrgemv(&transa, &m, a, ia, ja, x, y);
    }
    inline // complex double
    void csrgemv(const char transa, const MKL_INT m, const std::complex<double> *a, const MKL_INT *ia, const MKL_INT *ja, const std::complex<double> *x, std::complex<double> *y) {
        mkl_cspblas_zcsrgemv(&transa, &m, a, ia, ja, x, y);
    }
    
    // for symmetric matrix (NOT Hermitian matrix)
    inline // double
    void csrsymv(const char uplo, const MKL_INT m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y) {
        mkl_cspblas_dcsrsymv(&uplo, &m, a, ia, ja, x, y);
    }
    inline // complex double
    void csrsymv(const char uplo, const MKL_INT m, const std::complex<double> *a, const MKL_INT *ia, const MKL_INT *ja, const std::complex<double> *x, std::complex<double> *y) {
        mkl_cspblas_zcsrsymv(&uplo, &m, a, ia, ja, x, y);
    }
    
    // more general function to perform matrix vector product in mkl
    inline // double
    void mkl_csrmv(const char transa, const MKL_INT m, const MKL_INT k, const double alpha, const char *matdescra,
                   const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
                   const double *x, const double beta, double *y) {
        mkl_dcsrmv(&transa, &m, &k, &alpha, matdescra, val, indx, pntrb, pntre, x, &beta, y);
    }
    inline // complex double
    void mkl_csrmv(const char transa, const MKL_INT m, const MKL_INT k, const std::complex<double> alpha, const char *matdescra,
                   const std::complex<double> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
                   const std::complex<double> *x, const std::complex<double> beta, std::complex<double> *y) {
        mkl_zcsrmv(&transa, &m, &k, &alpha, matdescra, val, indx, pntrb, pntre, x, &beta, y);
    }
    
    inline // double
    void mkl_csrmm(const char transa, const MKL_INT m, const MKL_INT n, const MKL_INT k, const double alpha, const char *matdescra,
                   const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
                   const double *b, const MKL_INT ldb, const double beta, double *c, const MKL_INT ldc) {
        mkl_dcsrmm(&transa, &m, &n, &k, &alpha, matdescra, val, indx, pntrb, pntre, b, &ldb, &beta, c, &ldc);
    }
    inline // complex double
    void mkl_csrmm(const char transa, const MKL_INT m, const MKL_INT n, const MKL_INT k, const std::complex<double> alpha, const char *matdescra,
                   const std::complex<double> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
                   const std::complex<double> *b, const MKL_INT ldb, const std::complex<double> beta, std::complex<double> *c, const MKL_INT ldc) {
        mkl_zcsrmm(&transa, &m, &n, &k, &alpha, matdescra, val, indx, pntrb, pntre, b, &ldb, &beta, c, &ldc);
    }
    
    // sparse blas, convert csr to csc
    inline // double
    void mkl_csrcsc(const MKL_INT *job, const MKL_INT n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0,
                    double *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info) {
        mkl_dcsrcsc(job, &n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info);
    }
    inline // complex double
    void mkl_csrcsc(const MKL_INT *job, const MKL_INT n, std::complex<double> *Acsr, MKL_INT *AJ0, MKL_INT *AI0,
                    std::complex<double> *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info) {
        mkl_zcsrcsc(job, &n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info);
    }
    
    // lapack computational routine, computes all eigenvalues of a real symmetric tridiagonal matrix using QR algorithm.
    inline // double
    lapack_int sterf(const lapack_int &n, double *d, double *e) {
        return LAPACKE_dsterf(n, d, e);
    }
    
    // lapack computational routine, computes all eigenvalues and (optionally) eigenvectors of a symmetric/hermitian tridiagonal matrix using the divide and conquer method.
    inline // double
    lapack_int stedc(const int &matrix_layout, const char &compz, const lapack_int &n, double *d, double *e, double *z, const lapack_int &ldz) {
        return LAPACKE_dstedc(matrix_layout, compz, n, d, e, z, ldz);
    }
    inline // complex double (for the unitary matrix which brings the original matrix to tridiagonal form)
    lapack_int stedc(const int &matrix_layout, const char &compz, const lapack_int &n, double *d, double *e, std::complex<double> *z, const lapack_int &ldz) {
        return LAPACKE_zstedc(matrix_layout, compz, n, d, e, z, ldz);
    }
    
    
    
    //// lapack symmetric eigenvalue driver routine, using divide and conquer, for band matrix
    //inline // double
    //lapack_int bevd(const int &matrix_layout, const char &jobz, const char &uplo, const lapack_int &n, const lapack_int &kd,
    //                double *ab, const lapack_int &ldab, double *w, double *z, const lapack_int &ldz) {
    //    return LAPACKE_dsbevd(matrix_layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz);
    //}
    //lapack_int bevd(const int &matrix_layout, const char &jobz, const char &uplo, const lapack_int &n, const lapack_int &kd,
    //                std::complex<double> *ab, const lapack_int &ldab, double *w, std::complex<double> *z, const lapack_int &ldz) {
    //    return LAPACKE_zhbevd(matrix_layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz);
    //}
    
    
    // lapack, Computes the QR factorization of a general m-by-n matrix.
    inline // double
    lapack_int geqrf(const int &matrix_layout, const lapack_int &m, const lapack_int &n, double *a, const lapack_int &lda, double *tau) {
        return LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);
    }
    inline // complex double
    lapack_int geqrf(const int &matrix_layout, const lapack_int &m, const lapack_int &n, std::complex<double> *a, const lapack_int &lda, std::complex<double> *tau) {
        return LAPACKE_zgeqrf(matrix_layout, m, n, a, lda, tau);
    }
    
    // lapack, Multiplies a real matrix by the orthogonal matrix Q of the QR factorization formed by ?geqrf or ?geqpf.
    inline // double
    lapack_int ormqr(const int &matrix_layout, const char &side, const char &trans,
                     const lapack_int &m, const lapack_int &n, const lapack_int &k,
                     const double *a, const lapack_int &lda, const double *tau, double *c, const lapack_int &ldc) {
        return LAPACKE_dormqr(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }
}




// Note: to include the functions required to call this library

#endif /* qbasis_h */
