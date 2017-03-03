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
#include <chrono>
#include <cassert>
#include <boost/dynamic_bitset.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include "mkl.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() -1
#define omp_get_num_threads() -1
#define omp_get_num_procs() -1
#endif


// FUNCTIONS which need further change for Fermions:
// opr& transform(const std::vector<MKL_INT> &plan);
// opr_prod& transform(const std::vector<MKL_INT> &plan);
// basis_elem& transform(const std::vector<MKL_INT> &plan, MKL_INT &sgn);
// basis_elem& translate(const lattice &latt, const std::vector<MKL_INT> &disp, MKL_INT &sgn);
// mbasis_elem& transform(const std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> &plan, MKL_INT &sgn);

namespace qbasis {

//  -------------part 0: global vals, forward declarations ---------------------
//  ----------------------------------------------------------------------------
    // the startup cost of boost::dynamic_bitset is around 40 Bytes
    using DBitSet = boost::dynamic_bitset<>;
    static const double pi = 3.141592653589793238462643;
    // later let's combine these three as a unified name "precision"
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
    //class threads_pool;
    template <typename> class model;
    class lattice;
    
    bool operator<(const basis_elem&, const basis_elem&);
    bool operator==(const basis_elem&, const basis_elem&);
    bool operator!=(const basis_elem&, const basis_elem&);
    
    bool operator<(const mbasis_elem&, const mbasis_elem&);
    bool operator==(const mbasis_elem&, const mbasis_elem&);
    bool operator!=(const mbasis_elem&, const mbasis_elem&);
    bool trans_equiv(const mbasis_elem&, const mbasis_elem&, const lattice&);   // computational heavy, use with caution
    
    template <typename T> void swap(wavefunction<T>&, wavefunction<T>&);
    template <typename T> wavefunction<T> operator+(const wavefunction<T>&, const wavefunction<T>&);
    
    template <typename T> wavefunction<T> operator*(const wavefunction<T>&, const T&);
    template <typename T> wavefunction<T> operator*(const T&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const mbasis_elem&, const T&);
    template <typename T> wavefunction<T> operator*(const T&, const mbasis_elem&);
    
    
    
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
    template <typename T> mopr<T> operator*(const mopr<T>&, const T&);
    template <typename T> mopr<T> operator*(const T&, const mopr<T>&);

    // opr * | orb0, orb1, ..., ORB, ... > = | orb0, orb1, ..., opr*ORB, ... >, fermionic sign has to be computed when traversing orbitals
    template <typename T> wavefunction<T> operator*(const opr<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const opr<T>&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const opr_prod<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const opr_prod<T>&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const mopr<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const mopr<T>&, const wavefunction<T>&);
    
    // mopr * {a list of mbasis} -->> {a new list of mbasis}
    template <typename T> void gen_mbasis_by_mopr(const mopr<T>&, std::list<mbasis_elem>&);
    
    
    template <typename T> void swap(csr_mat<T>&, csr_mat<T>&);
    
    
    
    
    
    


//  --------------------  part 1: basis of the wave functions ------------------
//  ----------------------------------------------------------------------------
    
    
    
    // ------------ fundamental class for basis elements --------------
    // for given number of sites, store the bits for a single orbital
    class basis_elem {
        friend void swap(basis_elem &lhs, basis_elem &rhs);
        friend bool operator<(const basis_elem&, const basis_elem&);
        friend bool operator==(const basis_elem&, const basis_elem&);
        friend bool operator!=(const basis_elem&, const basis_elem&);
        template <typename T> friend wavefunction<T> operator*(const opr<T>&, const mbasis_elem&);
        friend class mbasis_elem;
    public:
        // default constructor
        basis_elem() = default;
        
        // constructor (initializer for all site to be the 0th state)
        // with total number of sites, local dimension of Hilbert space,
        // and Nfermion is the fermion density operator (single site)
        basis_elem(const MKL_INT &n_sites, const MKL_INT &dim_local_);                              // boson
        basis_elem(const MKL_INT &n_sites, const MKL_INT &dim_local_, const opr<double> &Nfermion); // fermion
        
        // constructor from total number of sites and a given name.
        // Note: if you use a given name here, your operator definitions HAVE TO BE CONSISTENT with this basis definition!!!
        // current choices of name s:
        // ***   spin-1/2            ***
        // ***   spin-1              ***
        // ***   dimer               ***
        // ***   electron            ***
        // ***   tJ                  ***
        // ***   spinless-fermion    ***
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
        
        // a few basic properties
        MKL_INT local_dimension() const { return static_cast<MKL_INT>(dim_local); }
        MKL_INT total_sites() const;
        bool q_fermion() const { return (! Nfermion_map.empty()); }
        bool q_zero() const { return bits.none(); }
        bool q_maximized() const;
        bool q_same_state_all_site() const;
        
        // read out the status of a particular site
        // storage order: site0, site1, site2, ..., site(N-1)
        MKL_INT siteRead(const MKL_INT &site) const;
        
        basis_elem& siteWrite(const MKL_INT &site, const MKL_INT &val);
        
        // return a vector of length dim_local, reporting # of each state
        std::vector<MKL_INT> statistics() const;
        
        // change basis_elem to the next available state
        basis_elem& increment();
        
        // translate the basis directly using bits (suitable for 1D)
        // not implemented yet
        
        // transform the basis according to the given plan, or directly according to translation (or other transformation) on a particular lattice
        // sgn = 0 or 1 denoting if extra minus sign generated by translating fermions
        basis_elem& transform(const std::vector<MKL_INT> &plan, MKL_INT &sgn);
        basis_elem& translate(const lattice &latt, const std::vector<MKL_INT> &disp, MKL_INT &sgn);
        
        // reset bits to 0
        basis_elem& reset() {bits.reset(); return *this; }
        
        void prt() const;
        void prt_nonzero() const;
        
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
        friend bool trans_equiv(const mbasis_elem&, const mbasis_elem&, const lattice&);
        //template <typename T> friend class opr;
        //template <typename T> friend T operator* (const opr<T>&, const mbasis_elem&);
        //template <typename T> friend T operator* (const opr_prod<T>&, const mbasis_elem&);
        template <typename T> friend wavefunction<T> operator*(const opr<T>&, const mbasis_elem&);
    public:
        // default constructor
        mbasis_elem() = default;
        
        // construcutor with total number of sites, and name of each orbital
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
        
        // a few basic properties
        MKL_INT total_sites() const;
        MKL_INT total_orbitals() const;
        MKL_INT local_dimension() const;
        bool q_zero() const;
        bool q_maximized() const;
        
        // a direct product of the statistics of all orbitals, size: dim_orb1 * dim_orb2 * ...
        std::vector<MKL_INT> statistics() const;
        
        // change mbasis_elem to the next available state
        mbasis_elem& increment();
        
        
        
        // translate the basis according to the given plan, all orbitals transform in same way: site1 -> site2
        mbasis_elem& transform(const std::vector<MKL_INT> &plan, MKL_INT &sgn);
        // different orbs transform in different ways: (site1, orb1) -> (site2, orb2)
        // outer vector: each element denotes one orbital
        // middle vector: each element denotes one site
        // inner pair: first=site, second=orbital
        mbasis_elem& transform(const std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> &plan, MKL_INT &sgn);
        mbasis_elem& translate(const lattice &latt, const std::vector<MKL_INT> &disp, MKL_INT &sgn);
        
        // change to a basis element which is the unique (fully determined by the lattice and its state) among its translational equivalents
        mbasis_elem& translate_to_unique_state(const lattice &latt, std::vector<MKL_INT> &disp_vec);
        
        // reset all bits to 0 in all orbitals
        mbasis_elem& reset();
        
        MKL_INT siteRead(const MKL_INT &site, const MKL_INT &orbital) const;
        
        mbasis_elem& siteWrite(const MKL_INT &site, const MKL_INT &orbital, const MKL_INT &val);
        
        double diagonal_operator(const opr<double>& lhs) const;
        std::complex<double> diagonal_operator(const opr<std::complex<double>>& lhs) const;
        
        double diagonal_operator(const opr_prod<double>& lhs) const;
        std::complex<double> diagonal_operator(const opr_prod<std::complex<double>>& lhs) const;
        
        double diagonal_operator(const mopr<double>& lhs) const;
        std::complex<double> diagonal_operator(const mopr<std::complex<double>>& lhs) const;
        
        void prt() const;
        void prt_nonzero() const;
        
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
        wavefunction(mbasis_elem &&old)      : elements(1, std::pair<mbasis_elem, T>(old, static_cast<T>(1.0))) {}
        
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
        
        std::pair<mbasis_elem, T>& operator[](MKL_INT n)
        {
            assert(n < size());
            assert(! elements.empty());
            auto it = elements.begin();
            for (decltype(n) i = 0; i < n; i++) ++it;
            return *it;
        }
        
        const std::pair<mbasis_elem, T>& operator[](MKL_INT n) const
        {
            assert(n < size());
            assert(! elements.empty());
            auto it = elements.begin();
            for (decltype(n) i = 0; i < n; i++) ++it;
            return *it;
        }
        
        // add one element
        wavefunction& operator+=(std::pair<mbasis_elem, T> ele);
        wavefunction& operator+=(const mbasis_elem &ele);
        
        // add a wave function
        wavefunction& operator+=(wavefunction<T> rhs);
        
        // multiply by a constant
        wavefunction& operator*=(const T &rhs);
        
        // simplify
        wavefunction& simplify();
        
        // check if sorted
        bool sorted() const;
        // check if sorted and there are no dulplicated terms
        bool sorted_fully() const;
        
        // for \sum_i \alpha_i * element[i], return \sum_i |\alpha_i|^2
        double amplitude();
        
        // check if zero
        bool q_zero() const { return elements.empty(); }
        
        MKL_INT size() const {return static_cast<MKL_INT>(elements.size()); }
        
        void prt() const;
        void prt_nonzero() const;
        
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
        opr() : mat(nullptr) {}
        
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
        
        // take Hermitian conjugate
        opr& dagger();
        
        // fermions not implemented yet
        opr& transform(const std::vector<MKL_INT> &plan);
        
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
        
        opr_prod& transform(const std::vector<MKL_INT> &plan);
        
        // question if it is proportional to identity operator
        bool q_prop_identity() const;
        
        // question if each opr is diagonal
        bool q_diagonal() const;
        
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
        friend mopr<T> operator* <> (const mopr<T>&, const T&);
        friend mopr<T> operator* <> (const T&, const mopr<T>&);
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
            assert(n < size());
            assert(! mats.empty());
            auto it = mats.begin();
            for (decltype(n) i = 0; i < n; i++) ++it;
            return *it;
        }
        
        const opr_prod<T>& operator[](MKL_INT n) const
        {
            assert(n < size());
            assert(! mats.empty());
            auto it = mats.begin();
            for (decltype(n) i = 0; i < n; i++) ++it;
            return *it;
        }
        
        MKL_INT size() const {return static_cast<MKL_INT>(mats.size()); }
        
        // simplify, need implementation
        mopr& simplify();
        
        // invert the sign
        mopr& negative();
        
        mopr& transform(const std::vector<MKL_INT> &plan);
        
        // destructor
        ~mopr() {}
        
        bool q_zero() const {return mats.empty(); }
        
        // question if each opr_prod is diagonal
        bool q_diagonal() const;
        
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
        //lil_mat() : mtx(nullptr) {}
        lil_mat() = default;
        
        // constructor with the Hilbert space dimension
        lil_mat(const MKL_INT &n, bool sym_ = false);
        
        // add one element
        void add(const MKL_INT &row, const MKL_INT &col, const T &val);
        
        
        // explicitly destroy, free space
        void destroy()
        {
            mat.clear();
            mat.shrink_to_fit();
//            if (mtx != nullptr) {
//                delete [] mtx;
//                mtx = nullptr;
//            }
        }
        
        // destructor
        ~lil_mat()
        {
//            if (mtx != nullptr) {
//                delete [] mtx;
//                mtx = nullptr;
//            }
        }
        
        MKL_INT dimension() const { return dim; }
        
        MKL_INT num_nonzero() const { return nnz; }
        
        bool q_sym() { return sym; }
        
        // print
        void prt() const;
        
        void use_full_matrix() {sym = false; }
        
    private:
        MKL_INT dim;    // dimension of the matrix
        MKL_INT nnz;    // number of non-zero entries
        bool sym;       // if storing only upper triangle
        //std::mutex *mtx; // mutex for adding elements in parallel
        std::vector<std::forward_list<lil_mat_elem<T>>> mat;
    };
    
    
    // 3-array form of csr sparse matrix format, zero based
    template <typename T> class csr_mat {
        friend void swap <> (csr_mat<T>&, csr_mat<T>&);
    public:
        // default constructor
        csr_mat() : val(nullptr), ja(nullptr), ia(nullptr) {}
        
        // copy constructor
        csr_mat(const csr_mat<T> &old);
        
        // move constructor
        csr_mat(csr_mat<T> &&old) noexcept;
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        csr_mat& operator=(csr_mat<T> old)
        {
            swap(*this, old);
            return *this;
        }
        
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
    
    
//  ----------------------------- part 5: Lattices  ----------------------------
//  ----------------------------------------------------------------------------
    class lattice {
    public:
        lattice() = default;
        
        // constructor from particular requirements. e.g. square, triangular...
        lattice(const std::string &name, const std::string &bc_, std::initializer_list<MKL_INT> lens);
        
        // coordinates <-> site indices
        // 1D: site = i * num_sub + sub
        // 2D: site = (i + j * L[0]) * num_sub + sub
        // 3D: site = (i + j * L[0] + k * L[0] * L[1]) * num_sub + sub
        void coor2site(const std::vector<MKL_INT> &coor, const MKL_INT &sub, MKL_INT &site) const;
        void site2coor(std::vector<MKL_INT> &coor, MKL_INT &sub, const MKL_INT &site) const;
        
        // return a vector containing the positions of each site after translation
        std::vector<MKL_INT> translation_plan(const std::vector<MKL_INT> &disp) const;
        
        // return a vector containing the positions of each site after c2 (180) or c4 (90) rotation
        std::vector<MKL_INT> c2_rotation_plan() const;
        std::vector<MKL_INT> c4_rotation_plan() const;
        std::vector<MKL_INT> reflection_plan() const;
        
        // combine two plans
        std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> plan_product(const std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> &lhs,
                                                                          const std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> &rhs) const;
        
        // inverse of a transformation
        std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> plan_inverse(const std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> &old) const;
        
        std::string boundary() const {
            return bc;
        }
        
        MKL_INT dimension() const {
            return dim;
        }
        
        MKL_INT total_sites() const {
            return Nsites;
        }
        
        std::vector<MKL_INT> Linear_size() const { return L; }
        
        MKL_INT Lx() const { return L[0]; }
        MKL_INT Ly() const { assert(L.size() > 1); return L[1]; }
        MKL_INT Lz() const { assert(L.size() > 2); return L[2]; }
        
    private:
        MKL_INT dim;
        MKL_INT num_sub;
        std::vector<std::vector<double>> a; // real space basis
        std::vector<std::vector<double>> b; // momentum space basis
        std::string bc;                     // boundary condition
        std::vector<MKL_INT> L;             // linear size in each dimension
        MKL_INT Nsites;
        
    };
    
    
//  ---------------part 6: Routines to construct Hamiltonian -------------------
//  ----------------------------------------------------------------------------
    
    template <typename T> class model {
//        friend MKL_INT generate_Ham_all_AtRow <> (model<T> &,
//                                                  threads_pool &);
    public:
        model() = default;
        
        ~model() {}
        
        void add_diagonal_Ham(const opr<T> &rhs)      { assert(rhs.q_diagonal()); Ham_diag += rhs; }
        void add_diagonal_Ham(const opr_prod<T> &rhs) { assert(rhs.q_diagonal()); Ham_diag += rhs; }
        void add_diagonal_Ham(const mopr<T> &rhs)     { assert(rhs.q_diagonal()); Ham_diag += rhs; }
        
        void add_offdiagonal_Ham(const opr<T> &rhs)      { Ham_off_diag += rhs; }
        void add_offdiagonal_Ham(const opr_prod<T> &rhs) { Ham_off_diag += rhs; }
        void add_offdiagonal_Ham(const mopr<T> &rhs)     { Ham_off_diag += rhs; }
        
        void enumerate_basis_all(); // naive way of enumerating all possible basis state
        void enumerate_basis_conserve(const MKL_INT &n_sites, std::initializer_list<std::string> lst,
                                      const mopr<std::complex<double>> &conserve, const double &val);
        
        void sort_basis_all();
        
        void generate_Ham_all_sparse(const bool &upper_triangle = true); // generate the full Hamiltonian in sparse matrix format
        
        void locate_E0(const MKL_INT &nev = 5, const MKL_INT &ncv = 15);
        
        void locate_Emax(const MKL_INT &nev = 5, const MKL_INT &ncv = 15);
        
        // lhs | phi >
        void moprXeigenvec(const mopr<T> &lhs, T* vec_new, const MKL_INT &which_col = 0);
        // < phi | lhs | phi >
        T measure(const mopr<T> &lhs, const MKL_INT &which_col=0);
        // < phi | lhs1^\dagger lhs2 | phi >
        T measure(const mopr<T> &lhs1, const mopr<T> &lhs2, const MKL_INT &which_col=0);
        
        double energy_min() { return E0; }
        double energy_max() { return Emax; }
        double energy_gap() { return gap; }
        
        MKL_INT dim_all;
        MKL_INT dim_repr;
        
        mopr<T> Ham_diag;
        mopr<T> Ham_off_diag;
        
        // add basis_repr later
        std::vector<qbasis::mbasis_elem> basis_all;
        
        std::vector<T> basis_coeff;
        
        csr_mat<T> HamMat_csr;
        
        std::vector<double> eigenvals;
        std::vector<T> eigenvecs;
        MKL_INT nconv;
        
        void prt_Ham_diag() { Ham_diag.prt(); }
        void prt_Ham_offdiag() { Ham_off_diag.prt(); }
        
        
        // later add conserved quantum operators and corresponding quantum numbers
        // later add measurement operators
    
    private:
        
        
        double Emax;
        double E0;
        double gap;
        
        // lil_mat<T> HamMat_lil;   // only for internal temporaty use
        
        
    };
    
    
    
//  ---------------------------part 7: Measurements ----------------------------
//  ----------------------------------------------------------------------------
    
    
    

    
//  --------------------------- Miscellaneous stuff ----------------------------
//  ----------------------------------------------------------------------------
    
    inline double conjugate(const double &rhs) { return rhs; }
    inline std::complex<double> conjugate(const std::complex<double> &rhs) { return std::conj(rhs); }
    
    // calculate base^index, in the case both are integers
    MKL_INT int_pow(const MKL_INT &base, const MKL_INT &index);
    
    // given two arrays: num & base, get the result of:
    // num[0] + num[1] * base[0] + num[2] * base[0] * base[1] + num[3] * base[0] * base[1] * base[2] + ...
    MKL_INT dynamic_base(const std::vector<MKL_INT> &num, const std::vector<MKL_INT> &base);
    // the other way around
    std::vector<MKL_INT> dynamic_base(const MKL_INT &total, const std::vector<MKL_INT> &base);
    // nums + 1
    std::vector<MKL_INT> dynamic_base_plus1(const std::vector<MKL_INT> &nums, const std::vector<MKL_INT> &base);
    
    template <typename T>
    bool is_sorted_norepeat(const std::vector<T> &array);
    
    // note: end means the position which has already passed the last element
    template <typename T>
    MKL_INT binary_search(const std::vector<T> &array, const T &val,
                          const MKL_INT &bgn, const MKL_INT &end);
    
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
    // comment 1: zdotc is a problematic function in lapack, when returning std::complex
    // comment 2: with my own version of dotc, it will slow things down without parallelization, need fix later
    inline // complex double
    std::complex<double> dotc(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx,
                              const std::complex<double> *y, const MKL_INT incy) {
        std::complex<double> result(0.0, 0.0);
        cblas_zdotc_sub(n, x, incx, y, incy, &result);
//        const std::complex<double> *xpt = x;
//        const std::complex<double> *ypt = y;
//        for (MKL_INT j = 0; j < n; j++) {
//            result += std::conj(*xpt) * (*ypt);
//            xpt += incx;
//            ypt += incy;
//        }
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
