#ifndef BASIS_H
#define BASIS_H
#include <string>
#include <vector>
#include <complex>
#include <utility>
#include <initializer_list>
#include <boost/dynamic_bitset.hpp>
#include "operators.h"

namespace qbasis {
    // the startup cost of boost::dynamic_bitset is around 40 Bytes
    using DBitSet = boost::dynamic_bitset<>;
    
    // forward declarations
    class basis_elem;
    class mbasis_elem;
    template <typename> class wavefunction;
    template <typename> class opr;
    template <typename> class opr_prod;
    template <typename> class mopr;
    
    bool operator<(const basis_elem&, const basis_elem&);
    bool operator==(const basis_elem&, const basis_elem&);

    bool operator<(const mbasis_elem&, const mbasis_elem&);
    bool operator==(const mbasis_elem&, const mbasis_elem&);
    
    template <typename T> void swap(wavefunction<T>&, wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const wavefunction<T>&, const T&);
    template <typename T> wavefunction<T> operator*(const T&, const wavefunction<T>&);
    
    // opr * | orb0, orb1, ..., ORB, ... > = | orb0, orb1, ..., opr*ORB, ... >, fermionic sign has to be computed when traversing orbitals
    template <typename T> wavefunction<T> operator*(const opr<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const opr<T>&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const opr_prod<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const opr_prod<T>&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const mopr<T>&, const mbasis_elem&);
    template <typename T> wavefunction<T> operator*(const mopr<T>&, const wavefunction<T>&);
    
    
    // ---------------- fundamental class for basis elements ------------------
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
    
}


#endif
