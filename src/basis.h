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
    bool operator<(const basis_elem&, const basis_elem&);
    bool operator==(const basis_elem&, const basis_elem&);
    
    class mbasis_elem;
    bool operator<(const mbasis_elem&, const mbasis_elem&);
    bool operator==(const mbasis_elem&, const mbasis_elem&);
    
    template <typename> class wavefunction;
    template <typename T> void swap(wavefunction<T>&, wavefunction<T>&);
    
    template <typename> class opr;
    
    // ---------------- fundamental class for basis elements ------------------
    // for given number of sites, store the bits for a single orbital
    class basis_elem {
        friend void swap(basis_elem &lhs, basis_elem &rhs);
        friend bool operator<(const basis_elem&, const basis_elem&);
        friend bool operator==(const basis_elem&, const basis_elem&);
        template <typename X> friend class opr;                                   // all instances of any type is a friend of basis_elem
    public:
        // default constructor
        basis_elem() = default;
        
        // constructor (initializer for all site to be the 0th state)
        // with total number of sites, local dimension of Hilbert space,
        // and Nfermion is the fermion density operator (single site)
        basis_elem(const int &n_sites, const int &dim_local_);                              // boson
        basis_elem(const int &n_sites, const int &dim_local_, const opr<double> &Nfermion); // fermion
        
        // constructor from total number of sites and a given name.
        // current choices:
        // "spin-1/2", "spin-1"
        basis_elem(const int &n_sites, const std::string &s);
        
        // copy constructor
        basis_elem(const basis_elem& old);
        
        // move constructor
        basis_elem(basis_elem &&old) noexcept :
            dim_local(old.dim_local), bits_per_site(old.bits_per_site),
            odd_fermion(old.odd_fermion), bits(std::move(old.bits))
        {
            old.odd_fermion = nullptr;
        }
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        basis_elem& operator=(basis_elem old)
        {
            swap(*this, old);
            return *this;
        }
        
        // destructor
        ~basis_elem() {
            if(odd_fermion != nullptr) delete [] odd_fermion;
        }
        
        int total_sites() const;
        
        bool q_fermion() const {return (odd_fermion != nullptr); }
        
        short local_dimension() const { return dim_local; }
        
        // read out the status of a particular site
        // storage order: site0, site1, site2, ..., site(N-1)
        short siteRead(const int &site) const;
        
        void siteWrite(const int &site, const short &val);                               // boson
        void siteWrite(const int &site, const short &val, const opr<double> &Nfermion);  // fermion
        
        void prt() const;
        
        // remember to change odd_fermion[site]
        //basis_elem& flip();
        
    private:
        short dim_local;
        short bits_per_site;
        bool *odd_fermion;    // an array specifying if there are odd # of fermions on each site; when == nullptr -> boson
        DBitSet bits;
    };
    
    
    
    
    // -------------- class for basis with several orbitals---------------
    // for given number of sites, and several orbitals, store the vectors of bits
    class mbasis_elem {
        friend void swap(mbasis_elem&, mbasis_elem&);
        friend bool operator<(const mbasis_elem&, const mbasis_elem&);
        friend bool operator==(const mbasis_elem&, const mbasis_elem&);
    public:
        // default constructor
        mbasis_elem() = default;
        
        // construcutor with total number of sites, and name  of each orbital
        mbasis_elem(const int &n_sites, std::initializer_list<std::string> lst);
        
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
        
        int total_sites() const;
        int total_orbitals() const;
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
    public:
        // default constructor
        wavefunction() = default;
        
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
        
        // check if sorted
        bool sorted() const;
        
    private:
        // store an array of basis elements, and their corresponding coefficients
        // note: there should not be any dulplicated elements
        std::vector<std::pair<mbasis_elem, T>> elements;
    };
    
}


#endif
