#ifndef BASIS_H
#define BASIS_H
#include <string>
#include <vector>
#include <initializer_list>
#include <boost/dynamic_bitset.hpp>

using DBitSet = boost::dynamic_bitset<>;

// ---------------- fundamental class for basis ------------------
// for given number of sites, store the bits for a single orbital
class basis {
public:
    //------------- constructors ------------
    // default constructor
    basis() = default;
    
    // constructor with total number of sites, and local dimension of Hilbert space
    basis(const int &n_sites, const int &dim_local_);
    
    // constructor from total number of sites and a given name.
    // current choices:
    // "spin-1/2", "spin-1"
    basis(const int &n_sites, const std::string &s);
    
    int total_sites() const { return static_cast<int>(bits.size()) / bits_per_site; };
    int local_dimension() const { return dim_local; }
    
    void test() const;

private:
    int dim_local;
    int bits_per_site;
    DBitSet bits;
    
    

};

// -------------- class for basis with several orbitals---------------
// for given number of sites, and several orbitals, store the vectors of bits
class mbasis {
public:
    // default constructor
    mbasis() = default;
    // construcutor with total number of sites, and local dimension (or name) of each orbital
    mbasis(const int &n_sites, std::initializer_list<int> lst);
    mbasis(const int &n_sites, std::initializer_list<std::string> lst);
    
    int total_sites() const;
    
    void test() const;
    
    
private:
    // store an array of basis, for multi-orbital site (or unit cell)
    std::vector<basis> mbits;
    
    
};


#endif
