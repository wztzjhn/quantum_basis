#ifndef BASIS_H
#define BASIS_H
#include <boost/dynamic_bitset.hpp>

class basis {
public:
    // constructors
    basis() = default;
    basis(const int &n_sites_, const int &dim_local_): 
        n_sites(n_sites_), dim_local(dim_local_), bits(static_cast<boost::dynamic_bitset<>::size_type>(n_sites_*dim_local_)) {};
    
    int total_sites() const { return n_sites; }
    int local_dimension() const { return dim_local; }
    void test() const;

private:
    int n_sites;
    int dim_local;
    boost::dynamic_bitset<> bits;


};


#endif
