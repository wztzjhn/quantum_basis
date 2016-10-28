#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include "basis.h"


// ----------------- implementation of basis ------------------
basis::basis(const int &n_sites, const int &dim_local_):
             dim_local(dim_local_),
             bits_per_site(static_cast<int>(ceil(log2(static_cast<double>(dim_local_)) - 1e-9))),
             bits(static_cast<DBitSet::size_type>(n_sites * bits_per_site)) {};

basis::basis(const int &n_sites, const std::string &s)
{
    if (s == "spin-1/2") {
        dim_local = 2;
        
    } else if (s == "spin-1") {
        dim_local = 3;
    }
    bits_per_site = static_cast<int>(ceil(log2(static_cast<double>(dim_local)) - 1e-9));
    bits = DBitSet(static_cast<DBitSet::size_type>(n_sites * bits_per_site));
}

void basis::test() const
{
    std::cout << "total sites(" << basis::total_sites()
              << ") * bits per site(" << bits_per_site
              << ") = size of bits(" << bits.size() << ")" << std::endl;
    std::cout << "number of blocks: " << bits.num_blocks() << std::endl;
    std::cout << "capacity: " << bits.capacity() << std::endl;
    std::cout << "number of bits on: " << bits.count() << std::endl;
    std::cout << "sizeof int: " << sizeof(dim_local) << std::endl;
    std::cout << "sizeof double: " << sizeof(0.3) << std::endl;
    std::cout << bits.empty() << std::endl;
    std::cout << bits << std::endl;
}



// ----------------- implementation of mbasis ------------------
mbasis::mbasis(const int &n_sites, std::initializer_list<int> lst)
{
    for (const auto &elem : lst) mbits.push_back(basis(n_sites, elem));
    mbits.shrink_to_fit();
}

mbasis::mbasis(const int &n_sites, std::initializer_list<std::string> lst)
{
    for (const auto &elem : lst) mbits.push_back(basis(n_sites, elem));
    mbits.shrink_to_fit();
}

int mbasis::total_sites() const
{
    assert(! mbits.empty());
    return mbits[0].total_sites();
}



void mbasis::test() const
{
    std::cout << "total sites: " << mbasis::total_sites() << std::endl;
    
}
