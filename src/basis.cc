#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <utility>
#include "basis.h"


// ----------------- implementation of basis ------------------
basis_elem::basis_elem(const int &n_sites, const bool &fermion_, const int &dim_local_):
             dim_local(static_cast<short>(dim_local_)),
             bits_per_site(static_cast<short>(ceil(log2(static_cast<double>(dim_local_)) - 1e-9))),
             fermion(fermion_),
             bits(static_cast<DBitSet::size_type>(n_sites * bits_per_site)) {};

basis_elem::basis_elem(const int &n_sites, const std::string &s)
{
    if (s == "spin-1/2") {
        dim_local = 2;
        fermion = false;
        
    } else if (s == "spin-1") {
        dim_local = 3;
        fermion = false;
    }
    bits_per_site = static_cast<short>(ceil(log2(static_cast<double>(dim_local)) - 1e-9));
    bits = DBitSet(static_cast<DBitSet::size_type>(n_sites * bits_per_site));
}

basis_elem::basis_elem(const basis_elem &old):
             dim_local(old.dim_local),
             bits_per_site(old.bits_per_site),
             fermion(old.fermion),
             bits(old.bits) {};

basis_elem::basis_elem(basis_elem &&old) noexcept :
             dim_local(old.dim_local),
             bits_per_site(old.bits_per_site),
             fermion(old.fermion),
             bits(std::move(old.bits)) {};

basis_elem& basis_elem::operator=(basis_elem old)
{
    swap(*this, old);
    return *this;
}

int basis_elem::total_sites() const
{
    if (bits_per_site > 0) {
        return static_cast<int>(bits.size()) / bits_per_site;
    } else {
        return 0;
    }
}

void basis_elem::prt() const
{
    std::cout << basis_elem::total_sites() << " sites * "
              << bits_per_site << " bits/site = "
              << bits.size() << " bits." << std::endl;
    std::cout << "local Hibert space: " << basis_elem::dim_local << std::endl;
    std::cout << "number of bits on: " << bits.count() << std::endl;
    std::cout << bits << std::endl;
}



// ----------------- implementation of mbasis ------------------
mbasis_elem::mbasis_elem(const int &n_sites, std::initializer_list<std::string> lst)
{
    for (const auto &elem : lst) mbits.push_back(basis_elem(n_sites, elem));
    mbits.shrink_to_fit();
}

int mbasis_elem::total_sites() const
{
    assert(! mbits.empty());
    return mbits[0].total_sites();
}



void mbasis_elem::test() const
{
    std::cout << "total sites: " << mbasis_elem::total_sites() << std::endl;
    
}
