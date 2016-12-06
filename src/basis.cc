#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <utility>
#include "basis.h"

namespace qbasis {
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
    
    basis_elem& basis_elem::flip()
    {
        bits.flip();
        return *this;
    }
    
    
    void swap(basis_elem &lhs, basis_elem &rhs)
    {
        using std::swap;
        swap(lhs.dim_local, rhs.dim_local);
        swap(lhs.bits_per_site, rhs.bits_per_site);
        swap(lhs.fermion, rhs.fermion);
        lhs.bits.swap(rhs.bits);
    }
    
    
    // ----------------- implementation of mbasis ------------------
    mbasis_elem::mbasis_elem(const int &n_sites, std::initializer_list<std::string> lst)
    {
        for (const auto &elem : lst) mbits.push_back(basis_elem(n_sites, elem));
        //std::cout << "size before shrink: " << mbits.capacity() << std::endl;
        mbits.shrink_to_fit();
        //std::cout << "size after shrink: " << mbits.capacity() << std::endl;
    }
    
    int mbasis_elem::total_sites() const
    {
        assert(! mbits.empty());
        return mbits[0].total_sites();
    }
    
    int mbasis_elem::total_orbitals() const
    {
        assert(! mbits.empty());
        return static_cast<int>(mbits.size());
    }
    
    
    
    void mbasis_elem::test() const
    {
        std::cout << "total sites: " << mbasis_elem::total_sites() << std::endl;
        
    }
    
    void swap(mbasis_elem &lhs, mbasis_elem &rhs)
    {
        using std::swap;
        swap(lhs.mbits, rhs.mbits);
    }
    
    
    
    void swap(wavefunction &lhs, wavefunction &rhs)
    {
        using std::swap;
        swap(lhs.elements, rhs.elements);
        swap(lhs.coeffs, rhs.coeffs);
    }
    
}
