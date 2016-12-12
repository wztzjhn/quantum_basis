#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <utility>
#include "basis.h"

namespace qbasis {
    // ----------------- implementation of basis ------------------
    basis_elem::basis_elem(const int &n_sites, const int &dim_local_):
        dim_local(static_cast<short>(dim_local_)),
        bits_per_site(static_cast<short>(ceil(log2(static_cast<double>(dim_local_)) - 1e-9))),
        odd_fermion(nullptr),
        bits(static_cast<DBitSet::size_type>(n_sites * bits_per_site)) {}
    
    basis_elem::basis_elem(const int &n_sites, const int &dim_local_, const opr<double> &Nfermion):
        dim_local(static_cast<short>(dim_local_)),
        bits_per_site(static_cast<short>(ceil(log2(static_cast<double>(dim_local_)) - 1e-9))),
        bits(static_cast<DBitSet::size_type>(n_sites * bits_per_site))
    {
        assert(Nfermion.q_diagonal() && ! Nfermion.q_zero());
        int fermion_density = static_cast<int>(ceil(Nfermion.mat[0] - 1e-9) + 1e-9);
        bool oddness = static_cast<bool>(fermion_density % 2);
        odd_fermion = new bool[n_sites];
        for (int j = 0; j < n_sites; j++) odd_fermion[j] = oddness;
    }
    
    basis_elem::basis_elem(const int &n_sites, const std::string &s)
    {
        if (s == "spin-1/2") {
            dim_local = 2;
            odd_fermion = nullptr;
        } else if (s == "spin-1") {
            dim_local = 3;
            odd_fermion = nullptr;
        }
        bits_per_site = static_cast<short>(ceil(log2(static_cast<double>(dim_local)) - 1e-9));
        bits = DBitSet(static_cast<DBitSet::size_type>(n_sites * bits_per_site));
    }
    
    
    basis_elem::basis_elem(const basis_elem& old):
        dim_local(old.dim_local), bits_per_site(old.bits_per_site), bits(old.bits)
    {
        if (old.q_fermion()) {
            odd_fermion = new bool[total_sites()];
            for (int j = 0; j < total_sites(); j++) odd_fermion[j] = old.odd_fermion[j];
        } else {
            odd_fermion = nullptr;
        }
        
    }
    
    int basis_elem::total_sites() const
    {
        if (bits_per_site > 0) {
            return static_cast<int>(bits.size()) / bits_per_site;
        } else {
            return 0;
        }
    }
    
    short basis_elem::siteRead(const int &site) const
    {
        assert(site >= 0 && site < total_sites());
        int bits_bgn = bits_per_site * site;
        int bits_end = bits_bgn + bits_per_site;
        short res = static_cast<short>(bits[bits_bgn]);
        for (auto j = bits_bgn + 1; j < bits_end; j++) {
            res = res + res + static_cast<short>(bits[j]);
        }
        return res;
    }
    
    void basis_elem::siteWrite(const int &site, const short &val)
    {
        assert(val >= 0 && val < dim_local);
        assert(! q_fermion());
        int bits_bgn = bits_per_site * site;
        auto temp = val;
        for (int j = bits_bgn + bits_per_site - 1; j >= bits_bgn; j--) {
            bits[j] = temp % 2;
            temp /= 2;
        }
    }
    
    void basis_elem::siteWrite(const int &site, const short &val, const opr<double> &Nfermion)
    {
        assert(val >= 0 && val < dim_local);
        assert(q_fermion() && Nfermion.q_diagonal() && ! Nfermion.q_zero());
        int bits_bgn = bits_per_site * site;
        auto temp = val;
        for (int j = bits_bgn + bits_per_site - 1; j >= bits_bgn; j--) {
            bits[j] = temp % 2;
            temp /= 2;
        }
        int fermion_density = static_cast<int>(ceil(Nfermion.mat[val] - 1e-9) + 1e-9);
        odd_fermion[site]   = static_cast<bool>(fermion_density % 2);
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
    
//    basis_elem& basis_elem::flip()    // remember to change odd_fermion[site]
//    {
//        bits.flip();
//        return *this;
//    }
    
    
    void swap(basis_elem &lhs, basis_elem &rhs)
    {
        using std::swap;
        swap(lhs.dim_local, rhs.dim_local);
        swap(lhs.bits_per_site, rhs.bits_per_site);
        swap(lhs.odd_fermion, rhs.odd_fermion);
        lhs.bits.swap(rhs.bits);
    }
    
    bool operator<(const basis_elem &lhs, const basis_elem &rhs)
    {
        assert(lhs.dim_local == rhs.dim_local);
        assert(lhs.bits_per_site == rhs.bits_per_site);
        assert(lhs.q_fermion() == rhs.q_fermion());
        return (lhs.bits < rhs.bits);
    }
    
    bool operator==(const basis_elem &lhs, const basis_elem &rhs)
    {
        assert(lhs.dim_local == rhs.dim_local);
        assert(lhs.bits_per_site == rhs.bits_per_site);
        assert(lhs.q_fermion() == rhs.q_fermion());
        return (lhs.bits == rhs.bits);
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
    
    bool operator<(const mbasis_elem &lhs, const mbasis_elem &rhs)
    {
        return (lhs.mbits < rhs.mbits);
    }
    
    bool operator==(const mbasis_elem &lhs, const mbasis_elem &rhs)
    {
        return (lhs.mbits == rhs.mbits);
    }
    
    
    
    
    // ----------------- implementation of wavefunction ------------------
    template <typename T>
    bool wavefunction<T>::sorted() const
    {
        if (elements.size() == 0) return true;
        bool check = true;
        for (decltype(elements.size()) j = 0; j < elements.size() - 1; j++) {
            if (elements[j+1].first < elements[j].first) {
                check = false;
                break;
            }
        }
        return check;
    }
    
    template <typename T>
    void swap(wavefunction<T> &lhs, wavefunction<T> &rhs)
    {
        using std::swap;
        swap(lhs.elements, rhs.elements);
    }
    
    
    // Explicit instantiation
    template class wavefunction<double>;
    template class wavefunction<std::complex<double>>;
    
}
