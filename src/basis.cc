#include <cmath>
#include <iostream>
#include "qbasis.h"

namespace qbasis {
    // ----------------- implementation of basis ------------------
    basis_elem::basis_elem(const MKL_INT &n_sites, const MKL_INT &dim_local_):
        dim_local(static_cast<short>(dim_local_)),
        bits_per_site(static_cast<short>(ceil(log2(static_cast<double>(dim_local_)) - 1e-9))),
        Nfermion_map(std::vector<int>()),
        bits(static_cast<DBitSet::size_type>(n_sites * bits_per_site)) {}
    
    basis_elem::basis_elem(const MKL_INT &n_sites, const MKL_INT &dim_local_, const opr<double> &Nfermion):
        dim_local(static_cast<short>(dim_local_)),
        bits_per_site(static_cast<short>(ceil(log2(static_cast<double>(dim_local_)) - 1e-9))),
        bits(static_cast<DBitSet::size_type>(n_sites * bits_per_site))
    {
        assert(Nfermion.q_diagonal() && ! Nfermion.q_zero());
        Nfermion_map = std::vector<int>(dim_local_);
        Nfermion_map.shrink_to_fit();
        for (decltype(Nfermion_map.size()) j = 0; j < Nfermion_map.size(); j++)
            Nfermion_map[j] = static_cast<int>(ceil(Nfermion.mat[j] - 1e-9) + 1e-9);
    }
    
    basis_elem::basis_elem(const MKL_INT &n_sites, const std::string &s)
    {
        if (s == "spin-1/2") {
            dim_local = 2;
            Nfermion_map = std::vector<int>();
        } else if (s == "spin-1") {
            dim_local = 3;
            Nfermion_map = std::vector<int>();
        }
        bits_per_site = static_cast<short>(ceil(log2(static_cast<double>(dim_local)) - 1e-9));
        bits = DBitSet(static_cast<DBitSet::size_type>(n_sites * bits_per_site));
    }
    
    MKL_INT basis_elem::total_sites() const
    {
        if (bits_per_site > 0) {
            return static_cast<MKL_INT>(bits.size()) / bits_per_site;
        } else {
            return 0;
        }
    }
    
    MKL_INT basis_elem::siteRead(const MKL_INT &site) const
    {
        assert(site >= 0 && site < total_sites());
        MKL_INT bits_bgn = bits_per_site * site;
        MKL_INT bits_end = bits_bgn + bits_per_site;
        MKL_INT res = bits[bits_end - 1];
        for (auto j = bits_end - 2; j >= bits_bgn; j--) res = res + res + bits[j];
        return res;
    }
    
    basis_elem &basis_elem::siteWrite(const MKL_INT &site, const MKL_INT &val)
    {
        assert(val >= 0 && val < local_dimension());
        MKL_INT bits_bgn = bits_per_site * site;
        MKL_INT bits_end = bits_bgn + bits_per_site;
        auto temp = val;
        for (MKL_INT j = bits_bgn; j < bits_end; j++) {
            bits[j] = temp % 2;
            temp /= 2;
        }
        return *this;
    }
    
    MKL_INT int_pow(const MKL_INT &base, const MKL_INT &index)
    {
        assert(index >= 0);
        MKL_INT res = 1;
        for (MKL_INT j = 0; j < index; j++) res *= base;
        return res;
    }
    
    bool basis_elem::q_maximized() const
    {
        if (int_pow(2, bits_per_site) == dim_local) {
            return bits.all();
        } else {
            for (MKL_INT site = 0; site < total_sites(); site++) {
                if (siteRead(site) != dim_local) return false;
            }
            return true;
        }
    }
    
    basis_elem &basis_elem::increment()
    {
        assert(! q_maximized());
        if (int_pow(2, bits_per_site) == dim_local) {     // no waste bit
            for(decltype(bits.size()) loop = 0; loop < bits.size(); ++loop)
            {
                if ((bits[loop] ^= 0x1) == 0x1) break;
            }
        } else {
            auto val = siteRead(0) + 1;
            MKL_INT site = 0;
            while (val >= dim_local && site < total_sites()) {
                siteWrite(site, val - dim_local);
                val = siteRead(++site) + 1;
            }
            assert(site < total_sites());
            siteWrite(site, val);
        }
        return *this;
    }
    
    void basis_elem::prt() const
    {
        std::cout << bits;
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
        swap(lhs.Nfermion_map, rhs.Nfermion_map);
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
    
    bool operator!=(const basis_elem &lhs, const basis_elem &rhs)
    {
        return (! (lhs == rhs));
    }
    
    
    // ----------------- implementation of mbasis ------------------
    mbasis_elem::mbasis_elem(const MKL_INT &n_sites, std::initializer_list<std::string> lst)
    {
        for (const auto &elem : lst) mbits.push_back(basis_elem(n_sites, elem));
        //std::cout << "size before shrink: " << mbits.capacity() << std::endl;
        mbits.shrink_to_fit();
        //std::cout << "size after shrink: " << mbits.capacity() << std::endl;
    }
    
    double mbasis_elem::diagonal_operator(const opr<double>& lhs) const
    {
        assert(lhs.q_diagonal() && (! lhs.fermion) && lhs.dim == mbits[lhs.orbital].local_dimension());
        if (lhs.q_zero()) {
            return 0.0;
        } else {
            return lhs.mat[mbits[lhs.orbital].siteRead(lhs.site)];
        }
    }
    
    std::complex<double> mbasis_elem::diagonal_operator(const opr<std::complex<double>>& lhs) const
    {
        assert(lhs.q_diagonal() && (! lhs.fermion) && lhs.dim == mbits[lhs.orbital].local_dimension());
        if (lhs.q_zero()) {
            return std::complex<double>(0.0, 0.0);
        } else {
            return lhs.mat[mbits[lhs.orbital].siteRead(lhs.site)];
        }
    }
    
    double mbasis_elem::diagonal_operator(const opr_prod<double>& lhs) const
    {
        if (lhs.q_zero()) {
            return 0.0;
        } else {
            double res = lhs.coeff;
            for (const auto &op : lhs.mat_prod) {
                assert(op.q_diagonal() && (! op.fermion) && op.dim == mbits[op.orbital].local_dimension());
                res *= diagonal_operator(op);
                if (std::abs(res) < opr_precision) break;
            }
            return res;
        }
    }
    
    std::complex<double> mbasis_elem::diagonal_operator(const opr_prod<std::complex<double>>& lhs) const
    {
        if (lhs.q_zero()) {
            return std::complex<double>(0.0, 0.0);
        } else {
            std::complex<double> res = lhs.coeff;
            for (const auto &op : lhs.mat_prod) {
                assert(op.q_diagonal() && (! op.fermion) && op.dim == mbits[op.orbital].local_dimension());
                res *= diagonal_operator(op);
                if (std::abs(res) < opr_precision) break;
            }
            return res;
        }
    }
    
    bool mbasis_elem::q_maximized() const
    {
        for (MKL_INT orb = 0; orb < total_orbitals(); orb++) {
            if (! mbits[orb].q_maximized()) return false;
        }
        return true;
    }
    
    mbasis_elem &mbasis_elem::increment()
    {
        assert(! q_maximized());
        for (MKL_INT orb = total_orbitals() - 1; orb >= 0; orb--) {
            if (mbits[orb].q_maximized()) {
                mbits[orb].reset();
            } else {
                mbits[orb].increment();
                break;
            }
        }
        return *this;
    }
    
    mbasis_elem &mbasis_elem::reset()
    {
        for(auto it = mbits.begin(); it != mbits.end(); it++) it->reset();
        return *this;
    }
    
    MKL_INT mbasis_elem::total_sites() const
    {
        assert(! mbits.empty());
        return mbits[0].total_sites();
    }
    
    MKL_INT mbasis_elem::total_orbitals() const
    {
        assert(! mbits.empty());
        return static_cast<MKL_INT>(mbits.size());
    }
    
    
    
    void mbasis_elem::prt() const
    {
        if (total_orbitals() == 1) {
            mbits[0].prt();
        } else {
            for (MKL_INT j = 0; j < total_orbitals(); j++) {
                mbits[j].prt();
                std::cout << ", ";
            }
        }
        
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
    
    bool operator!=(const mbasis_elem &lhs, const mbasis_elem &rhs)
    {
        return (! (lhs == rhs));
    }
    
    
    
    
    // ----------------- implementation of wavefunction ------------------
    template <typename T>
    wavefunction<T> &wavefunction<T>::operator+=(std::pair<mbasis_elem, T> ele)
    {
        if (std::abs(ele.second) < opr_precision) return *this;
        if (elements.empty()) {
            elements.push_back(std::move(ele));
        } else {
            auto it = elements.begin();
            while (it != elements.end() && it->first < ele.first) it++;
            if (it == elements.end()) {
                elements.insert(it, std::move(ele));
            } else if(ele.first == it->first) {
                it->second += ele.second;
                if (std::abs(it->second) < opr_precision) elements.erase(it);
            } else {
                elements.insert(it, std::move(ele));
            }
        }
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::operator+=(wavefunction<T> rhs)
    {
        if (rhs.elements.empty()) return *this;  // adding zero
        if (elements.empty()) {                  // itself zero
            swap(*this, rhs);
            return *this;
        }
        for (auto &ele : rhs.elements) *this += ele;
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::operator*=(const T &rhs)
    {
        if (elements.empty()) return *this;      // itself zero
        if (std::abs(rhs) < opr_precision) {     // multiply by zero
            elements.clear();
            return *this;
        }
        for (auto it = elements.begin(); it != elements.end(); it++) it->second *= rhs;
        return *this;
    }
    
    template <typename T>
    bool wavefunction<T>::sorted() const
    {
        if (elements.size() == 0) return true;
        bool check = true;
        auto it = elements.begin();
        auto it_prev = it++;
        while (it != elements.end()) {
            if (it->first < it_prev->first) {
                check = false;
                break;
            }
            it_prev = it++;
        }
        return check;
    }
    
    template <typename T>
    void swap(wavefunction<T> &lhs, wavefunction<T> &rhs)
    {
        using std::swap;
        swap(lhs.elements, rhs.elements);
    }
    
    template <typename T>
    wavefunction<T> operator*(const wavefunction<T> &lhs, const T &rhs)
    {
        wavefunction<T> prod = lhs;
        prod *= rhs;
        return prod;
    }
    
    template <typename T>
    wavefunction<T> operator*(const T &lhs, const wavefunction<T> &rhs)
    {
        wavefunction<T> prod = rhs;
        prod *= lhs;
        return prod;
    }
    
    // ----------------- implementation of operator * wavefunction ------------

    // example of sign count (spin fermion model):
    // site 0: one fermion, site 1: 0 fermion, site 2: one fermion, site 3: one fermion
    // |psi> = f_0^\dagger f_2^\dagger f_3^\dagger |0>
    // f_1^\dagger |psi> = - f_0^\dagger f_1^\dagger f_2^\dagger f_3^\dagger |0>
    template <typename T>
    wavefunction<T> operator*(const opr<T> &lhs, const mbasis_elem &rhs)
    {
        wavefunction<T> res;
        assert(lhs.dim == rhs.mbits[lhs.orbital].local_dimension());
        MKL_INT col = rhs.mbits[lhs.orbital].siteRead(lhs.site);
        MKL_INT displacement = col * lhs.dim;
        if (lhs.diagonal) {
            assert(! lhs.fermion);
            if (std::abs(lhs.mat[col]) > opr_precision)
                res += std::pair<mbasis_elem, T>(rhs, lhs.mat[col]);
        } else {
            MKL_INT sgn = 0;
            if (lhs.fermion) {                         // count # of fermions traversed by this operator
                for (MKL_INT orb_cnt = 0; orb_cnt < lhs.orbital; orb_cnt++) {
                    if (rhs.mbits[orb_cnt].q_fermion()) {
                        auto &ele = rhs.mbits[orb_cnt];
                        for (MKL_INT site_cnt = 0; site_cnt < ele.total_sites(); site_cnt++) {
                            sgn = (sgn + ele.Nfermion_map[ele.siteRead(site_cnt)]) % 2;
                        }
                    }
                }
                if (rhs.mbits[lhs.orbital].q_fermion()) {
                    auto &ele = rhs.mbits[lhs.orbital];
                    for (MKL_INT site_cnt = 0; site_cnt < lhs.site; site_cnt++) {
                        sgn = (sgn + ele.Nfermion_map[ele.siteRead(site_cnt)]) % 2;
                    }
                }
            }
            for (MKL_INT row = 0; row < lhs.dim; row++) {
                auto coeff = (sgn == 0 ? lhs.mat[row + displacement] : (-lhs.mat[row + displacement]));
                if (std::abs(coeff) > opr_precision) {
                    mbasis_elem state_new(rhs);
                    state_new.mbits[lhs.orbital].siteWrite(lhs.site, row);
                    res += std::pair<mbasis_elem, T>(state_new, coeff);
                }
            }
        }
        return res;
    }

    template <typename T>
    wavefunction<T> operator*(const opr<T> &lhs, const wavefunction<T> &rhs)
    {
        if (rhs.elements.empty()) return rhs;                                       // zero wavefunction
        wavefunction<T> res;
        for (auto it = rhs.elements.begin(); it != rhs.elements.end(); it++)
            res += (it->second * (lhs * it->first));
        return res;
    }
    
    template <typename T>
    wavefunction<T> operator*(const opr_prod<T> &lhs, const mbasis_elem &rhs)
    {
        if (lhs.q_zero()) return wavefunction<T>();                                 // zero operator
        wavefunction<T> res(rhs);
        for (auto rit = lhs.mat_prod.rbegin(); rit != lhs.mat_prod.rend(); rit++) {
            res = (*rit) * res;
        }
        res *= lhs.coeff;
        return res;
    }
    
    template <typename T>
    wavefunction<T> operator*(const opr_prod<T> &lhs, const wavefunction<T> &rhs)
    {
        if (rhs.elements.empty()) return rhs;                                       // zero wavefunction
        if (lhs.q_zero()) return wavefunction<T>();                                 // zero operator
        wavefunction<T> res(rhs);
        for (auto rit = lhs.mat_prod.rbegin(); rit != lhs.mat_prod.rend(); rit++) {
            res = (*rit) * res;
        }
        res *= lhs.coeff;
        return res;
    }

    template <typename T>
    wavefunction<T> operator*(const mopr<T> &lhs, const mbasis_elem &rhs)
    {
        if (lhs.q_zero()) return wavefunction<T>();                                // zero operator
        wavefunction<T> res;
        for (auto it = lhs.mats.begin(); it != lhs.mats.end(); it++)
            res += ( (*it) * rhs );
        return res;
    }
    
    template <typename T>
    wavefunction<T> operator*(const mopr<T> &lhs, const wavefunction<T> &rhs)
    {
        if (rhs.elements.empty()) return rhs;                                      // zero wavefunction
        if (lhs.q_zero()) return wavefunction<T>();                                // zero operator
        wavefunction<T> res;
        for (auto it = lhs.mats.begin(); it != lhs.mats.end(); it++)
            res += ( (*it) * rhs );
        return res;
    }
    
    // Explicit instantiation
    template class wavefunction<double>;
    template class wavefunction<std::complex<double>>;
    
    template void swap(wavefunction<double>&, wavefunction<double>&);
    template void swap(wavefunction<std::complex<double>>&, wavefunction<std::complex<double>>&);
    
    template wavefunction<double> operator*(const wavefunction<double>&, const double&);
    template wavefunction<std::complex<double>> operator*(const wavefunction<std::complex<double>>&, const std::complex<double>&);
    
    template wavefunction<double> operator*(const double&, const wavefunction<double>&);
    template wavefunction<std::complex<double>> operator*(const std::complex<double>&, const wavefunction<std::complex<double>>&);
    
    //template double operator*(const opr<double>&, const basis_elem&);
    //template std::complex<double> operator*(const opr<std::complex<double>>&, const basis_elem&);
    
    //template double operator*(const opr<double>&, const mbasis_elem&);
    //template std::complex<double> operator*(const opr<std::complex<double>>&, const mbasis_elem&);
    
    //template double operator*(const opr_prod<double>&, const mbasis_elem&);
    //template std::complex<double> operator*(const opr_prod<std::complex<double>>&, const mbasis_elem&);
    
    template wavefunction<double> operator*(const opr<double> &lhs, const mbasis_elem &rhs);
    template wavefunction<std::complex<double>> operator*(const opr<std::complex<double>> &lhs, const mbasis_elem &rhs);
    
    template wavefunction<double> operator*(const opr<double> &lhs, const wavefunction<double> &rhs);
    template wavefunction<std::complex<double>> operator*(const opr<std::complex<double>> &lhs, const wavefunction<std::complex<double>> &rhs);

    template wavefunction<double> operator*(const opr_prod<double> &lhs, const mbasis_elem &rhs);
    template wavefunction<std::complex<double>> operator*(const opr_prod<std::complex<double>> &lhs, const mbasis_elem &rhs);
    
    template wavefunction<double> operator*(const opr_prod<double> &lhs, const wavefunction<double> &rhs);
    template wavefunction<std::complex<double>> operator*(const opr_prod<std::complex<double>> &lhs, const wavefunction<std::complex<double>> &rhs);
    
    template wavefunction<double> operator*(const mopr<double>&, const mbasis_elem&);
    template wavefunction<std::complex<double>> operator*(const mopr<std::complex<double>>&, const mbasis_elem&);
    
    template wavefunction<double> operator*(const mopr<double> &lhs, const wavefunction<double> &rhs);
    template wavefunction<std::complex<double>> operator*(const mopr<std::complex<double>> &lhs, const wavefunction<std::complex<double>> &rhs);
    
}
