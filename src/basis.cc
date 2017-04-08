#include <cmath>
#include <iostream>
#include "qbasis.h"

namespace qbasis {
    // ----------------- implementation of basis ------------------
    basis_elem::basis_elem(const MKL_INT &n_sites, const MKL_INT &dim_local_):
        dim_local(static_cast<short>(dim_local_)),
        bits_per_site(static_cast<short>(ceil(log2(static_cast<double>(dim_local_)) - 1e-9))),
        Nfermion_map(std::vector<int>()),
        bits(static_cast<DBitSet::size_type>(n_sites * bits_per_site))
    {
        Nfermion_map.shrink_to_fit();
    }
    
    basis_elem::basis_elem(const MKL_INT &n_sites, const MKL_INT &dim_local_, const opr<double> &Nfermion):
        dim_local(static_cast<short>(dim_local_)),
        bits_per_site(static_cast<short>(ceil(log2(static_cast<double>(dim_local_)) - 1e-9))),
        bits(static_cast<DBitSet::size_type>(n_sites * bits_per_site))
    {
        auto Nf = Nfermion;
        Nf.simplify();
        assert(Nf.q_diagonal() && ! Nf.q_zero());
        Nfermion_map = std::vector<int>(dim_local_);
        Nfermion_map.shrink_to_fit();
        for (decltype(Nfermion_map.size()) j = 0; j < Nfermion_map.size(); j++)
            Nfermion_map[j] = static_cast<int>(ceil(Nf.mat[j] - 1e-9) + 1e-9);
    }
    
    basis_elem::basis_elem(const MKL_INT &n_sites, const std::string &s)
    {
        if (s == "spin-1/2") {
            dim_local = 2;                            // { |up>, |dn> }
            Nfermion_map = std::vector<int>();
        } else if (s == "spin-1") {                   // { |up>, |0>, |dn> }
            dim_local = 3;
            Nfermion_map = std::vector<int>();
        } else if (s == "dimer") {                    // { |s>, |t+>, |t->, |t0> } or { |s>, |tx>, |ty>, |tz> }
            dim_local = 4;
            Nfermion_map = std::vector<int>();
        } else if (s == "electron") {                 // { |0>, |up>, |dn>, |up+dn> }
            dim_local = 4;
            Nfermion_map = std::vector<int>{0,1,1,2};
        } else if (s == "tJ") {                       // { |0>, |up>, |dn> }
            dim_local = 3;
            Nfermion_map = std::vector<int>{0,1,1};
        } else if (s == "spinless-fermion") {         // { |0>, |1> }
            dim_local = 2;
            Nfermion_map = std::vector<int>{0,1};
        }
        Nfermion_map.shrink_to_fit();
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
    
    bool basis_elem::q_maximized() const
    {
        if (int_pow(2, bits_per_site) == dim_local) {
            return bits.all();
        } else {
            for (MKL_INT site = 0; site < total_sites(); site++) {
                if (siteRead(site) != dim_local - 1) return false;
            }
            return true;
        }
    }
    
    bool basis_elem::q_same_state_all_site() const
    {
        if (total_sites() <= 1) return true;
        for (MKL_INT j = 1; j < total_sites(); j++) {
            if (siteRead(j-1) != siteRead(j)) return false;
        }
        return true;
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
    
    std::vector<MKL_INT> basis_elem::statistics() const
    {
        std::vector<MKL_INT> results(dim_local,0);
        for (MKL_INT site = 0; site < total_sites(); site++) results[siteRead(site)]++;
        return results;
    }
    
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // in future, replace with quick sort to improve performance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    basis_elem &basis_elem::transform(const std::vector<MKL_INT> &plan, MKL_INT &sgn)
    {
        if (! q_fermion()) {
            sgn = 0;
        } else {
            std::vector<MKL_INT> plan_fermion;
            for (MKL_INT site = 0; site < total_sites(); site++) {
                auto state = siteRead(site);
                if (Nfermion_map[state] % 2 != 0) plan_fermion.push_back(plan[site]); // keeps all the sites which have fermion
                // using bubble sort to count how many times we are exchanging fermions
                sgn = bubble_sort(plan_fermion, 0, static_cast<MKL_INT>(plan_fermion.size())) % 2;
            }
        }
        assert(static_cast<MKL_INT>(plan.size()) == total_sites());
        auto res = *this;
        for (MKL_INT site = 0; site < total_sites(); site++) {
            MKL_INT bits_bgn1 = bits_per_site * site;
            MKL_INT bits_bgn2 = bits_per_site * plan[site];
            for (MKL_INT j = 0; j < bits_per_site; j++) res.bits[bits_bgn2+j] = bits[bits_bgn1+j];
        }
        swap(*this, res);
        return *this;
    }
    
    basis_elem &basis_elem::translate(const qbasis::lattice &latt, const std::vector<MKL_INT> &disp, MKL_INT &sgn)
    {
        assert(latt.dimension() == static_cast<MKL_INT>(disp.size()));
        auto plan = latt.translation_plan(disp);
        transform(plan, sgn);
        return *this;
    }
    
    void basis_elem::prt() const
    {
        std::cout << bits;
    }
    
    void basis_elem::prt_nonzero() const
    {
        for (MKL_INT j = 0; j < total_sites(); j++) {
            auto st = siteRead(j);
            if (st != 0) {
                std::cout << "site " << j << ", state " << st << std::endl;
            }
        }
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
    
    MKL_INT mbasis_elem::siteRead(const MKL_INT &site, const MKL_INT &orbital) const
    {
        assert(orbital < total_orbitals());
        return mbits[orbital].siteRead(site);
    }
    
    mbasis_elem &mbasis_elem::siteWrite(const MKL_INT &site, const MKL_INT &orbital, const MKL_INT &val)
    {
        assert(orbital < total_orbitals());
        mbits[orbital].siteWrite(site, val);
        return *this;
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
    
    double mbasis_elem::diagonal_operator(const mopr<double>& lhs) const
    {
        if (lhs.q_zero()) {
            return 0.0;
        } else {
            double res = 0.0;
            for (MKL_INT j = 0; j < lhs.size(); j++) {
                auto op = lhs[j];
                res += diagonal_operator(op);
            }
            return res;
        }
    }
    
    std::complex<double> mbasis_elem::diagonal_operator(const mopr<std::complex<double> > &lhs) const
    {
        if (lhs.q_zero()) {
            return 0.0;
        } else {
            std::complex<double> res = 0.0;
            for (MKL_INT j = 0; j < lhs.size(); j++) {
                auto op = lhs[j];
                res += diagonal_operator(op);
            }
            return res;
        }
    }
    
    bool mbasis_elem::q_zero() const
    {
        for (MKL_INT orb = 0; orb < total_orbitals(); orb++) {
            if (! mbits[orb].q_zero()) return false;
        }
        return true;
    }
    
    bool mbasis_elem::q_maximized() const
    {
        for (MKL_INT orb = 0; orb < total_orbitals(); orb++) {
            if (! mbits[orb].q_maximized()) return false;
        }
        return true;
    }
    
    std::vector<MKL_INT> mbasis_elem::statistics() const {
        std::vector<MKL_INT> results(local_dimension(),0);
        std::vector<MKL_INT> state(total_orbitals());
        std::vector<MKL_INT> base(total_orbitals());
        for (MKL_INT orb = 0; orb < total_orbitals(); orb++) base[orb] = mbits[orb].local_dimension();
        for (MKL_INT site = 0; site < total_sites(); site++) {
            for (MKL_INT orb = 0; orb < total_orbitals(); orb++) state[orb] = mbits[orb].siteRead(site);
            results[dynamic_base(state, base)]++;
        }
        return results;
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
    
    mbasis_elem &mbasis_elem::transform(const std::vector<MKL_INT> &plan, MKL_INT &sgn) {
        sgn = 0;
        for (auto it = mbits.begin(); it != mbits.end(); it++) {
            MKL_INT sgn0;
            it->transform(plan, sgn0);
            sgn = (sgn + sgn0) % 2;
        }
        return *this;
    }
    
    mbasis_elem &mbasis_elem::transform(const std::vector<std::vector<std::pair<MKL_INT, MKL_INT>>> &plan, MKL_INT &sgn) {
        sgn = 0;
        
        // implement fermionic transformation later
        assert(std::none_of(mbits.begin(), mbits.end(), [](basis_elem x){ return x.q_fermion(); }));
        assert(plan.size() == mbits.size());
        assert(static_cast<MKL_INT>(plan[0].size()) == total_sites());
        
        auto res = *this;
        
        for (MKL_INT orb1 = 0; orb1 < total_orbitals(); orb1++) {
            for (MKL_INT site1 = 0; site1 < total_sites(); site1++) {
                auto site2 = plan[orb1][site1].first;
                auto orb2  = plan[orb1][site1].second;
                assert(mbits[orb1].local_dimension() == mbits[orb2].local_dimension());
                MKL_INT bits_bgn1 = mbits[orb1].bits_per_site * site1;
                MKL_INT bits_bgn2 = mbits[orb2].bits_per_site * site2;
                for (MKL_INT j = 0; j < mbits[orb1].bits_per_site; j++) res.mbits[orb2].bits[bits_bgn2+j] = mbits[orb1].bits[bits_bgn1+j];
            }
        }
        swap(res, *this);
        return *this;
    }
    
    mbasis_elem &mbasis_elem::translate(const qbasis::lattice &latt, const std::vector<MKL_INT> &disp, MKL_INT &sgn) {
        auto plan = latt.translation_plan(disp);
        transform(plan, sgn);
        return *this;
    }
    
    // need re-write this function to more general cases (pbc mixing obc in different dimensions)
    mbasis_elem &mbasis_elem::translate_to_unique_state(const qbasis::lattice &latt, std::vector<MKL_INT> &disp_vec) {
        assert(latt.total_sites() == total_sites());
        MKL_INT orb_smart = 0;
        while (orb_smart < total_orbitals() && mbits[orb_smart].q_same_state_all_site()) orb_smart++;
        if (orb_smart == total_orbitals()) return *this; // if every site in the same state
        
        auto statis  = mbits[orb_smart].statistics();
        MKL_INT state_smart = 0;
        
        std::vector<std::string> pbc(latt.dimension(),"pbc");
        std::vector<std::string> obc(latt.dimension(),"obc");
        std::vector<std::string> PBC(latt.dimension(),"PBC");
        std::vector<std::string> OBC(latt.dimension(),"OBC");
        
        if (latt.boundary() == pbc || latt.boundary() == PBC) {
            state_smart = 0;
        } else if (latt.boundary() == obc || latt.boundary() == OBC) {
            state_smart = 1;
        }
        while (statis[state_smart] == 0) state_smart++;
        assert(state_smart < mbits[orb_smart].local_dimension() && statis[state_smart] > 0);
        
        // following nomenclature of trans_equiv()
        MKL_INT num_sites_smart = statis[state_smart];
        std::vector<MKL_INT> sites_smart;
        for (MKL_INT site = 0; site < mbits[orb_smart].total_sites(); site++) {                // search for sites_smart
            if (mbits[orb_smart].siteRead(site) == state_smart) sites_smart.push_back(site);
            if (static_cast<MKL_INT>(sites_smart.size()) >= num_sites_smart) break;
        }
        assert(static_cast<MKL_INT>(sites_smart.size()) == num_sites_smart);
        
        if (latt.boundary() == pbc || latt.boundary() == PBC) {
            // now we want to translate site_smart to the highest site, to minimize the state in < comparison
            std::vector<MKL_INT> coor_smart(latt.dimension());
            MKL_INT sub_smart;
            auto state_min = *this;
            auto linear_size = latt.Linear_size();
            std::vector<MKL_INT> disp(latt.dimension(),0);
            disp_vec = disp;
            for (MKL_INT cnt = 0; cnt < num_sites_smart; cnt++) {
                latt.site2coor(coor_smart, sub_smart, sites_smart[cnt]);
                for (MKL_INT j = 0; j < latt.dimension(); j++)
                    disp[j] = linear_size[j] - 1 - coor_smart[j];
                auto state_new = *this;
                MKL_INT sgn;
                state_new.translate(latt, disp, sgn);
                if(state_new < state_min) {
                    state_min = state_new;
                    disp_vec = disp;
                }
            }
            swap(state_min, *this);
        } else if (latt.boundary() == obc || latt.boundary() == OBC) {
            std::vector<MKL_INT> lowest_coors(latt.dimension());
            MKL_INT sub0;
            latt.site2coor(lowest_coors, sub0, sites_smart[0]);
            //this->prt(); std::cout << std::endl;
            std::vector<MKL_INT> highest_coors = lowest_coors;
            
            for (MKL_INT site = 0; site < latt.total_sites(); site++) {
                std::vector<MKL_INT> coor(latt.dimension());
                MKL_INT sub;
                latt.site2coor(coor, sub, site);
                std::vector<MKL_INT> temp(total_orbitals());
                for (MKL_INT orb = 0; orb < total_orbitals(); orb++) temp[orb] = siteRead(site, orb);
                if (std::any_of(temp.begin(), temp.end(), [](MKL_INT a){return a != 0; })) {
                    for (MKL_INT dim = 0; dim < latt.dimension(); dim++) {
                        if (coor[dim] < lowest_coors[dim]) lowest_coors[dim] = coor[dim];
                        if (coor[dim] > highest_coors[dim]) highest_coors[dim] = coor[dim];
                    }
                }
            }
            
            disp_vec.resize(latt.dimension());
            MKL_INT sgn;
            for (MKL_INT dim = 0; dim < latt.dimension(); dim++) {
                assert(lowest_coors[dim] >= 0 && lowest_coors[dim] < latt.Linear_size()[dim]);
                assert(highest_coors[dim] >= 0 && highest_coors[dim] < latt.Linear_size()[dim]);
                assert(lowest_coors[dim] <= highest_coors[dim]);
                disp_vec[dim] = (latt.Linear_size()[dim] - 1 - highest_coors[dim] - lowest_coors[dim])/2;
                if (lowest_coors[dim] + disp_vec[dim]
                    > latt.Linear_size()[dim] - 1 - (highest_coors[dim] + disp_vec[dim])) {
                    disp_vec[dim]--;
                }
            }
            this->translate(latt, disp_vec, sgn);
            
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
    
    MKL_INT mbasis_elem::local_dimension() const
    {
        assert(! mbits.empty());
        MKL_INT res = 1;
        for (decltype(mbits.size()) j = 0; j < mbits.size(); j++) {
            res *= mbits[j].local_dimension();
        }
        return res;
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
    
    void mbasis_elem::prt_nonzero() const
    {
        for (MKL_INT j = 0; j < total_orbitals(); j++) {
            std::cout << "orb " << j << std::endl;
            mbits[j].prt_nonzero();
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
    
    bool trans_equiv(const mbasis_elem &lhs, const mbasis_elem &rhs, const lattice &latt)
    {
        assert(latt.total_sites() == lhs.total_sites());
        auto statis  = lhs.statistics();
        auto statis2 = rhs.statistics();
        if (statis != statis2) return false;
        std::vector<std::pair<MKL_INT,MKL_INT>> statis_temp(statis.size());
        for (decltype(statis_temp.size()) j = 0; j < statis_temp.size(); j++) {
            statis_temp[j].first = j;
            statis_temp[j].second = statis[j];
            //std::cout << "stat [" << j << "] = " << statis[j] << std::endl;
        }
        std::sort(statis_temp.begin(), statis_temp.end(),
                  [](const std::pair<MKL_INT,MKL_INT> &a, const std::pair<MKL_INT,MKL_INT> &b){ return a.second < b.second; });
//        for (decltype(stat.size()) j = 0; j < stat.size(); j++) {
//            std::cout << "stat[" << stat[j].first << "] = " << stat[j].second << std::endl;
//        }
        MKL_INT min_idx = 0;
        while (statis_temp[min_idx].second < 1) min_idx++;
        assert(min_idx < static_cast<MKL_INT>(statis_temp.size()));
        if (min_idx == static_cast<MKL_INT>(statis_temp.size()) - 1) return true;     // if every site in the same state
        min_idx = statis_temp[min_idx].first;
        MKL_INT num_sites_smart = statis[min_idx];
//        std::cout << "min (but 0): " << std::endl;
//        std::cout << "statis[" << min_idx << "]=" << num_sites_smart << std::endl;
        
        std::vector<MKL_INT> base(lhs.total_orbitals());
        for (MKL_INT orb = 0; orb < static_cast<MKL_INT>(base.size()); orb++) base[orb] = lhs.mbits[orb].local_dimension();
        auto state_smart = dynamic_base(min_idx, base);
        
        MKL_INT site_lhs_smart;
        std::vector<MKL_INT> sites_rhs_smart(statis[min_idx]);
        
        // nomenclature:
        // state_smart: an array containing the states in each orbital, and we want to locate which sites have this state
        // num_sites_smart: a # representing how many sites have this state
        // site_lhs_smart:  (any) one site from lhs, which has state_smart
        // sites_rhs_smart: all sites from rhs, which have state_smart
        
        for (MKL_INT site = 0; site < lhs.total_sites(); site++) {                // search for site_lhs_smart
            MKL_INT flag = 1;
            for (MKL_INT orb = 0; orb < lhs.total_orbitals(); orb++) {
                if (lhs.mbits[orb].siteRead(site) != state_smart[orb]) {
                    flag = 0;
                    break;
                }
            }
            if (flag) {                       // a matching site found
                site_lhs_smart = site;
                break;
            }
        }
        MKL_INT cnt = 0;
        for (MKL_INT site = 0; site < rhs.total_sites(); site++) {                // search for sites_rhs_smart
            MKL_INT flag = 1;
            for (MKL_INT orb = 0; orb < rhs.total_orbitals(); orb++) {
                if (rhs.mbits[orb].siteRead(site) != state_smart[orb]) {
                    flag = 0;
                    break;
                }
            }
            if (flag) {                       // a matching site found
                sites_rhs_smart[cnt] = site;
                cnt++;
            }
            if (cnt >= num_sites_smart) break;
        }
        
        std::vector<std::string> pbc(latt.dimension(),"pbc");
        std::vector<std::string> obc(latt.dimension(),"obc");
        std::vector<std::string> PBC(latt.dimension(),"PBC");
        std::vector<std::string> OBC(latt.dimension(),"OBC");
        
        // later we can change it to be a more general consition
        // for obc, restrict the translation possibility: only local state 0 can be shifted outside boundary
        std::vector<MKL_INT> coor0(latt.dimension());
        MKL_INT sub0;
        latt.site2coor(coor0, sub0, sites_rhs_smart[0]);
        std::vector<std::vector<MKL_INT>> extremal_coors(latt.dimension(), std::vector<MKL_INT>(2));
        for (MKL_INT dim = 0; dim < latt.dimension(); dim++) {
            extremal_coors[dim][0] = coor0[dim];
            extremal_coors[dim][1] = coor0[dim];
        }
        if (latt.boundary() == obc || latt.boundary() == OBC) {
            for (MKL_INT site = 0; site < latt.total_sites(); site++) {
                std::vector<MKL_INT> coor(latt.dimension());
                MKL_INT sub;
                latt.site2coor(coor, sub, site);
                std::vector<MKL_INT> temp(rhs.total_orbitals());
                for (MKL_INT orb = 0; orb < rhs.total_orbitals(); orb++) temp[orb] = rhs.siteRead(site, orb);
                if (std::any_of(temp.begin(), temp.end(), [](MKL_INT a){return a != 0; })) {
                    for (MKL_INT dim = 0; dim < latt.dimension(); dim++) {
                        if (coor[dim] < extremal_coors[dim][0]) extremal_coors[dim][0] = coor[dim];
                        if (coor[dim] > extremal_coors[dim][1]) extremal_coors[dim][1] = coor[dim];
                    }
                }
            }
        }
        for (MKL_INT dim = 0; dim < latt.dimension(); dim++) {
            assert(extremal_coors[dim][0] >= 0 && extremal_coors[dim][0] < latt.Linear_size()[dim]);
            assert(extremal_coors[dim][1] >= 0 && extremal_coors[dim][1] < latt.Linear_size()[dim]);
            assert(extremal_coors[dim][0] <= extremal_coors[dim][1]);
        }
        
        // now we want to translate rhs to compare to lhs
        std::vector<MKL_INT> coor_lhs_smart(latt.dimension()), coor_rhs_smart(latt.dimension());
        MKL_INT sub_lhs_smart, sub_rhs_smart;
        latt.site2coor(coor_lhs_smart, sub_lhs_smart, site_lhs_smart);
        for (MKL_INT cnt = 0; cnt < num_sites_smart; cnt++) {
            latt.site2coor(coor_rhs_smart, sub_rhs_smart, sites_rhs_smart[cnt]);
            if (sub_lhs_smart != sub_rhs_smart) continue;         // no way to shift to a different sublattice
            std::vector<MKL_INT> disp(latt.dimension());
            bool flag = false;
            for (MKL_INT dim = 0; dim < latt.dimension(); dim++) {
                disp[dim] = coor_lhs_smart[dim] - coor_rhs_smart[dim];
                if (latt.boundary() == obc || latt.boundary() == OBC) {      // should not cross boundary
                    if (disp[dim] > 0 && extremal_coors[dim][1] + disp[dim] >= latt.Linear_size()[dim]) {
                        flag = true;
                        break;
                    } else if (disp[dim] < 0 && extremal_coors[dim][0] + disp[dim] < 0) {
                        flag = true;
                        break;
                    }
                }
            }
            if (flag) continue;
            auto rhs_new = rhs;
            MKL_INT sgn;
            rhs_new.translate(latt, disp, sgn);
            if (lhs == rhs_new) return true;
        }
        return false;
    }
    
    template <typename T> void gen_mbasis_by_mopr(const mopr<T> &Ham, std::list<mbasis_elem> &basis)
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::list<mbasis_elem> basis_new;
        
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT j = 0; j < static_cast<MKL_INT>(basis.size()); j++) {
            std::list<mbasis_elem> temp;
            auto it = basis.begin();
            std::advance(it, j);
            auto phi0 = *it;
            auto states_new = Ham * phi0;
            for (MKL_INT cnt = 0; cnt < states_new.size(); cnt++) {
                if (states_new[cnt].first != phi0) temp.push_back(states_new[cnt].first);
            }
            #pragma omp critical
            basis_new.splice(basis_new.end(), temp);
        }
        std::cout << "mbasis size: " << basis.size() << " -> " << (basis.size() + basis_new.size()) << "(dulp) -> ";
        basis.splice(basis.end(),basis_new);
        
        basis.sort();
        auto it = basis.begin();
        auto it_prev = it++;
        while (it != basis.end()) {
            if (*it == *it_prev) {
                it = basis.erase(it);
            } else {
                it_prev = it++;
            }
        }
        std::cout << basis.size() << ". (";
        end   = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s)" << std::endl;
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
    wavefunction<T> &wavefunction<T>::operator+=(const mbasis_elem &ele)
    {
        (*this) += std::pair<mbasis_elem, T>(ele, static_cast<T>(1.0));
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
    wavefunction<T> &wavefunction<T>::simplify()
    {
        if (! this->sorted()) {
            elements.sort([](const std::pair<mbasis_elem, T> &lhs, const std::pair<mbasis_elem, T> &rhs){return lhs.first < rhs.first; });
        }
        auto it = elements.begin();
        auto it_prev = it++;
        while (it != elements.end()) {  // combine all terms
            if (it->first == it_prev->first) {
                it_prev->second += it->second;
                it = elements.erase(it);
            } else {
                it_prev = it++;
            }
        }
        it = elements.begin();
        while (it != elements.end()) { // remove all zero terms
            if (std::abs(it->second) < opr_precision) {
                it = elements.erase(it);
            } else {
                it++;
            }
        }
        return *this;
    }
    
    template <typename T>
    bool wavefunction<T>::sorted() const
    {
        if (elements.size() == 0 || elements.size() == 1) return true;
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
    bool wavefunction<T>::sorted_fully() const
    {
        if (elements.size() == 0 || elements.size() == 1) return true;
        bool check = true;
        auto it = elements.begin();
        auto it_prev = it++;
        while (it != elements.end()) {
            if (! (it_prev->first < it->first)) {
                check = false;
                break;
            }
            it_prev = it++;
        }
        return check;
    }
    
    template <typename T>
    double wavefunction<T>::amplitude()
    {
        if (q_zero()) return 0.0;
        simplify();
        double res = 0.0;
        for (auto it = elements.begin(); it != elements.end(); it++) {
            res += std::norm(it->second);
        }
        return res;
    }
    
    template <typename T>
    void wavefunction<T>::prt() const
    {
        for (auto &ele : elements) {
            std::cout << "coeff: " << ele.second << std::endl;
            ele.first.prt();
            std::cout << std::endl;
        }
    }
    
    template <typename T>
    void wavefunction<T>::prt_nonzero() const
    {
        for (auto &ele : elements) {
            std::cout << "coeff: " << ele.second << std::endl;
            ele.first.prt_nonzero();
            std::cout << std::endl;
        }
    }
    
    template <typename T>
    void swap(wavefunction<T> &lhs, wavefunction<T> &rhs)
    {
        using std::swap;
        swap(lhs.elements, rhs.elements);
    }
    
    template <typename T>
    wavefunction<T> operator+(const wavefunction<T> &lhs, const wavefunction<T> &rhs)
    {
        wavefunction<T> sum = lhs;
        sum += rhs;
        return sum;
    }
    
    template <typename T>
    wavefunction<T> operator*(const mbasis_elem &lhs, const T &rhs)
    {
        wavefunction<T> prod;
        prod += std::pair<mbasis_elem, T>(lhs, rhs);
        return prod;
    }
    
    template <typename T>
    wavefunction<T> operator*(const T &lhs, const mbasis_elem &rhs)
    {
        wavefunction<T> prod;
        prod += std::pair<mbasis_elem, T>(rhs, lhs);
        return prod;
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
                assert(rhs.mbits[lhs.orbital].q_fermion());
                auto &ele = rhs.mbits[lhs.orbital];
                for (MKL_INT site_cnt = 0; site_cnt < lhs.site; site_cnt++) {
                    sgn = (sgn + ele.Nfermion_map[ele.siteRead(site_cnt)]) % 2;
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
    template void gen_mbasis_by_mopr(const mopr<double> &Ham, std::list<mbasis_elem> &basis);
    template void gen_mbasis_by_mopr(const mopr<std::complex<double>> &Ham, std::list<mbasis_elem> &basis);
    
    template class wavefunction<double>;
    template class wavefunction<std::complex<double>>;
    
    template void swap(wavefunction<double>&, wavefunction<double>&);
    template void swap(wavefunction<std::complex<double>>&, wavefunction<std::complex<double>>&);
    
    template wavefunction<double> operator+(const wavefunction<double>&, const wavefunction<double>&);
    template wavefunction<std::complex<double>> operator+(const wavefunction<std::complex<double>>&, const wavefunction<std::complex<double>>&);
    
    template wavefunction<double> operator*(const mbasis_elem&, const double&);
    template wavefunction<std::complex<double>> operator*(const mbasis_elem&, const std::complex<double>&);
    
    template wavefunction<double> operator*(const double&, const mbasis_elem&);
    template wavefunction<std::complex<double>> operator*(const std::complex<double>&, const mbasis_elem&);
    
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
