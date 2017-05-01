#include <iostream>
#include "qbasis.h"

namespace qbasis {
    // ----------------- implementation of lattice ------------------
    lattice::lattice(const std::string &name, const std::vector<uint32_t> &L_, const std::vector<std::string> &bc_) : L(L_), bc(bc_)
    {
        assert(L.size() == bc.size());
        dim = static_cast<uint32_t>(L.size());
        a = std::vector<std::vector<double>>(dim, std::vector<double>(dim, 0.0));
        b = std::vector<std::vector<double>>(dim, std::vector<double>(dim, 0.0));
        if (name == "chain" || name == "Chain" || name == "CHAIN") {
            assert(L.size() == 1);
            num_sub = 1;
            a[0][0] = 1.0;
            b[0][0] = 2.0 * pi;
            Nsites = L[0] * num_sub;
        } else if (name == "square" || name == "Square" || name == "SQUARE") {
            assert(L.size() == 2);
            num_sub = 1;
            a[0][0] = 1.0;      a[0][1] = 0.0;
            a[1][0] = 0.0;      a[1][1] = 1.0;
            b[0][0] = 2.0 * pi; b[0][1] = 0.0;
            b[1][0] = 0.0;      b[1][1] = 2.0 * pi;
            Nsites = L[0] * L[1] * num_sub;
        } else if (name == "triangular" || name == "Triangular" || name == "TRIANGULAR") {
            assert(L.size() == 2);
            num_sub = 1;
            a[0][0] = 1.0;      a[0][1] = 0.0;
            a[1][0] = 0.5;      a[1][1] = 0.5 * sqrt(3.0);
            b[0][0] = 2.0 * pi; b[0][1] = -2.0 * pi / sqrt(3.0);
            b[1][0] = 0.0;      b[1][1] = 4.0 * pi / sqrt(3.0);
            Nsites = L[0] * L[1] * num_sub;
        } else if (name == "kagome" || name == "Kagome" || name == "KAGOME") {
            assert(L.size() == 2);
            num_sub = 3;
            a[0][0] = 1.0;      a[0][1] = 0.0;
            a[1][0] = 0.5;      a[1][1] = 0.5 * sqrt(3.0);
            b[0][0] = 2.0 * pi; b[0][1] = -2.0 * pi / sqrt(3.0);
            b[1][0] = 0.0;      b[1][1] = 4.0 * pi / sqrt(3.0);
            Nsites = L[0] * L[1] * num_sub;
        } else if (name == "honeycomb" || name == "Honeycomb" || name == "HONEYCOMB") {
            assert(L.size() == 2);
            num_sub = 2;
            a[0][0] = 1.0;      a[0][1] = 0.0;
            a[1][0] = 0.5;      a[1][1] = 0.5 * sqrt(3.0);
            b[0][0] = 2.0 * pi; b[0][1] = -2.0 * pi / sqrt(3.0);
            b[1][0] = 0.0;      b[1][1] = 4.0 * pi / sqrt(3.0);
            Nsites = L[0] * L[1] * num_sub;
        } else if (name == "cubic" || name == "Cubic" || name == "CUBIC") {
            assert(L.size() == 3);
            num_sub = 1;
            a[0][0] = 1.0;      a[0][1] = 0.0;      a[0][2] = 0.0;
            a[1][0] = 0.0;      a[1][1] = 1.0;      a[1][2] = 0.0;
            a[2][0] = 0.0;      a[2][1] = 0.0;      a[2][2] = 1.0;
            b[0][0] = 2.0 * pi; b[0][1] = 0.0;      b[0][2] = 0.0;
            b[1][0] = 0.0;      b[1][1] = 2.0 * pi; b[1][2] = 0.0;
            b[2][0] = 0.0;      b[2][1] = 0.0;      b[2][2] = 2.0 * pi;
            Nsites = L[0] * L[1] * L[2] * num_sub;
        } else {
            std::cout << "Lattice not recognized! " << std::endl;
            assert(false);
        }
        
        if (num_sub % 2 == 0) {
            dim_spec = dim;
        } else {
            dim_spec = 0;
            for (uint32_t d = 0; d < dim; d++) {
                if (L[d] % 2 == 0) {
                    dim_spec = d;
                    break;
                }
            }
        }
        
        for (uint32_t j = 0; j < dim; j++) {
            assert(bc[j] == "pbc" || bc[j] == "PBC" || bc[j] == "obc" || bc[j] == "OBC");
        }
        
    }
    
    void lattice::coor2site(const std::vector<int> &coor, const int &sub, uint32_t &site) const
    {
        assert(static_cast<uint32_t>(coor.size()) == dim);
        std::vector<uint32_t> coor2, base;
        std::vector<uint32_t> dim_arr;  // let dim_spec to be counted first
        int sub_temp = sub;
        while (sub_temp < 0) sub_temp += static_cast<int>(num_sub);
        while (sub_temp >= static_cast<int>(num_sub)) sub_temp -= static_cast<int>(num_sub);
        
        if (dim_spec != dim) {
            dim_arr.push_back(dim_spec);
            for (uint32_t j = 0; j < dim; j++) {
                if (j != dim_spec) dim_arr.push_back(j);
            }
            for (uint32_t &j : dim_arr) {
                int coor_temp = coor[j];
                while (coor_temp < 0) coor_temp += L[j];
                while (coor_temp >= static_cast<int>(L[j])) coor_temp -= L[j];
                coor2.push_back(static_cast<uint32_t>(coor_temp));
                base.push_back(L[j]);
            }
            coor2.push_back(static_cast<uint32_t>(sub_temp));
            base.push_back(num_sub);
        } else {
            assert(num_sub % 2 == 0);
            for (uint32_t j = 0; j < dim; j++) dim_arr.push_back(j);
            coor2.push_back(static_cast<uint32_t>(sub_temp));
            base.push_back(num_sub);
            for (uint32_t &j : dim_arr) {
                int coor_temp = coor[j];
                while (coor_temp < 0) coor_temp += L[j];
                while (coor_temp >= static_cast<int>(L[j])) coor_temp -= L[j];
                coor2.push_back(static_cast<uint32_t>(coor_temp));
                base.push_back(L[j]);
            }
        }
        site = dynamic_base(coor2, base);
    }
    
    void lattice::site2coor(std::vector<int> &coor, int &sub, const uint32_t &site) const
    {
        assert(site < Nsites);
        coor.resize(dim);
        std::vector<uint32_t> base;
        std::vector<uint32_t> dim_arr;  // let dim_spec to be counted first
        if (dim_spec != dim) {
            dim_arr.push_back(dim_spec);
            for (uint32_t j = 0; j < dim; j++) {
                if (j != dim_spec) dim_arr.push_back(j);
            }
            for (uint32_t &j : dim_arr) base.push_back(L[j]);
            base.push_back(num_sub);
            auto coor_temp = dynamic_base(site, base);
            sub = static_cast<int>(coor_temp.back());
            for (uint32_t j = 0; j < dim; j++) coor[dim_arr[j]] = coor_temp[j];
        } else {
            for (uint32_t j = 0; j < dim; j++)  dim_arr.push_back(j);
            base.push_back(num_sub);
            for (uint32_t &j : dim_arr) base.push_back(L[j]);
            auto coor_temp = dynamic_base(site, base);
            sub = static_cast<int>(coor_temp.front());
            for (uint32_t j = 0; j < dim; j++) coor[dim_arr[j]] = coor_temp[j+1];
        }
    }
    
    std::vector<uint32_t> lattice::translation_plan(const std::vector<int> &disp) const
    {
        assert(static_cast<uint32_t>(disp.size()) == dim);
        std::vector<uint32_t> result(total_sites());
        std::vector<int> coor(dim), temp(dim);
        int sub;
        for (uint32_t site = 0; site < total_sites(); site++) {
            site2coor(coor, sub, site);
            for (uint32_t j = 0; j < dim; j++) temp[j] = coor[j] + disp[j];
            coor2site(temp,sub,result[site]);
        }
        return result;
    }
    
    std::vector<uint32_t> lattice::c4_rotation_plan() const
    {
        assert(dim == 2);
        assert(L[0] == L[1]);
        assert(std::abs(a[0][0] * a[1][0] + a[0][1] * a[1][1]) < opr_precision); // basis orthogonal
        std::vector<uint32_t> result(total_sites());
        std::vector<int> coor(dim), temp(dim);
        int sub;
        
        // currently only the simplest case implemented: one sublattice. More complicated cases come later
        assert(num_sub == 1);
        if (num_sub == 1) {
            for (uint32_t site = 0; site < total_sites(); site++) {
                site2coor(coor, sub, site);
                temp[0] = static_cast<int>(L[1]) - 1 - coor[1];
                temp[1] = coor[0];
                coor2site(temp, sub, result[site]);
            }
        }
        return result;
    }
    
    
    std::vector<std::vector<std::pair<uint32_t,uint32_t>>> lattice::plan_product(
        const std::vector<std::vector<std::pair<uint32_t,uint32_t>>> &lhs,
        const std::vector<std::vector<std::pair<uint32_t,uint32_t>>> &rhs) const
    {
        assert(lhs.size() == rhs.size());
        assert(lhs[0].size() == rhs[0].size() && static_cast<uint32_t>(lhs[0].size()) == total_sites());
        uint32_t orb_tot  = lhs.size();
        uint32_t site_tot = lhs[0].size();
        std::vector<std::vector<std::pair<uint32_t,uint32_t>>> res(orb_tot, std::vector<std::pair<uint32_t,uint32_t>>(site_tot));
        for (uint32_t orb0 = 0; orb0 < orb_tot; orb0++) {
            for (uint32_t site0 = 0; site0 < site_tot; site0++) {
                auto site1 = rhs[orb0][site0].first;
                auto orb1  = rhs[orb0][site0].second;
                res[orb0][site0] = lhs[orb1][site1];
            }
        }
        return res;
    }
    
    std::vector<std::vector<std::pair<uint32_t,uint32_t>>> lattice::plan_inverse(
        const std::vector<std::vector<std::pair<uint32_t,uint32_t>>> &old) const
    {
        assert(static_cast<uint32_t>(old[0].size()) == total_sites());
        uint32_t orb_tot = old.size();
        uint32_t site_tot = old[0].size();
        std::vector<std::vector<std::pair<uint32_t,uint32_t>>> res(orb_tot, std::vector<std::pair<uint32_t,uint32_t>>(site_tot));
        for (uint32_t orb0 = 0; orb0 < orb_tot; orb0++) {
            for (uint32_t site0 = 0; site0 < site_tot; site0++) {
                auto site1 = old[orb0][site0].first;
                auto orb1  = old[orb0][site0].second;
                res[orb1][site1].first = site0;
                res[orb1][site1].second = orb0;
            }
        }
        return res;
    }
    
}
