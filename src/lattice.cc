#include "qbasis.h"

namespace qbasis {
    // ----------------- implementation of lattice ------------------
    lattice::lattice(const std::string &name, const std::vector<MKL_INT> &L_, const std::vector<std::string> &bc_) : L(L_), bc(bc_)
    {
        assert(L.size() == bc.size());
        dim = static_cast<MKL_INT>(L.size());
        a = std::vector<std::vector<double>>(dim, std::vector<double>(dim, 0.0));
        b = std::vector<std::vector<double>>(dim, std::vector<double>(dim, 0.0));
        if (name == "chain") {
            assert(L.size() == 1);
            num_sub = 1;
            a[0][0] = 1.0;
            b[0][0] = 2.0 * pi;
            Nsites = L[0] * num_sub;
        } else if (name == "square") {
            assert(L.size() == 2);
            num_sub = 1;
            a[0][0] = 1.0;      a[0][1] = 0.0;
            a[1][0] = 0.0;      a[1][1] = 1.0;
            b[0][0] = 2.0 * pi; b[0][1] = 0.0;
            b[1][0] = 0.0;      b[1][1] = 2.0 * pi;
            Nsites = L[0] * L[1] * num_sub;
        } else if (name == "triangular") {
            assert(L.size() == 2);
            num_sub = 1;
            a[0][0] = 1.0;      a[0][1] = 0.0;
            a[1][0] = 0.5;      a[1][1] = 0.5 * sqrt(3.0);
            b[0][0] = 2.0 * pi; b[0][1] = -2.0 * pi / sqrt(3.0);
            b[1][0] = 0.0;      b[1][1] = 4.0 * pi / sqrt(3.0);
            Nsites = L[0] * L[1] * num_sub;
        } else if (name == "cubic") {
            assert(L.size() == 3);
            num_sub = 1;
            a[0][0] = 1.0;      a[0][1] = 0.0;      a[0][2] = 0.0;
            a[1][0] = 0.0;      a[1][1] = 1.0;      a[1][2] = 0.0;
            a[2][0] = 0.0;      a[2][1] = 0.0;      a[2][2] = 1.0;
            b[0][0] = 2.0 * pi; b[0][1] = 0.0;      b[0][2] = 0.0;
            b[1][0] = 0.0;      b[1][1] = 2.0 * pi; b[1][2] = 0.0;
            b[2][0] = 0.0;      b[2][1] = 0.0;      b[2][2] = 2.0 * pi;
            Nsites = L[0] * L[1] * L[2] * num_sub;
        }
        for (MKL_INT j = 0; j < dim; j++) {
            assert(bc[j] == "pbc" || bc[j] == "PBC" || bc[j] == "obc" || bc[j] == "OBC");
        }
        
    }
    
    void lattice::coor2site(const std::vector<MKL_INT> &coor, const MKL_INT &sub, MKL_INT &site) const {
        assert(static_cast<MKL_INT>(coor.size()) == dim);
        assert(sub >= 0 && sub < num_sub);
        std::vector<MKL_INT> coor2 = {sub};
        std::vector<MKL_INT> base = {num_sub};
        coor2.insert(coor2.end(), coor.begin(), coor.end());
        base.insert(base.end(), L.begin(), L.end());
        for (MKL_INT j = 0; j <= dim; j++) {
            while(coor2[j] < 0) coor2[j] += base[j];
            while(coor2[j] >= base[j]) coor2[j] -= base[j];
        }
        site = dynamic_base(coor2, base);
    }
    
    void lattice::site2coor(std::vector<MKL_INT> &coor, MKL_INT &sub, const MKL_INT &site) const {
        assert(site >= 0 && site < Nsites);
        coor.resize(dim);
        sub = site % num_sub;
        auto temp = (site - sub) / num_sub;  // temp == i + j * L[0] + k * L[0] * L[1] + ...
        for (MKL_INT n = 0; n < dim - 1; n++) {
            coor[n] = temp % L[n];
            temp = (temp - coor[n]) / L[n];
        }
        coor[dim-1] = temp;
    }
    
    std::vector<MKL_INT> lattice::translation_plan(const std::vector<MKL_INT> &disp) const {
        assert(static_cast<MKL_INT>(disp.size()) == dim);
        std::vector<MKL_INT> result(total_sites());
        std::vector<MKL_INT> coor(dim), temp(dim);
        MKL_INT sub;
        for (MKL_INT site = 0; site < total_sites(); site++) {
            site2coor(coor, sub, site);
            for (MKL_INT j = 0; j < dim; j++) temp[j] = coor[j] + disp[j];
            coor2site(temp,sub,result[site]);
        }
        return result;
    }
    
    std::vector<MKL_INT> lattice::c4_rotation_plan() const {
        assert(dim == 2);
        assert(L[0] == L[1]);
        assert(std::abs(a[0][0] * a[1][0] + a[0][1] * a[1][1]) < opr_precision); // basis orthogonal
        std::vector<MKL_INT> result(total_sites());
        std::vector<MKL_INT> coor(dim), temp(dim);
        MKL_INT sub;
        
        // currently only the simplest case implemented: one sublattice. More complicated cases come later
        assert(num_sub == 1);
        if (num_sub == 1) {
            for (MKL_INT site = 0; site < total_sites(); site++) {
                site2coor(coor, sub, site);
                temp[0] = L[1] - 1 - coor[1];
                temp[1] = coor[0];
                coor2site(temp, sub, result[site]);
            }
        }
        return result;
    }
    
    
    std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> lattice::plan_product(const std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> &lhs,
                                                                               const std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> &rhs) const {
        assert(lhs.size() == rhs.size());
        assert(lhs[0].size() == rhs[0].size() && static_cast<MKL_INT>(lhs[0].size()) == total_sites());
        MKL_INT orb_tot = lhs.size();
        MKL_INT site_tot = lhs[0].size();
        std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> res(orb_tot, std::vector<std::pair<MKL_INT,MKL_INT>>(site_tot));
        for (MKL_INT orb0 = 0; orb0 < orb_tot; orb0++) {
            for (MKL_INT site0 = 0; site0 < site_tot; site0++) {
                auto site1 = rhs[orb0][site0].first;
                auto orb1  = rhs[orb0][site0].second;
                res[orb0][site0] = lhs[orb1][site1];
            }
        }
        return res;
    }
    
    std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> lattice::plan_inverse(const std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> &old) const {
        assert(static_cast<MKL_INT>(old[0].size()) == total_sites());
        MKL_INT orb_tot = old.size();
        MKL_INT site_tot = old[0].size();
        std::vector<std::vector<std::pair<MKL_INT,MKL_INT>>> res(orb_tot, std::vector<std::pair<MKL_INT,MKL_INT>>(site_tot));
        for (MKL_INT orb0 = 0; orb0 < orb_tot; orb0++) {
            for (MKL_INT site0 = 0; site0 < site_tot; site0++) {
                auto site1 = old[orb0][site0].first;
                auto orb1  = old[orb0][site0].second;
                res[orb1][site1].first = site0;
                res[orb1][site1].second = orb0;
            }
        }
        return res;
        
    }
    
}
