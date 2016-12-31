#include "qbasis.h"

namespace qbasis {
    // ----------------- implementation of lattice ------------------
    lattice::lattice(const std::string &name, const std::string &bc_, std::initializer_list<MKL_INT> lens) : bc(bc_)
    {
        if (name == "square") {
            assert(lens.size() == 2);
            dim = 2;
            num_sub = 1;
            a = std::vector<std::vector<double>>(2, std::vector<double>(2, 0.0));
            b = a;
            a[0][0] = 1.0; a[0][1] = 0.0;
            a[1][0] = 0.0; a[1][1] = 1.0;
            b[0][0] = 2.0 * pi; b[0][1] = 0.0;
            b[1][0] = 0.0; b[1][1] = 2.0 * pi;
            L = std::vector<MKL_INT>(dim);
            auto it_arg = lens.begin();
            auto it_L   = L.begin();
            while(it_arg != lens.end()) {
                *it_L = *it_arg;
                it_L++;
                it_arg++;
            }
            Nsites = L[0] * L[1];
        }
    }
    
    void lattice::coor2site(const std::vector<MKL_INT> &coor, const MKL_INT &sub, MKL_INT &site) const {
        assert(coor.size() == dim);
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
        assert(disp.size() == dim);
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
    
    
}
