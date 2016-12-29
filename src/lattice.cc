#include "qbasis.h"

namespace qbasis {
    // ----------------- implementation of basis ------------------
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
    
    void lattice::coor2site(const MKL_INT &i, const MKL_INT &j, const MKL_INT &sub, MKL_INT &site) {
        assert(dim == 2);
        assert(sub >= 0 && sub < num_sub);
        MKL_INT i2 = i;
        MKL_INT j2 = j;
        while(i2 < 0) i2 += L[0];
        while(j2 < 0) j2 += L[1];
        while(i2 >= L[0]) i2 -= L[0];
        while(j2 >= L[1]) j2 -= L[1];
        site = ( i2 + j2 * L[0] ) * num_sub + sub;
    }
    
    void lattice::site2coor(MKL_INT &i, MKL_INT &j, MKL_INT &sub, const MKL_INT &site) {
        assert(dim == 2);
        assert(site >= 0 && site < Nsites);
        sub = site % num_sub;
        auto temp = (site - sub) / num_sub;  // temp == i + j * L[0]
        i = temp % L[0];
        j = (temp - i) / L[0];
    }
    
}
