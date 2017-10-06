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
        
        dim_spec = dim;
        if (num_sub % 2 != 0) {
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
        
        site2coor_map.resize(Nsites);
        coor2site_map.resize(num_sub);
        for (uint32_t j = 0; j < Nsites; j++) {
            std::vector<int> coor;
            int sub;
            site2coor_old(coor, sub, j);
            site2coor_map[j].first  = coor;
            site2coor_map[j].second = sub;
            coor2site_map[sub][coor] = j;
        }
        
    }
    
    bool lattice::q_dividable() const {
        if (total_sites() % 2 != 0) return false;
        if (dim_spec == dim && num_sub % 2 != 0) return false;
        return true;
    }
    
    void lattice::coor2site(const std::vector<int> &coor, const int &sub, uint32_t &site, std::vector<int> &work) const
    {
        assert(coor.size() == dim);
        if (work.size() != dim) work.resize(dim);
        int sub_temp = sub;
        if (sub_temp < 0 || sub_temp >= static_cast<int>(num_sub)) sub_temp = sub % static_cast<int>(num_sub);
        if (sub_temp < 0) sub_temp += static_cast<int>(num_sub);
        for (uint32_t d = 0; d < dim; d++) {
            work[d] = coor[d];
            if (work[d] < 0 || work[d] >= static_cast<int>(L[d])) work[d] = coor[d] % static_cast<int>(L[d]);
            if (work[d] < 0) work[d] += static_cast<int>(L[d]);
        }
        site = coor2site_map[sub_temp].at(work);
    }
    
    void lattice::coor2site_old(const std::vector<int> &coor, const int &sub, uint32_t &site) const
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
        site = dynamic_base<uint32_t,uint32_t>(coor2, base);
    }
    
    void lattice::site2coor(std::vector<int> &coor, int &sub, const uint32_t &site) const
    {
        assert(site < Nsites);
        coor = site2coor_map[site].first;
        sub  = site2coor_map[site].second;
        return;
    }
    
    void lattice::site2coor_old(std::vector<int> &coor, int &sub, const uint32_t &site) const
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
    
    void lattice::coor2cart(const std::vector<int> &coor, const int &sub, std::vector<double> &cart) const
    {
        // temporarily assume only 1 sublattice
        assert(num_sub == 1);
        assert(sub == 0);
        if (cart.size() != dim) cart.resize(dim);
        std::fill(cart.begin(), cart.end(), 0.0);
        for (uint32_t d_out = 0; d_out < dim; d_out++) {                         // loop over x,y,z
            for (uint32_t d_in = 0; d_in < dim; d_in++) {                        // loop over a0, a1,...
                cart[d_out] += coor[d_in] * a[d_in][d_out];
            }
        }
    }
    
    std::vector<std::vector<uint32_t>> lattice::divisor_v1(const std::vector<bool> &trans_sym) const
    {
        assert(trans_sym.size() == dim);
        std::vector<std::vector<uint32_t>> res(dim, std::vector<uint32_t>{1});
        
        for (uint32_t d = 0; d < dim; d++) {
            if (trans_sym[d]) {
                assert(bc[d] == "pbc" || bc[d] == "PBC");
                for (uint32_t j = 2; j <= L[d]; j++) {
                    if (L[d] % j == 0) res[d].push_back(j);
                }
            }
        }
        return res;
    }
    
    /*
    std::vector<std::vector<uint32_t>> lattice::divisor_v2(const std::vector<bool> &trans_sym) const
    {
        assert(trans_sym.size() == dim);
        std::vector<std::vector<uint32_t>> res;
        std::vector<uint32_t> base;                       // for enumerating the possible translations
        for (uint32_t d = 0; d < dim; d++) {
            if (trans_sym[d]) {
                assert(bc[d] == "pbc" || bc[d] == "PBC");
                base.push_back(L[d]);
            }
        }
        std::vector<uint32_t> disp(base.size(),0);
        std::vector<uint32_t> disp2(dim,1);               // each dimension = disp[d] + 1
        res.push_back(disp2);
        if (! base.empty()) {
            disp = dynamic_base_plus1(disp, base);
            while (! dynamic_base_overflow(disp, base)) {
                bool dividable = true;
                uint32_t pos = 0;
                disp2.clear();
                for (uint32_t d = 0; d < dim; d++) {
                    if (trans_sym[d]) {
                        disp2.push_back(disp[pos++] + 1);
                    } else {
                        disp2.push_back(1);
                    }
                    if (L[d] % disp2[d] != 0) {
                        dividable = false;
                        break;
                    }
                }
                if (dividable) res.push_back(disp2);
                disp = dynamic_base_plus1(disp, base);
            }
        }
        return res;
    }
    */
    
    uint32_t volume_from_vec(const std::vector<uint32_t> &vecs)
    {
        assert(vecs.size() == 1 || vecs.size() == 4 || vecs.size() == 9);
        if (vecs.size() == 1) {
            return vecs[0];
        } else if (vecs.size() == 4) {
            int x = static_cast<int>(vecs[0] * vecs[3]);
            int y = static_cast<int>(vecs[1] * vecs[2]);
            int res = (x > y ? (x-y) : (y-x));
            return static_cast<uint32_t>(res);
        } else {
            int x = static_cast<int>(vecs[0] * vecs[4] * vecs[8] + vecs[1] * vecs[5] * vecs[6] + vecs[2] * vecs[3] * vecs[7]);
            int y = static_cast<int>(vecs[0] * vecs[5] * vecs[7] + vecs[1] * vecs[3] * vecs[8] + vecs[2] * vecs[4] * vecs[6]);
            int res = (x > y ? (x-y) : (y-x));
            return static_cast<uint32_t>(res);
        }
    }
    
    std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> lattice::trans_subgroups(const std::vector<bool> &trans_sym) const
    {
        assert(trans_sym.size() == dim);
        std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> res;
        
        uint32_t dim_trans = 0;
        std::vector<uint32_t> L_trans;
        std::vector<uint32_t> base0;                       // for enumerating the possible translations
        uint32_t lattice_size = 1;
        for (uint32_t d = 0; d < dim; d++) {
            if (trans_sym[d]) {
                assert(bc[d] == "pbc" || bc[d] == "PBC");
                dim_trans++;
                L_trans.push_back(L[d]);
                base0.push_back(L[d]+1);
                lattice_size *= L[d];
            }
        }
        std::vector<uint32_t> base;
        for (decltype(base0.size()) d = 0; d < base0.size(); d++) base.insert(base.end(), base0.begin(), base0.end());
        
        // the translational symmetric part of the original lattice
        lattice latt_trans;
        if (dim_trans == 1) {
            latt_trans = lattice("chain",  L_trans, std::vector<std::string>(dim_trans,"pbc"));
        } else if (dim == 2) {
            latt_trans = lattice("square", L_trans, std::vector<std::string>(dim_trans,"pbc"));
        } else {
            latt_trans = lattice("cubic",  L_trans, std::vector<std::string>(dim_trans,"pbc"));
        }
        assert(lattice_size = latt_trans.total_sites());
        
        std::cout << "Translation subgroups on lattice with size ";
        for (uint32_t d = 0; d < dim_trans - 1; d++) {
            std::cout << L_trans[d] << " x ";
        }
        std::cout << L_trans[dim_trans-1] << ":" << std::endl;
        
        std::vector<uint32_t> disp_total(base.size(),0);
        while (! dynamic_base_overflow(disp_total, base)) {
            uint32_t unitcell_size = volume_from_vec(disp_total);               // NOT always equal to omega_g
            // Here we make an assumption (need proof in future):
            // for those whose unitcell_size != omega_g, they are always equivalent to another set of basis,
            // which satisfies unitcell_size == omega_g.
            // with such assumption, we no longer care such exceptions.
            if (unitcell_size == 0 || lattice_size % unitcell_size != 0) {      // not a qualified set of basis
                disp_total = dynamic_base_plus1(disp_total, base);
                continue;
            }
            
            std::pair<std::vector<std::vector<uint32_t>>,uint32_t> ele;
            ele.second = unitcell_size;
            ele.first.resize(dim_trans);
            for (auto &it : ele.first) it.resize(dim_trans);
            for (uint32_t d_ou = 0; d_ou < dim_trans; d_ou++) {
                for (uint32_t d_in = 0; d_in < dim_trans; d_in++) {
                    ele.first[d_ou][d_in] = disp_total[d_in + d_ou * dim_trans];
                }
            }
            res.push_back(ele);
            disp_total = dynamic_base_plus1(disp_total, base);
        }
        
        // for each basis combination, create a covering
        std::vector<std::pair<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>, std::vector<uint32_t>>> covering(res.size());
        std::vector<uint32_t> temp(lattice_size);
        for (uint32_t j = 0; j < lattice_size; j++) temp[j] = j;
        for (decltype(covering.size()) j = 0; j < covering.size(); j++) {
            covering[j].first  = res[j];
            covering[j].second = temp;
        }
        res.clear();
        
        // for each basis vector, draw the covering
        // use OPENMP here!!!
        for (decltype(covering.size()) j = 0; j < covering.size(); j++) {
            uint32_t omega_g = covering[j].first.second;
            assert(omega_g > 0 && omega_g <= lattice_size);
            std::vector<uint32_t> covering_list;
            for (uint32_t loop = 0; loop < omega_g; loop++) {
                // find the position of the 1st number, which is not in the covering_list
                uint32_t pos0 = 0;
                while ((! covering_list.empty()) &&
                       pos0 < lattice_size &&
                       covering[j].second[pos0] <= covering_list.back()) pos0++;
                if (pos0 >= lattice_size) {                                     // omega_g != unitcell_size
                    covering[j].first.second = loop;                            // recalculate omega_g
                    break;
                }
                covering_list.push_back(covering[j].second[pos0]);
                std::vector<int> coor0;
                int sub0 = 0;
                latt_trans.site2coor(coor0, sub0, pos0);
                
                // paint all translational equivalents to the same number
                // naive way, improve in future
                std::vector<uint32_t> base_naive(dim_trans,lattice_size);
                std::vector<uint32_t> coor_naive(dim_trans,0);
                coor_naive = dynamic_base_plus1(coor_naive, base_naive);
                int sub1 = 0;
                uint32_t pos1;
                std::vector<int> work(coor0);
                while (! dynamic_base_overflow(coor_naive, base_naive)) {
                    std::vector<int> coor1(coor0);
                    for (uint32_t d_ou = 0; d_ou < dim_trans; d_ou++) {
                        for (uint32_t d_in = 0; d_in < dim_trans; d_in++) {
                            coor1[d_in] += static_cast<int>(coor_naive[d_ou] * covering[j].first.first[d_ou][d_in]);
                        }
                    }
                    latt_trans.coor2site(coor1, sub1, pos1, work);
                    covering[j].second[pos1] = covering_list.back();
                    coor_naive = dynamic_base_plus1(coor_naive, base_naive);
                }
            }
        }
        
        /*
        std::cout << " ------------------ " << std::endl;
        for (uint32_t j = 0; j < covering.size(); j++) {
            std::cout << "j = " << j << std::endl;
            std::cout << "{ ";
            for (uint32_t d_ou=0; d_ou < dim_trans; d_ou++) {
                std::cout << "(";
                for (uint32_t d_in = 0; d_in < dim_trans; d_in++) {
                    std::cout << covering[j].first.first[d_ou][d_in] << ",";
                }
                std::cout << "), ";
            }
            std::cout << covering[j].first.second << " }" << std::endl;
            for (uint32_t kk = 0; kk < lattice_size; kk++) {
                std::cout << covering[j].second[kk] << ",";
            }
            std::cout << std::endl << std::endl;
        }
        std::cout << " ------------------ " << std::endl;
        */
        
        // sort the basis with the coverings, then you see lots of dulplicates
        auto cmp = [](const std::pair<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>, std::vector<uint32_t>> &lhs,
                      const std::pair<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>, std::vector<uint32_t>> &rhs)
        {
            if (lhs.first.second == rhs.first.second) {
                auto &llhs = lhs.first;
                auto &rrhs = rhs.first;
                if (lhs.second == rhs.second) {
                    std::vector<uint32_t> lhs_len(llhs.first.size(),0), rhs_len(rrhs.first.size(),0);
                    for (uint32_t d_ou = 0; d_ou < lhs_len.size(); d_ou++) {
                        for (uint32_t d_in = 0; d_in < lhs_len.size(); d_in++) {
                            lhs_len[d_ou] += llhs.first[d_ou][d_in] * llhs.first[d_ou][d_in];
                            rhs_len[d_ou] += rrhs.first[d_ou][d_in] * rrhs.first[d_ou][d_in];
                        }
                    }
                    return lhs_len < rhs_len;
                } else {
                    return lhs.second < rhs.second;
                }
            } else {
                return lhs.first.second < rhs.first.second;
            }
        };
        std::sort(covering.begin(), covering.end(),cmp);
        
        /*
        for (uint32_t j = 0; j < covering.size(); j++) {
            std::cout << "j = " << j << std::endl;
            std::cout << "{ ";
            for (uint32_t d_ou=0; d_ou < dim_trans; d_ou++) {
                std::cout << "(";
                for (uint32_t d_in = 0; d_in < dim_trans; d_in++) {
                    std::cout << covering[j].first.first[d_ou][d_in] << ",";
                }
                std::cout << "), ";
            }
            std::cout << covering[j].first.second << " }" << std::endl;
            for (uint32_t kk = 0; kk < lattice_size; kk++) {
                std::cout << covering[j].second[kk] << ",";
            }
            std::cout << std::endl << std::endl;
        }
        */
        
        assert(! covering.empty());
        std::vector<uint32_t> pattern;
        for (decltype(covering.size()) j = 0; j < covering.size(); j++) {
            if (covering[j].second == pattern) { // dulplicate
                continue;
            } else {
                auto &ele_old = covering[j].first;
                auto ele_new = ele_old;
                ele_new.first.resize(dim);
                // resize each basis
                uint32_t d_ou_pos = 0;
                for (uint32_t d_ou = 0; d_ou < dim; d_ou++) {
                    if (trans_sym[d_ou]) {
                        ele_new.first[d_ou].resize(dim);
                        uint32_t d_in_pos = 0;
                        for (uint32_t d_in = 0; d_in < dim; d_in++) {
                            if (trans_sym[d_in]) {
                                ele_new.first[d_ou][d_in] = ele_old.first[d_ou_pos][d_in_pos++];
                            } else {
                                ele_new.first[d_ou][d_in] = 0;
                            }
                        }
                        d_ou_pos++;
                    } else {
                        ele_new.first[d_ou] = std::vector<uint32_t>(dim,0);
                    }
                }
                
                res.push_back(ele_new);
                pattern = covering[j].second;
                
                auto check_pattern = pattern;
                std::sort(check_pattern.begin(),check_pattern.end());
                auto it = std::unique(check_pattern.begin(),check_pattern.end());
                assert(std::distance(check_pattern.begin(), it) == covering[j].first.second);
                
                std::cout << "g = " << res.size() - 1 << ",\t";
                std::cout << "{ ";
                for (uint32_t d_ou=0; d_ou < dim_trans; d_ou++) {
                    std::cout << "(";
                    for (uint32_t d_in = 0; d_in < dim_trans; d_in++) {
                        std::cout << res.back().first[d_ou][d_in] << ",";
                    }
                    std::cout << "), ";
                }
                std::cout << res.back().second << " }, pattern =\t";
                for (auto &eee : pattern) std::cout << eee << ",";
                std::cout << std::endl;
                
            }
        }
        std::cout << std::endl;
        res.shrink_to_fit();
        
        return res;
    }
    
    /*
    std::vector<uint32_t> lattice::translation_plan(const std::vector<int> &disp) const
    {
        assert(static_cast<uint32_t>(disp.size()) == dim);
        std::vector<uint32_t> result(total_sites());
        std::vector<int> coor(dim), work(dim);
        int sub;
        for (uint32_t site = 0; site < total_sites(); site++) {
            site2coor(coor, sub, site);
            for (uint32_t j = 0; j < dim; j++) coor[j] += disp[j];
            coor2site(coor,sub,result[site], work);
        }
        return result;
    }
    */
    
    void lattice::translation_plan(std::vector<uint32_t> &plan, const std::vector<int> &disp,
                                   std::vector<int> &scratch_coor, std::vector<int> &scratch_work) const
    {
        if (plan.size() != Nsites) plan.resize(Nsites);
        assert(disp.size() == dim);
        if (scratch_coor.size() != dim) scratch_coor.resize(dim);
        if (scratch_work.size() != dim) scratch_work.resize(dim);
        int sub;
        for (uint32_t site = 0; site < Nsites; site++) {
            site2coor(scratch_coor, sub, site);
            for (uint32_t j = 0; j < dim; j++) scratch_coor[j] += disp[j];
            coor2site(scratch_coor,sub,plan[site], scratch_work);
        }
    }
    
    std::vector<uint32_t> lattice::rotation_plan(const uint32_t &origin, const double &angle) const
    {
        assert(dim == 2);
        std::vector<uint32_t> result(total_sites());
        std::vector<int> coor(dim), work(dim);
        std::vector<double> x0(dim), x1(dim), xwork(dim);
        int sub;
        
        // obtain x0
        site2coor(coor, sub, origin);
        x0[0] = coor[0] * a[0][0] + coor[1] * a[1][0];
        x0[1] = coor[0] * a[0][1] + coor[1] * a[1][1];
        
        // rotation matrix
        std::vector<double> matR(4);
        matR[0] = cos(angle);
        matR[1] = sin(angle);
        matR[2] = -matR[1];
        matR[3] = matR[0];
        
        // currently only the simplest case implemented: one sublattice. More complicated cases come later
        assert(num_sub == 1);
        if (num_sub == 1) {
            for (uint32_t site = 0; site < total_sites(); site++) {
                site2coor(coor, sub, site);
                xwork[0] = coor[0] * a[0][0] + coor[1] * a[1][0] - x0[0];
                xwork[1] = coor[0] * a[0][1] + coor[1] * a[1][1] - x0[1];
                x1[0] = x0[0] + matR[0] * xwork[0] + matR[2] * xwork[1];
                x1[1] = x0[1] + matR[1] * xwork[0] + matR[3] * xwork[1];
                xwork[0] = ( b[0][0] * x1[0] + b[0][1] * x1[1] ) * 0.5 / pi;
                xwork[1] = ( b[1][0] * x1[0] + b[1][1] * x1[1] ) * 0.5 / pi;
                coor[0] = static_cast<int>(xwork[0] >= 0 ? xwork[0] + 0.5 : xwork[0] - 0.5);
                coor[1] = static_cast<int>(xwork[1] >= 0 ? xwork[1] + 0.5 : xwork[1] - 0.5);
                if (std::abs(coor[0] - xwork[0]) > opr_precision || std::abs(coor[1] - xwork[1]) > opr_precision)
                    std::cerr << "Lattice rotation failed!";
                coor2site(coor, sub, result[site], work);
            }
        }
        
        // check no repetition
        auto result_check = result;
        std::sort(result_check.begin(), result_check.end());
        assert(is_sorted_norepeat(result_check));
        
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
            
    lattice divide_lattice(const lattice &parent)
    {
        lattice child(parent);
        auto dim_spec = parent.dim_spec;
        auto dim = parent.dim;
        assert(parent.total_sites() % 2 == 0);
        if (dim_spec == dim) {
            assert(parent.num_sub % 2 == 0);
            child.num_sub /= 2;
        } else {
            child.L[dim_spec] /= 2;
            for (uint32_t d = 0; d < dim; d++) {
                child.a[dim_spec][d] *= 2;
                child.b[dim_spec][d] /= 2;
            }
        }
        child.Nsites /= 2;
        
        
        child.site2coor_map.clear();
        child.coor2site_map.clear();
        child.site2coor_map.resize(child.Nsites);
        child.coor2site_map.resize(child.num_sub);
        for (uint32_t j = 0; j < child.Nsites; j++) {
            std::vector<int> coor;
            int sub;
            child.site2coor_old(coor, sub, j);
            child.site2coor_map[j].first  = coor;
            child.site2coor_map[j].second = sub;
            child.coor2site_map[sub][coor] = j;
        }
        
        return child;
    }
    
}
