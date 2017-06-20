#include <iostream>
#include <bitset>
#include <algorithm>
#include "qbasis.h"

namespace qbasis {
    
    // -------------- implementation of basis_prop ---------------
    basis_prop::basis_prop(const uint32_t &n_sites, const uint8_t &dim_local_,
                           const std::vector<uint32_t> &Nf_map,
                           const bool &dilute_):
        dim_local(dim_local_),
        num_sites(n_sites),
        Nfermion_map(Nf_map),
        dilute(dilute_)
    {
        bits_per_site = static_cast<uint8_t>(ceil(log2(static_cast<double>(dim_local)) - 1e-9));
        name = "unknown";
        if (! dilute_) {
            num_bytes = static_cast<uint16_t>((bits_per_site * num_sites) / 8 + 1);
            bits_ignore = static_cast<uint8_t>(num_bytes * 8 - bits_per_site * num_sites);
        } else {
            assert(false); // modify later
        }
        
    }
    
    basis_prop::basis_prop(const uint32_t &n_sites, const std::string &s, const extra_info &ex):
        num_sites(n_sites), name(s)
    {
        if (s == "spin-1/2") {
            dim_local = 2;                            // { |up>, |dn> }
            Nfermion_map = std::vector<uint32_t>();
            dilute = false;
        } else if (s == "spin-1") {                   // { |up>, |0>, |dn> }
            dim_local = 3;
            Nfermion_map = std::vector<uint32_t>();
            dilute = false;
        } else if (s == "dimer") {                    // { |s>, |t+>, |t->, |t0> } or { |s>, |tx>, |ty>, |tz> }
            dim_local = 4;
            Nfermion_map = std::vector<uint32_t>();
            dilute = false;
        } else if (s == "electron") {                 // { |0>, |up>, |dn>, |up+dn> }
            dim_local = 4;
            Nfermion_map = std::vector<uint32_t>{0,1,1,2};
            dilute = false;
        } else if (s == "tJ") {                       // { |0>, |up>, |dn> }
            dim_local = 3;
            Nfermion_map = std::vector<uint32_t>{0,1,1};
            dilute = false;
        } else if (s == "spinless-fermion") {         // { |0>, |1> }
            dim_local = 2;
            Nfermion_map = std::vector<uint32_t>{0,1};
            dilute = false;
        } else if (s == "boson") {                    // { |0>, |1>, ... |Nmax> }
            dim_local = ex.Nmax + 1;
            Nfermion_map = std::vector<uint32_t>();
            dilute = false;
        } else {
            std::cout << "basis " << s << " not provided yet" << std::endl;
            assert(false);
        }
        bits_per_site = static_cast<uint8_t>(ceil(log2(static_cast<double>(dim_local)) - 1e-9));
        assert(bits_per_site <= 8);
        if (! dilute) {
            num_bytes = static_cast<uint16_t>((bits_per_site * num_sites) / 8 + 1);
            bits_ignore = static_cast<uint8_t>(num_bytes * 8 - bits_per_site * num_sites);
        } else {
            assert(false); // modify later
        }
    }
    
    
    
    // ----------------- implementation of mbasis ------------------
    mbasis_elem::mbasis_elem(const std::vector<basis_prop> &props)
    {
        uint16_t total_bytes = 2; // first 2 bytes used for storing the length (in terms of bytes) of mbits
        for (decltype(props.size()) orb = 0; orb < props.size(); orb++) total_bytes += props[orb].num_bytes;
        mbits = static_cast<uint8_t*>(malloc(total_bytes * sizeof(uint8_t)));
        
        mbits[0] = total_bytes / 256;
        mbits[1] = total_bytes % 256;
        
        // later we can allow to initialize to specific value
        for (uint16_t byte_pos = 2; byte_pos < total_bytes; byte_pos++) mbits[byte_pos] = 0;
    }
    
    mbasis_elem::mbasis_elem(const mbasis_elem& old)
    {
        if (old.mbits != nullptr) {
            uint16_t total_bytes = static_cast<uint16_t>(old.mbits[0] * 256) + static_cast<uint16_t>(old.mbits[1]);
            mbits = static_cast<uint8_t*>(malloc(total_bytes * sizeof(uint8_t)));
            for (uint16_t byte_pos = 0; byte_pos < total_bytes; byte_pos++) mbits[byte_pos] = old.mbits[byte_pos];
        } else {
            mbits = nullptr;
        }
#ifdef DEBUG
//        printf("copy from &mbits = %p  to  %p\n",static_cast<void*>(old.mbits),static_cast<void*>(mbits));
#endif
    }
    
    mbasis_elem::mbasis_elem(mbasis_elem &&old) noexcept
    {
#ifdef DEBUG
//        printf("move from &mbits = %p\n",static_cast<void*>(old.mbits));
#endif
        mbits = old.mbits;
        old.mbits = nullptr;
    }
    
    mbasis_elem::~mbasis_elem()
    {
#ifdef DEBUG
//        printf("desctuctor used, &mbits = %p\n",static_cast<void*>(mbits));
#endif
        if (mbits != nullptr)
        {
            free(mbits);
            mbits = nullptr;
        }
    }
    
    uint8_t mbasis_elem::siteRead(const std::vector<basis_prop> &props,
                                  const uint32_t &site, const uint32_t &orbital) const
    {
        assert(orbital < props.size());
        uint16_t byte_pos = 2;
        for (uint32_t orb = 0; orb < orbital; orb++) byte_pos += props[orb].num_bytes;
        
        uint8_t res;
        if (! props[orbital].dilute) {
            uint32_t bits_per_site = props[orbital].bits_per_site;
            uint32_t bit_pos = bits_per_site * site;
            byte_pos += bit_pos / 8;
            bit_pos %= 8;
            if (bit_pos + bits_per_site <= 8) {
                uint8_t mask = (1 << bits_per_site) - 1;
                res = (mbits[byte_pos] >> bit_pos);
                res &= mask;
            } else {   // crossing boundary
                uint16_t mask = (static_cast<uint16_t>(1) << bits_per_site) - 1;
                uint16_t res_temp = (static_cast<uint16_t>(mbits[byte_pos+1]) << 8) | mbits[byte_pos];
                res = static_cast<uint8_t>((res_temp >> bit_pos) & mask);
            }
        } else {
            assert(false); // implement later
        }
        return res;
    }
    
    mbasis_elem &mbasis_elem::siteWrite(const std::vector<basis_prop> &props,
                                        const uint32_t &site, const uint32_t &orbital, const uint8_t &val)
    {
        assert(orbital < props.size());
        assert(site < props[orbital].num_sites);
        uint16_t byte_pos = 2;
        for (uint32_t orb = 0; orb < orbital; orb++) byte_pos += props[orb].num_bytes;
        
        if (! props[orbital].dilute) {
            uint32_t bits_per_site = props[orbital].bits_per_site;
            uint32_t bit_pos = bits_per_site * site;
            byte_pos += bit_pos / 8;
            bit_pos %= 8;
            if (bit_pos + bits_per_site <= 8) {
                uint8_t mask = ~(((1 << bits_per_site) - 1) << bit_pos);
                mbits[byte_pos] &= mask;
                mbits[byte_pos] |= (val << bit_pos);
            } else {   // crossing boundary
                uint32_t num_bits_l = 8 - bit_pos;
                uint32_t num_bits_h = bits_per_site - num_bits_l;
                uint8_t mask_h = ~((1 << num_bits_h) - 1);
                mbits[byte_pos+1] &= mask_h;
                mbits[byte_pos+1] |= (val >> num_bits_l);
                uint8_t mask_l = (1 << bit_pos) - 1;
                mbits[byte_pos] &= mask_l;
                mbits[byte_pos] |= static_cast<uint8_t>(val << bit_pos);
            }
        } else {
            assert(false); // implement later
        }
        return *this;
    }
    
    mbasis_elem &mbasis_elem::reset(const std::vector<basis_prop> &props, const uint32_t &orbital)
    {
        assert(orbital < props.size());
        uint16_t byte_pos_bgn = 2;
        for (uint32_t orb = 0; orb < orbital; orb++) byte_pos_bgn += props[orb].num_bytes;
        uint16_t byte_pos_end = byte_pos_bgn + props[orbital].num_bytes;
        for (uint16_t byte_pos = byte_pos_bgn; byte_pos < byte_pos_end; byte_pos++) mbits[byte_pos] = 0;
        return *this;
    }
    
    mbasis_elem &mbasis_elem::reset()
    {
        if (mbits == nullptr) return *this;
        uint16_t total_bytes = static_cast<uint16_t>(mbits[0] * 256) + static_cast<uint16_t>(mbits[1]);
        for (uint16_t byte_pos = 2; byte_pos < total_bytes; byte_pos++) mbits[byte_pos] = 0;
        return *this;
    }
    
    mbasis_elem &mbasis_elem::increment(const std::vector<basis_prop> &props, const uint32_t &orbital)
    {
        assert(! q_maximized(props,orbital));
        auto dim_local = props[orbital].dim_local;
        auto num_sites = props[orbital].num_sites;
        
        if (int_pow<uint8_t,uint64_t>(2, props[orbital].bits_per_site) == static_cast<uint64_t>(dim_local)) {     // no waste bit
            uint16_t byte_pos_bgn = 2;
            for (uint32_t orb = 0; orb < orbital; orb++) byte_pos_bgn += props[orb].num_bytes;
            uint16_t byte_pos_end = byte_pos_bgn + props[orbital].num_bytes;
            for (uint16_t byte_pos = byte_pos_bgn; byte_pos < byte_pos_end; byte_pos++) {
                if (mbits[byte_pos] != 255u) {
                    mbits[byte_pos]++;
                    break;
                } else {
                    mbits[byte_pos] = 0;
                }
            }
        } else {
            uint8_t val = siteRead(props, 0, orbital) + 1;
            uint32_t site = 0;

            while (val >= dim_local && site < num_sites) {
                siteWrite(props, site, orbital, val - dim_local);
                val = siteRead(props, ++site, orbital) + 1;
            }
            assert(site < num_sites);
            siteWrite(props, site, orbital, val);
        }
        return *this;
    }
    
    mbasis_elem &mbasis_elem::increment(const std::vector<basis_prop> &props)
    {
        assert(! q_maximized(props));
        for (decltype(props.size()) orb = 0; orb < props.size(); orb++) {
            if (q_maximized(props,orb)) {
                reset(props, orb);
            } else {
                increment(props, orb);
                break;
            }
        }
        return *this;
    }
    
    bool mbasis_elem::q_zero(const std::vector<basis_prop> &props, const uint32_t &orbital) const
    {
        assert(orbital < props.size());
        uint16_t byte_pos_bgn = 2;
        for (uint32_t orb = 0; orb < orbital; orb++) byte_pos_bgn += props[orb].num_bytes;
        uint16_t byte_pos_end = byte_pos_bgn + props[orbital].num_bytes;
        for (uint16_t byte_pos = byte_pos_bgn; byte_pos < byte_pos_end; byte_pos++) {
            if (mbits[byte_pos] != 0) return false;
        }
        return true;
    }
    
    bool mbasis_elem::q_zero() const
    {
        uint16_t total_bytes = static_cast<uint16_t>(mbits[0] * 256) + static_cast<uint16_t>(mbits[1]);
        for (uint16_t byte_pos = 2; byte_pos < total_bytes; byte_pos++) {
            if (mbits[byte_pos] != 0) return false;
        }
        return true;
    }
    
    bool mbasis_elem::q_maximized(const std::vector<basis_prop> &props, const uint32_t &orbital) const
    {
        assert(orbital < props.size());
        auto dim_local = props[orbital].dim_local;
        
        if (int_pow<uint8_t,uint64_t>(2, props[orbital].bits_per_site) == static_cast<uint64_t>(dim_local)) {     // no waste bit
            uint16_t byte_pos_first = 2;
            uint16_t byte_pos_last;
            for (uint32_t orb = 0; orb < orbital; orb++) byte_pos_first += props[orb].num_bytes;
            byte_pos_last = byte_pos_first + props[orbital].num_bytes - 1;
            for (auto byte_pos = byte_pos_first; byte_pos < byte_pos_last; byte_pos++) {
                if (mbits[byte_pos] != 255u) return false;
            }
            return mbits[byte_pos_last] == static_cast<uint8_t>((1 << (8 - props[orbital].bits_ignore)) - 1);
        } else {
            for (decltype(props[orbital].num_sites) site = 0; site < props[orbital].num_sites; site++) {
                if (siteRead(props, site, orbital) != dim_local - 1) return false;
            }
            return true;
        }
    }
    
    bool mbasis_elem::q_maximized(const std::vector<basis_prop> &props) const
    {
        for (decltype(props.size()) orb = 0; orb < props.size(); orb++) {
            if (! q_maximized(props, orb)) return false;
        }
        return true;
    }
    
    bool mbasis_elem::q_same_state_all_site(const std::vector<basis_prop> &props, const uint32_t &orbital) const
    {
        uint32_t total_sites = props[orbital].num_sites;
        if (total_sites <= 1) return true;
        auto val0 = siteRead(props, 0, orbital);
        for (uint32_t j = 1; j < total_sites; j++) {
            auto val = siteRead(props, j, orbital);
            if (val != val0) return false;
        }
        return true;
    }
    
    bool mbasis_elem::q_same_state_all_site(const std::vector<basis_prop> &props) const
    {
        for (uint32_t orb = 0; orb < props.size(); orb++) {
            if (! q_same_state_all_site(props, orb)) return false;
        }
        return true;
    }
    
    uint64_t mbasis_elem::label(const std::vector<basis_prop> &props, const uint32_t &orbital) const
    {
        auto dim_local = props[orbital].dim_local;
        auto num_sites = props[orbital].num_sites;
        uint64_t res = 0;
        
        if (int_pow<uint8_t,uint64_t>(2, props[orbital].bits_per_site) == static_cast<uint64_t>(dim_local)) {     // no waste bit
            uint16_t byte_pos_bgn = 2;
            for (uint32_t orb = 0; orb < orbital; orb++) byte_pos_bgn += props[orb].num_bytes;
            uint16_t byte_pos_end = byte_pos_bgn + props[orbital].num_bytes;
            for (uint16_t byte_pos = byte_pos_end - 1; byte_pos > byte_pos_bgn; byte_pos--)
                res = (res + mbits[byte_pos]) * 256;
            res += mbits[byte_pos_bgn];
        } else {
            std::vector<uint8_t> nums;
            std::vector<uint8_t> base(num_sites,dim_local);
            for (decltype(num_sites) site = 0; site < num_sites; site++)
                nums.push_back(siteRead(props, site, orbital));
            res = dynamic_base<uint8_t, uint64_t>(nums, base);
        }
        return res;
    }
    
    uint64_t mbasis_elem::label(const std::vector<basis_prop> &props) const
    {
        if (props.size() == 1) {
            return label(props, 0);
        } else {
            std::vector<uint64_t> base, nums;
            for (uint32_t orb = 0; orb < props.size(); orb++) {
                nums.push_back(label(props,orb));
                base.push_back(int_pow<uint32_t, uint64_t>(static_cast<uint32_t>(props[orb].dim_local), props[orb].num_sites));
            }
            return dynamic_base<uint64_t, uint64_t>(nums, base);
        }
    }
    
    std::vector<uint32_t> mbasis_elem::statistics(const std::vector<basis_prop> &props, const uint32_t &orbital) const
    {
        std::vector<uint32_t> results(props[orbital].dim_local,0);
        for (uint32_t site = 0; site < props[orbital].num_sites; site++)
            results[siteRead(props, site, orbital)]++;
        return results;
    }
    
    std::vector<uint32_t> mbasis_elem::statistics(const std::vector<basis_prop> &props) const
    {
        // relax this condition in future
        for (uint32_t orb = 0; orb < props.size() - 1; orb++) assert(props[orb].num_sites == props[orb+1].num_sites);
        
        uint32_t local_dimension = 1;
        for (decltype(props.size()) j = 0; j < props.size(); j++) local_dimension *= props[j].dim_local;
        uint32_t total_orbitals = props.size();
        uint32_t total_sites    = props[0].num_sites;
        
        std::vector<uint32_t> results(local_dimension,0);
        std::vector<uint8_t> state(total_orbitals);
        std::vector<uint8_t> base(total_orbitals);
        for (uint32_t orb = 0; orb < total_orbitals; orb++) base[orb] = props[orb].dim_local;
        for (uint32_t site = 0; site < total_sites; site++) {
            for (uint32_t orb = 0; orb < total_orbitals; orb++) state[orb] = siteRead(props, site, orb);
            results[dynamic_base<uint8_t,uint32_t>(state, base)]++;
        }
        return results;
    }
    
    void mbasis_elem::prt_bits(const std::vector<basis_prop> &props) const
    {
#ifdef DEBUG
        printf("&mbits = %p\n",static_cast<void*>(mbits));
#endif
        uint16_t byte_pos_bgn = 2;
        for (decltype(props.size()) orb = 0; orb < props.size(); orb++) {
            std::cout << "orb " << static_cast<unsigned>(orb) << "(ignore "
                      << static_cast<unsigned>(props[orb].bits_ignore) << " bits): ";
            uint16_t byte_pos_end = byte_pos_bgn + props[orb].num_bytes;
            for (auto byte_pos = byte_pos_end - 1; byte_pos >= byte_pos_bgn; byte_pos--) {
                std::cout << std::bitset<8>(mbits[byte_pos]) << ",";
            }
            std::cout << std::endl;
            byte_pos_bgn = byte_pos_end;
        }
    }
    
    void mbasis_elem::prt_states(const std::vector<basis_prop> &props) const
    {
        for (decltype(props.size()) orb = 0; orb < props.size(); orb++) {
            std::cout << "--- orb " << static_cast<unsigned>(orb) << " ---" << std::endl;
            auto total_sites = props[orb].num_sites;
            for (decltype(total_sites) j = 0; j < total_sites; j++) {
                auto st = siteRead(props, j, orb);
                if (st != 0) {
                    std::cout << "site " << static_cast<unsigned>(j) << ", state " << static_cast<unsigned>(st) << std::endl;
                }
            }
        }
    }
    
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // in future, replace with quick sort to improve performance!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    mbasis_elem &mbasis_elem::transform(const std::vector<basis_prop> &props,
                                        const std::vector<uint32_t> &plan, int &sgn,
                                        const uint32_t &orbital) {
        uint32_t total_sites = props[orbital].num_sites;
        assert(plan.size() == total_sites);
        if (! props[orbital].q_fermion()) {
            sgn = 0;
        } else {
            std::vector<uint32_t> plan_fermion;
            for (uint32_t site = 0; site < total_sites; site++) {
                uint8_t state = siteRead(props, site, orbital);
                // keeps all the sites which have fermion
                if (props[orbital].Nfermion_map[state] % 2 != 0) plan_fermion.push_back(plan[site]);
            }
            // using bubble sort to count how many times we are exchanging fermions
            sgn = bubble_sort(plan_fermion, 0, static_cast<int>(plan_fermion.size())) % 2;
        }
        // store the values
        std::vector<uint8_t> vals(plan.size());
        for (decltype(total_sites) site = 0; site < total_sites; site++) {
            vals[site] = siteRead(props, site, orbital);
        }
        // write the values
        for (decltype(total_sites) site = 0; site < total_sites; site++) {
            siteWrite(props, plan[site], orbital, vals[site]);
        }
        return *this;
    }
    
    mbasis_elem &mbasis_elem::transform(const std::vector<basis_prop> &props,
                                        const std::vector<uint32_t> &plan, int &sgn) {
        sgn = 0;
        for (uint32_t orb = 0; orb < props.size(); orb++) {
            int sgn0;
            transform(props, plan, sgn0, orb);
            sgn = (sgn + sgn0) % 2;
        }
        return *this;
    }
    
    mbasis_elem &mbasis_elem::transform(const std::vector<basis_prop> &props,
                                        const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> &plan, int &sgn) {
        sgn = 0;
        uint32_t total_orbitals = props.size();
        uint32_t total_sites    = props[0].num_sites;
        // implement fermionic transformation later
        assert(std::none_of(props.begin(), props.end(), [](basis_prop x){ return x.q_fermion(); }));
        assert(plan.size() == props.size());
        assert(static_cast<uint32_t>(plan[0].size()) == total_sites);
        
        // store the values
        std::vector<std::vector<uint8_t>> vals(total_orbitals,std::vector<uint8_t>(total_sites));
        for (uint32_t orb = 0; orb < total_orbitals; orb++) {
            for (uint32_t site = 0; site < total_sites; site++)
                vals[orb][site] = siteRead(props, site, orb);
        }
        // write the values
        for (uint32_t orb1 = 0; orb1 < total_orbitals; orb1++) {
            for (uint32_t site1 = 0; site1 < total_sites; site1++) {
                auto site2 = plan[orb1][site1].first;
                auto orb2  = plan[orb1][site1].second;
                assert(props[orb1].dim_local == props[orb2].dim_local);
                siteWrite(props, site2, orb2, vals[orb1][site1]);
            }
        }
        return *this;
    }
    
    mbasis_elem &mbasis_elem::translate(const std::vector<basis_prop> &props,
                                        const qbasis::lattice &latt, const std::vector<int> &disp, int &sgn,
                                        const uint32_t &orbital) {
        auto plan = latt.translation_plan(disp);
        transform(props, plan, sgn, orbital);
        return *this;
    }
    
    mbasis_elem &mbasis_elem::translate(const std::vector<basis_prop> &props,
                                        const qbasis::lattice &latt, const std::vector<int> &disp, int &sgn) {
        auto plan = latt.translation_plan(disp);
        transform(props, plan, sgn);
        return *this;
    }
    
    // need re-write this function to more general cases (pbc mixing obc in different dimensions)
    mbasis_elem &mbasis_elem::translate_to_unique_state(const std::vector<basis_prop> &props,
                                                        const qbasis::lattice &latt, std::vector<int> &disp_vec) {
        for (decltype(latt.dimension()) j = 1; j < latt.dimension(); j++) {
            assert(latt.boundary()[j-1] == latt.boundary()[j]); // relax in future
        }
        uint32_t total_sites    = props[0].num_sites;
        uint32_t total_orbitals = props.size();
        assert(latt.total_sites() == total_sites);
        uint32_t orb_smart = 0;
        while (orb_smart < total_orbitals && q_same_state_all_site(props, orb_smart)) orb_smart++;
        if (orb_smart == total_orbitals) { // if every site in the same state
            disp_vec = std::vector<int>(latt.dimension(),0.0);
            return *this;
        }
        
        auto statis = statistics(props, orb_smart);
        uint8_t state_smart = 0;
        
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
        assert(state_smart < props[orb_smart].dim_local);
        
        // following nomenclature of trans_equiv()
        uint32_t num_sites_smart = statis[state_smart];
        std::vector<uint32_t> sites_smart;
        for (uint32_t site = 0; site < total_sites; site++) {                // search for sites_smart
            if (siteRead(props, site, orb_smart) == state_smart) sites_smart.push_back(site);
            if (static_cast<uint32_t>(sites_smart.size()) >= num_sites_smart) break;
        }
        
        if (latt.boundary() == pbc || latt.boundary() == PBC) {
            // now we want to translate site_smart to the highest site, to minimize the state in < comparison
            std::vector<int> coor_smart(latt.dimension());
            int sub_smart;
            auto state_min = *this;
            auto linear_size = latt.Linear_size();
            std::vector<int> disp(latt.dimension(),0);
            disp_vec = disp;
            for (uint32_t cnt = 0; cnt < num_sites_smart; cnt++) {
                latt.site2coor(coor_smart, sub_smart, sites_smart[cnt]);
                for (uint32_t j = 0; j < latt.dimension(); j++)
                    disp[j] = static_cast<int>(linear_size[j]) - 1 - coor_smart[j];
                auto state_new = *this;
                int sgn;
                state_new.translate(props, latt, disp, sgn);
                if(state_new < state_min) {
                    state_min = state_new;
                    disp_vec = disp;
                }
            }
            swap(state_min, *this);
        } else if (latt.boundary() == obc || latt.boundary() == OBC) {
            std::vector<int> lowest_coors(latt.dimension());
            int sub0;
            latt.site2coor(lowest_coors, sub0, sites_smart[0]);
            //this->prt(); std::cout << std::endl;
            std::vector<int> highest_coors = lowest_coors;
            
            for (uint32_t site = 0; site < latt.total_sites(); site++) {
                std::vector<int> coor(latt.dimension());
                int sub;
                latt.site2coor(coor, sub, site);
                std::vector<uint8_t> temp(total_orbitals);
                for (uint32_t orb = 0; orb < total_orbitals; orb++) temp[orb] = siteRead(props, site, orb);
                if (std::any_of(temp.begin(), temp.end(), [](uint8_t a){return a != 0; })) {
                    for (uint32_t dim = 0; dim < latt.dimension(); dim++) {
                        if (coor[dim] < lowest_coors[dim]) lowest_coors[dim] = coor[dim];
                        if (coor[dim] > highest_coors[dim]) highest_coors[dim] = coor[dim];
                    }
                }
            }
            
            disp_vec.resize(latt.dimension());
            int sgn;
            for (uint32_t dim = 0; dim < latt.dimension(); dim++) {
                assert(lowest_coors[dim] >= 0 && lowest_coors[dim] < static_cast<int>(latt.Linear_size()[dim]));
                assert(highest_coors[dim] >= 0 && highest_coors[dim] < static_cast<int>(latt.Linear_size()[dim]));
                assert(lowest_coors[dim] <= highest_coors[dim]);
                disp_vec[dim] = static_cast<int>((latt.Linear_size()[dim]) - 1 - highest_coors[dim] - lowest_coors[dim])/2;
                if (lowest_coors[dim] + disp_vec[dim]
                    > static_cast<int>(latt.Linear_size()[dim]) - 1 - (highest_coors[dim] + disp_vec[dim])) {
                    disp_vec[dim]--;
                }
            }
            this->translate(props, latt, disp_vec, sgn);
        }
        return *this;
    }
    
    double mbasis_elem::diagonal_operator(const std::vector<basis_prop> &props, const opr<double>& lhs) const
    {
        assert(lhs.q_diagonal() && (! lhs.fermion) && lhs.dim == props[lhs.orbital].dim_local);
        if (lhs.q_zero()) {
            return 0.0;
        } else {
            return lhs.mat[siteRead(props, lhs.site, lhs.orbital)];
        }
    }
    
    std::complex<double> mbasis_elem::diagonal_operator(const std::vector<basis_prop> &props, const opr<std::complex<double>>& lhs) const
    {
        assert(lhs.q_diagonal() && (! lhs.fermion) && lhs.dim == props[lhs.orbital].dim_local);
        if (lhs.q_zero()) {
            return std::complex<double>(0.0, 0.0);
        } else {
            return lhs.mat[siteRead(props, lhs.site, lhs.orbital)];
        }
    }
    
    double mbasis_elem::diagonal_operator(const std::vector<basis_prop> &props, const opr_prod<double>& lhs) const
    {
        if (lhs.q_zero()) {
            return 0.0;
        } else {
            double res = lhs.coeff;
            for (const auto &op : lhs.mat_prod) {
                assert(op.q_diagonal() && (! op.fermion) && op.dim == props[op.orbital].dim_local);
                res *= diagonal_operator(props, op);
                if (std::abs(res) < opr_precision) break;
            }
            return res;
        }
    }
    
    std::complex<double> mbasis_elem::diagonal_operator(const std::vector<basis_prop> &props, const opr_prod<std::complex<double>>& lhs) const
    {
        if (lhs.q_zero()) {
            return std::complex<double>(0.0, 0.0);
        } else {
            std::complex<double> res = lhs.coeff;
            for (const auto &op : lhs.mat_prod) {
                assert(op.q_diagonal() && (! op.fermion) && op.dim == props[op.orbital].dim_local);
                res *= diagonal_operator(props, op);
                if (std::abs(res) < opr_precision) break;
            }
            return res;
        }
    }
    
    double mbasis_elem::diagonal_operator(const std::vector<basis_prop> &props, const mopr<double> &lhs) const
    {
        if (lhs.q_zero()) {
            return 0.0;
        } else {
            double res = 0.0;
            for (decltype(lhs.size()) j = 0; j < lhs.size(); j++) {
                auto op = lhs[j];
                res += diagonal_operator(props, op);
            }
            return res;
        }
    }
    
    std::complex<double> mbasis_elem::diagonal_operator(const std::vector<basis_prop> &props, const mopr<std::complex<double>> &lhs) const
    {
        if (lhs.q_zero()) {
            return 0.0;
        } else {
            std::complex<double> res = 0.0;
            for (decltype(lhs.size()) j = 0; j < lhs.size(); j++) {
                auto op = lhs[j];
                res += diagonal_operator(props, op);
            }
            return res;
        }
    }
    

    // ------------------ friends ------------------
    
    void swap(mbasis_elem &lhs, mbasis_elem &rhs)
    {
        using std::swap;
        swap(lhs.mbits, rhs.mbits);
    }
    
    bool operator<(const mbasis_elem &lhs, const mbasis_elem &rhs)
    {
        assert(lhs.mbits[0] == rhs.mbits[0] && lhs.mbits[1] == rhs.mbits[1]);
        uint16_t total_bytes = static_cast<uint16_t>(lhs.mbits[0] * 256) + static_cast<uint16_t>(lhs.mbits[1]);
        for (uint16_t byte_pos = total_bytes - 1; byte_pos > 2; byte_pos--) {
            if (lhs.mbits[byte_pos] < rhs.mbits[byte_pos]) {
                return true;
            } else if (rhs.mbits[byte_pos] < lhs.mbits[byte_pos]) {
                return false;
            }
        }
        return (lhs.mbits[2] < rhs.mbits[2]);
    }
    
    bool operator==(const mbasis_elem &lhs, const mbasis_elem &rhs)
    {
        uint16_t total_bytes = static_cast<uint16_t>(lhs.mbits[0] * 256) + static_cast<uint16_t>(lhs.mbits[1]);
        assert(lhs.mbits[0] == rhs.mbits[0] && lhs.mbits[1] == rhs.mbits[1]);
        for (uint16_t byte_pos = 2; byte_pos < total_bytes; byte_pos++) {
            if (lhs.mbits[byte_pos] != rhs.mbits[byte_pos]) return false;
        }
        return true;
    }
    
    bool operator!=(const mbasis_elem &lhs, const mbasis_elem &rhs)
    {
        return (! (lhs == rhs));
    }
    
    bool trans_equiv(const mbasis_elem &lhs, const mbasis_elem &rhs,
                     const std::vector<basis_prop> &props, const lattice &latt)
    {
        for (decltype(latt.dimension()) j = 1; j < latt.dimension(); j++) {
            assert(latt.boundary()[j-1] == latt.boundary()[j]); // relax in future
        }
        uint32_t total_sites    = props[0].num_sites;
        uint32_t total_orbitals = props.size();
        assert(latt.total_sites() == total_sites);
        auto statis  = lhs.statistics(props);
        auto statis2 = rhs.statistics(props);
        if (statis != statis2) return false;
        std::vector<std::pair<uint32_t,uint32_t>> statis_temp(statis.size());
        for (uint32_t j = 0; j < statis_temp.size(); j++) {
            statis_temp[j].first = j;
            statis_temp[j].second = statis[j];
            //std::cout << "stat [" << j << "] = " << statis[j] << std::endl;
        }
        std::sort(statis_temp.begin(), statis_temp.end(),
                  [](const std::pair<uint32_t,uint32_t> &a, const std::pair<uint32_t,uint32_t> &b){ return a.second < b.second; });
        uint32_t min_idx = 0;
        while (statis_temp[min_idx].second < 1) min_idx++;
        assert(min_idx < static_cast<uint32_t>(statis_temp.size()));
        if (min_idx == static_cast<uint32_t>(statis_temp.size()) - 1) return true;     // if every site in the same state
        min_idx = statis_temp[min_idx].first;
        uint32_t num_sites_smart = statis[min_idx];
        
        std::vector<uint32_t> base(total_orbitals);
        for (uint32_t orb = 0; orb < static_cast<uint32_t>(base.size()); orb++) base[orb] = props[orb].dim_local;
        auto state_smart = dynamic_base(min_idx, base);
        
        uint32_t site_lhs_smart;
        std::vector<uint32_t> sites_rhs_smart(statis[min_idx]);
        
        // nomenclature:
        // state_smart: an array containing the states in each orbital, and we want to locate which sites have this state
        // num_sites_smart: a # representing how many sites have this state
        // site_lhs_smart:  (any) one site from lhs, which has state_smart
        // sites_rhs_smart: all sites from rhs, which have state_smart
        
        for (uint32_t site = 0; site < total_sites; site++) {                // search for site_lhs_smart
            int flag = 1;
            for (uint32_t orb = 0; orb < total_orbitals; orb++) {
                if (lhs.siteRead(props, site, orb) != state_smart[orb]) {
                    flag = 0;
                    break;
                }
            }
            if (flag) {                       // a matching site found
                site_lhs_smart = site;
                break;
            }
        }
        uint32_t cnt = 0;
        for (uint32_t site = 0; site < total_sites; site++) {                // search for sites_rhs_smart
            int flag = 1;
            for (uint32_t orb = 0; orb < total_orbitals; orb++) {
                if (rhs.siteRead(props, site, orb) != state_smart[orb]) {
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
        std::vector<int> coor0(latt.dimension());
        int sub0;
        latt.site2coor(coor0, sub0, sites_rhs_smart[0]);
        std::vector<std::vector<int>> extremal_coors(latt.dimension(), std::vector<int>(2));
        for (decltype(latt.dimension()) dim = 0; dim < latt.dimension(); dim++) {
            extremal_coors[dim][0] = coor0[dim];
            extremal_coors[dim][1] = coor0[dim];
        }
        if (latt.boundary() == obc || latt.boundary() == OBC) {
            for (uint32_t site = 0; site < total_sites; site++) {
                std::vector<int> coor(latt.dimension());
                int sub;
                latt.site2coor(coor, sub, site);
                std::vector<uint8_t> temp(total_orbitals);
                for (uint32_t orb = 0; orb < total_orbitals; orb++) temp[orb] = rhs.siteRead(props, site, orb);
                if (std::any_of(temp.begin(), temp.end(), [](uint8_t a){return a != 0; })) {
                    for (uint32_t dim = 0; dim < latt.dimension(); dim++) {
                        if (coor[dim] < extremal_coors[dim][0]) extremal_coors[dim][0] = coor[dim];
                        if (coor[dim] > extremal_coors[dim][1]) extremal_coors[dim][1] = coor[dim];
                    }
                }
            }
        }
        for (uint32_t dim = 0; dim < latt.dimension(); dim++) {
            assert(extremal_coors[dim][0] >= 0 && extremal_coors[dim][0] < static_cast<int>(latt.Linear_size()[dim]));
            assert(extremal_coors[dim][1] >= 0 && extremal_coors[dim][1] < static_cast<int>(latt.Linear_size()[dim]));
            assert(extremal_coors[dim][0] <= extremal_coors[dim][1]);
        }
        
        // now we want to translate rhs to compare to lhs
        std::vector<int> coor_lhs_smart(latt.dimension()), coor_rhs_smart(latt.dimension());
        int sub_lhs_smart, sub_rhs_smart;
        latt.site2coor(coor_lhs_smart, sub_lhs_smart, site_lhs_smart);
        for (uint32_t cnt = 0; cnt < num_sites_smart; cnt++) {
            latt.site2coor(coor_rhs_smart, sub_rhs_smart, sites_rhs_smart[cnt]);
            if (sub_lhs_smart != sub_rhs_smart) continue;         // no way to shift to a different sublattice
            std::vector<int> disp(latt.dimension());
            bool flag = false;
            for (uint32_t dim = 0; dim < latt.dimension(); dim++) {
                disp[dim] = coor_lhs_smart[dim] - coor_rhs_smart[dim];
                if (latt.boundary() == obc || latt.boundary() == OBC) {      // should not cross boundary
                    if (disp[dim] > 0 && extremal_coors[dim][1] + disp[dim] >= static_cast<int>(latt.Linear_size()[dim])) {
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
            int sgn;
            rhs_new.translate(props, latt, disp, sgn);
            if (lhs == rhs_new) return true;
        }
        return false;
    }
    
    mbasis_elem zipper_prod(const std::vector<basis_prop> &props_a, const std::vector<basis_prop> &props_b,
                            const std::vector<basis_prop> &props,
                            const mbasis_elem &sub_a, const mbasis_elem &sub_b)
    {
        uint32_t num_orb = props.size();
        assert(props_a.size() == num_orb && props_b.size() == num_orb);
        
        mbasis_elem res(props);
        for (uint32_t orb = 0; orb < num_orb; orb++) {
            uint32_t num_sub_sites = props_a[orb].num_sites;
            assert(props_b[orb].num_sites == num_sub_sites && props[orb].num_sites == 2 * num_sub_sites);
            for (uint32_t site = 0; site < num_sub_sites; site++) {
                res.siteWrite(props, site + site,     orb, sub_a.siteRead(props_a, site, orb)); // from sub_a
                res.siteWrite(props, site + site + 1, orb, sub_b.siteRead(props_b, site, orb)); // from sub_b
            }
        }
        return res;
    }
    
    std::vector<mbasis_elem> enumerate_basis_all(const std::vector<basis_prop> &props)
    {
        uint32_t n_orbs  = props.size();
        uint64_t dim_all = 1;
        std::vector<uint8_t> base;
        for (uint32_t orb = 0; orb < n_orbs; orb++) {
            dim_all *= int_pow<uint32_t, uint64_t>(static_cast<uint32_t>(props[orb].dim_local), props[orb].num_sites);
            for (uint32_t site = 0; site < props[orb].num_sites; site++)
                base.push_back(props[orb].dim_local);
        }
        auto GS = mbasis_elem(props);
        std::vector<mbasis_elem> res(dim_all, GS);
        
        // array to help distributing jobs to different threads
        std::vector<uint64_t> job_array;
        for (uint64_t j = 0; j < dim_all; j+=1000) job_array.push_back(j);
        uint64_t total_chunks = job_array.size();
        job_array.push_back(dim_all);
        
        #pragma omp parallel for schedule(dynamic,1)
        for (uint64_t chunk = 0; chunk < total_chunks; chunk++) {
            // get a new starting basis element
            uint64_t state_num = job_array[chunk];
            auto dist = dynamic_base<uint8_t,uint64_t>(state_num, base);
            uint32_t pos = 0;
            for (uint32_t orb = 0; orb < n_orbs; orb++)
                for (uint32_t site = 0; site < props[orb].num_sites; site++) res[state_num].siteWrite(props, site, orb, dist[pos++]);
            state_num++;
            while (state_num < job_array[chunk + 1]) {
                res[state_num] = res[state_num - 1];
                res[state_num].increment(props);
                state_num++;
            }
        }
        return res;
    }
    
    void classify_trans_full2rep(const std::vector<basis_prop> &props,
                                 const std::vector<mbasis_elem> &basis_all,
                                 const lattice &latt,
                                 const std::vector<bool> &trans_sym,
                                 std::vector<mbasis_elem> &reps,
                                 std::vector<uint64_t> &belong2rep,
                                 std::vector<std::vector<int>> &dist2rep)
    {
        assert(latt.dimension() == trans_sym.size());
        assert(is_sorted_norepeat(basis_all));
        auto bc = latt.boundary();
        auto L = latt.Linear_size();
        uint64_t dim_all = basis_all.size();
        uint64_t dim_all_theoretical = 1;
        for (auto &prop : props)
            dim_all_theoretical *= int_pow<uint32_t, uint64_t>(static_cast<uint32_t>(prop.dim_local), prop.num_sites);
        reps.clear();
        belong2rep.resize(dim_all);
        dist2rep.resize(dim_all);
        uint64_t unreachable = dim_all + 10;
        std::fill(belong2rep.begin(), belong2rep.end(), unreachable);
        
        std::vector<uint32_t> base;                       // for enumerating the possible translations
        for (uint32_t d = 0; d < latt.dimension(); d++) {
            if (trans_sym[d]) {
                assert(bc[d] == "pbc" || bc[d] == "PBC");
                base.push_back(L[d]);
            }
        }
        std::vector<uint32_t> disp(base.size());
        std::vector<int> disp2;
        
        for (uint64_t i = 0; i < dim_all; i++) {
            if (belong2rep[i] != unreachable) continue;          // already fixed
            reps.push_back(basis_all[i]);
            belong2rep[i] = (reps.size() - 1);                   // fix now
            dist2rep[i] = std::vector<int>(latt.dimension(),0);  // fix now
            if (! base.empty()) {
                std::fill(disp.begin(), disp.end(), 0);
                disp = dynamic_base_plus1(disp, base);
                while (! dynamic_base_overflow(disp, base)) {
                    uint32_t pos = 0;
                    disp2.clear();
                    for (uint32_t d = 0; d < latt.dimension(); d++) {
                        if (trans_sym[d]) {
                            disp2.push_back(static_cast<int>(disp[pos++]));
                        } else {
                            disp2.push_back(0);
                        }
                    }
                    auto basis_temp = basis_all[i];
                    int sgn;
                    basis_temp.translate(props, latt, disp2, sgn);
                    uint64_t j = binary_search<mbasis_elem,uint64_t>(basis_all, basis_temp, 0, dim_all);
                    if (j < dim_all) {                          // found
                        if (belong2rep[j] == unreachable) {     // not fixed
                            belong2rep[j] = (reps.size() - 1);  // fix now
                            dist2rep[j] = disp2;                // fix now
                        }
                    } else {
                        assert(dim_all < dim_all_theoretical);
                    }
                    disp = dynamic_base_plus1(disp, base);
                }
            }
        }
    }
    
    void classify_trans_rep2group(const std::vector<basis_prop> &props,
                                  const std::vector<mbasis_elem> &reps,
                                  const lattice &latt,
                                  const std::vector<bool> &trans_sym,
                                  std::vector<std::vector<uint32_t>> &groups,
                                  std::vector<uint32_t> &omega_g,
                                  std::vector<uint32_t> &belong2group)
    {
        uint32_t dim = latt.dimension();
        auto L = latt.Linear_size();
        uint64_t dim_repr = reps.size();
        
        auto div_v1 = latt.divisor_v1(trans_sym);
        groups = latt.divisor_v2(trans_sym);
        uint32_t num_groups = 1;
        for (uint32_t d = 0; d < dim; d++) {
            assert(is_sorted_norepeat(div_v1[d]));
            num_groups *= div_v1[d].size();
        }
        assert(num_groups == groups.size());
        
        belong2group.resize(dim_repr);
        omega_g.resize(dim_repr);
        std::fill(omega_g.begin(), omega_g.end(), 0);
        
        // divisor = x -> translate x to comeback
        // for each representative, find its group, by trying translating according to the smallest possible divisor
        for (uint64_t j = 0; j < dim_repr; j++) {
            std::vector<uint32_t> div(dim,0); // set to 0, only for double checking purpose
            for (uint32_t d = 0; d < dim; d++) {
                for (auto it = div_v1[d].begin(); it != div_v1[d].end(); it++) {
                    std::vector<int> disp(dim,0);
                    disp[d] = static_cast<int>(*it);
                    int sgn;
                    auto basis_temp = reps[j];
                    basis_temp.translate(props, latt, disp, sgn);
                    if (basis_temp == reps[j]) {
                        div[d] = *it;
                        break;
                    }
                }
                assert(div[d] != 0);
            }
            // now div obtained, we can find its group label
            uint32_t g_label = binary_search<std::vector<uint32_t>,uint32_t>(groups, div, 0, num_groups);
            assert(g_label < num_groups);
            belong2group[j] = g_label;
            if (omega_g[g_label] == 0) {
                omega_g[g_label] = 1;
                for (uint32_t d = 0; d < dim; d++) omega_g[g_label] *= div[d];
            }
        }
    }
    
    // ----------------- implementation of wavefunction ------------------
    template <typename T>
    std::pair<mbasis_elem, T> &wavefunction<T>::operator[](uint32_t n)
    {
        assert(n < size());
        assert(! elements.empty());
        auto it = elements.begin();
        for (decltype(n) i = 0; i < n; i++) ++it;
        return *it;
    }
    
    template <typename T>
    const std::pair<mbasis_elem, T> &wavefunction<T>::operator[](uint32_t n) const
    {
        assert(n < size());
        assert(! elements.empty());
        auto it = elements.begin();
        for (decltype(n) i = 0; i < n; i++) ++it;
        return *it;
    }
    
    template <typename T>
    void wavefunction<T>::prt_bits(const std::vector<basis_prop> &props) const
    {
        for (auto &ele : elements) {
            std::cout << "coeff: " << ele.second << std::endl;
            ele.first.prt_bits(props);
            std::cout << std::endl;
        }
    }
    
    template <typename T>
    void wavefunction<T>::prt_states(const std::vector<basis_prop> &props) const
    {
        for (auto &ele : elements) {
            std::cout << "coeff: " << ele.second << std::endl;
            ele.first.prt_states(props);
            std::cout << std::endl;
        }
    }
    
    template <typename T>
    bool wavefunction<T>::q_sorted() const
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
    bool wavefunction<T>::q_sorted_fully() const
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
        if (! this->q_sorted()) {
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
    wavefunction<T> oprXphi(const opr<T> &lhs, const mbasis_elem &rhs, const std::vector<basis_prop> &props)
    {
        wavefunction<T> res;
        auto dim = props[lhs.orbital].dim_local;
        assert(lhs.dim == dim);
        uint32_t col = rhs.siteRead(props, lhs.site, lhs.orbital); // actually col <= 255
        if (lhs.diagonal) {
            assert(! lhs.fermion);
            if (std::abs(lhs.mat[col]) > opr_precision)
                res += std::pair<mbasis_elem, T>(rhs, lhs.mat[col]);
        } else {
            uint32_t displacement = col * lhs.dim;
            bool flag = true;
            for (uint8_t row = 0; row < dim; row++) {
                if (std::abs(lhs.mat[row + displacement]) > opr_precision) {
                    flag = false;
                    break;
                }
            }
            if (flag) return res;                                 // the full column == 0
            int sgn = 0;
            if (lhs.fermion) {                                    // count # of fermions traversed by this operator
                for (uint32_t orb_cnt = 0; orb_cnt < lhs.orbital; orb_cnt++) {
                    if (props[orb_cnt].q_fermion()) {
                        for (uint32_t site_cnt = 0; site_cnt < props[orb_cnt].num_sites; site_cnt++) {
                            sgn = (sgn + props[orb_cnt].Nfermion_map[rhs.siteRead(props, site_cnt, orb_cnt)]) % 2;
                        }
                    }
                }
                assert(props[lhs.orbital].q_fermion());
                for (uint32_t site_cnt = 0; site_cnt < lhs.site; site_cnt++) {
                    sgn = (sgn + props[lhs.orbital].Nfermion_map[rhs.siteRead(props, site_cnt, lhs.orbital)]) % 2;
                }
            }
            for (uint8_t row = 0; row < dim; row++) {
                auto coeff = (sgn == 0 ? lhs.mat[row + displacement] : (-lhs.mat[row + displacement]));
                if (std::abs(coeff) > opr_precision) {
                    mbasis_elem state_new(rhs);
                    state_new.siteWrite(props, lhs.site, lhs.orbital, row);
                    res += std::pair<mbasis_elem, T>(state_new, coeff);
                }
            }
        }
        return res;
    }

    template <typename T>
    wavefunction<T> oprXphi(const opr<T> &lhs, const wavefunction<T> &rhs, const std::vector<basis_prop> &props)
    {
        if (rhs.elements.empty()) return rhs;                                       // zero wavefunction
        wavefunction<T> res;
        for (auto it = rhs.elements.begin(); it != rhs.elements.end(); it++)
            res += (it->second * oprXphi(lhs, it->first, props));
        return res;
    }
    
    template <typename T>
    wavefunction<T> oprXphi(const opr_prod<T> &lhs, const mbasis_elem &rhs, const std::vector<basis_prop> &props)
    {
        if (lhs.q_zero()) return wavefunction<T>();                                 // zero operator
        wavefunction<T> res(rhs);
        for (auto rit = lhs.mat_prod.rbegin(); rit != lhs.mat_prod.rend(); rit++) {
            res = oprXphi((*rit), res, props);
        }
        res *= lhs.coeff;
        return res;
    }
    
    template <typename T>
    wavefunction<T> oprXphi(const opr_prod<T> &lhs, const wavefunction<T> &rhs, const std::vector<basis_prop> &props)
    {
        if (rhs.elements.empty()) return rhs;                                       // zero wavefunction
        if (lhs.q_zero()) return wavefunction<T>();                                 // zero operator
        wavefunction<T> res(rhs);
        for (auto rit = lhs.mat_prod.rbegin(); rit != lhs.mat_prod.rend(); rit++) {
            res = oprXphi((*rit), res, props);
        }
        res *= lhs.coeff;
        return res;
    }

    template <typename T>
    wavefunction<T> oprXphi(const mopr<T> &lhs, const mbasis_elem &rhs, const std::vector<basis_prop> &props)
    {
        if (lhs.q_zero()) return wavefunction<T>();                                // zero operator
        wavefunction<T> res;
        for (auto it = lhs.mats.begin(); it != lhs.mats.end(); it++)
            res += oprXphi((*it), rhs, props);
        return res;
    }
    
    template <typename T>
    wavefunction<T> oprXphi(const mopr<T> &lhs, const wavefunction<T> &rhs, const std::vector<basis_prop> &props)
    {
        if (rhs.elements.empty()) return rhs;                                      // zero wavefunction
        if (lhs.q_zero()) return wavefunction<T>();                                // zero operator
        wavefunction<T> res;
        for (auto it = lhs.mats.begin(); it != lhs.mats.end(); it++)
            res += oprXphi((*it), rhs, props);
        return res;
    }
    
    
    template <typename T> void gen_mbasis_by_mopr(const mopr<T> &Ham, std::list<mbasis_elem> &basis, const std::vector<basis_prop> &props)
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::list<mbasis_elem> basis_new;
        
        #pragma omp parallel for schedule(dynamic,1)
        for (decltype(basis.size()) j = 0; j < basis.size(); j++) {
            std::list<mbasis_elem> temp;
            auto it = basis.begin();
            std::advance(it, j);
            auto phi0 = *it;
            auto states_new = oprXphi(Ham, phi0, props);
            for (decltype(states_new.size()) cnt = 0; cnt < states_new.size(); cnt++) {
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
    
    
    // Explicit instantiation
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
    
    template wavefunction<double> oprXphi(const opr<double>&, const mbasis_elem&, const std::vector<basis_prop>&);
    template wavefunction<std::complex<double>> oprXphi(const opr<std::complex<double>>&, const mbasis_elem&, const std::vector<basis_prop>&);
    
    template wavefunction<double> oprXphi(const opr<double>&, const wavefunction<double>&, const std::vector<basis_prop>&);
    template wavefunction<std::complex<double>> oprXphi(const opr<std::complex<double>>&, const wavefunction<std::complex<double>>&, const std::vector<basis_prop>&);
    
    template wavefunction<double> oprXphi(const opr_prod<double>&, const mbasis_elem&, const std::vector<basis_prop>&);
    template wavefunction<std::complex<double>> oprXphi(const opr_prod<std::complex<double>>&, const mbasis_elem&, const std::vector<basis_prop>&);
    
    template wavefunction<double> oprXphi(const opr_prod<double>&, const wavefunction<double>&, const std::vector<basis_prop>&);
    template wavefunction<std::complex<double>> oprXphi(const opr_prod<std::complex<double>>&, const wavefunction<std::complex<double>>&, const std::vector<basis_prop>&);
    
    template wavefunction<double> oprXphi(const mopr<double>&, const mbasis_elem&, const std::vector<basis_prop>&);
    template wavefunction<std::complex<double>> oprXphi(const mopr<std::complex<double>>&, const mbasis_elem&, const std::vector<basis_prop>&);
    
    template wavefunction<double> oprXphi(const mopr<double>&, const wavefunction<double>&, const std::vector<basis_prop>&);
    template wavefunction<std::complex<double>> oprXphi(const mopr<std::complex<double>>&, const wavefunction<std::complex<double>>&, const std::vector<basis_prop>&);
    
    template void gen_mbasis_by_mopr(const mopr<double>&, std::list<mbasis_elem>&, const std::vector<basis_prop>&);
    template void gen_mbasis_by_mopr(const mopr<std::complex<double>>&, std::list<mbasis_elem>&, const std::vector<basis_prop>&);
    
}
