#include <bitset>
#include <climits>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "qbasis.h"
#include "graph.h"

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
        } else if (s == "spin-3/2") {                 // { |3/2>, |1/2>, |-1/2>, |-3/2> }
            dim_local = 4;
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
    
    void basis_prop::split(basis_prop &sub1, basis_prop &sub2) const
    {
        sub1 = *this;
        if (num_sites % 2 == 0) {
            sub1.num_sites /= 2;
            sub1.num_bytes = static_cast<uint16_t>((sub1.bits_per_site * sub1.num_sites) / 8 + 1);
            sub1.bits_ignore = static_cast<uint8_t>(sub1.num_bytes * 8 - sub1.bits_per_site * sub1.num_sites);
            sub2 = sub1;
        } else {
            sub1.num_sites = (sub1.num_sites + 1) / 2;
            sub1.num_bytes = static_cast<uint16_t>((sub1.bits_per_site * sub1.num_sites) / 8 + 1);
            sub1.bits_ignore = static_cast<uint8_t>(sub1.num_bytes * 8 - sub1.bits_per_site * sub1.num_sites);
            sub2 = *this;
            sub2.num_sites = (sub2.num_sites - 1) / 2;
            sub2.num_bytes = static_cast<uint16_t>((sub2.bits_per_site * sub2.num_sites) / 8 + 1);
            sub2.bits_ignore = static_cast<uint8_t>(sub2.num_bytes * 8 - sub2.bits_per_site * sub2.num_sites);
            assert(sub1.num_sites + sub2.num_sites == num_sites);
        }
    }
    
    void basis_props_split(const std::vector<basis_prop> &parent,
                           std::vector<basis_prop> &sub1,
                           std::vector<basis_prop> &sub2)
    {
        auto num_orbs = parent.size();
        sub1.resize(num_orbs);
        sub2.resize(num_orbs);
        for (decltype(num_orbs) orb = 0; orb < num_orbs; orb++)
            parent[orb].split(sub1[orb], sub2[orb]);
    }
    
    bool q_bosonic(const std::vector<basis_prop> &props)
    {
        for (auto &ele : props) {
            if (ele.q_fermion()) return false;
        }
        return true;
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
            std::memcpy(mbits, old.mbits, total_bytes);
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
    
    bool mbasis_elem::q_zero_site(const std::vector<basis_prop> &props, const uint32_t &site) const
    {
        for (uint32_t orb = 0; orb < props.size(); orb++) {
            if (siteRead(props, site, orb) != 0) return false;
        }
        return true;
    }
    
    bool mbasis_elem::q_zero_orbital(const std::vector<basis_prop> &props, const uint32_t &orbital) const
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
    
    uint64_t mbasis_elem::label(const std::vector<basis_prop> &props, const uint32_t &orbital,
                                std::vector<uint8_t> &work) const
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
            if (work.size() < num_sites + num_sites) work.resize(num_sites + num_sites);
            std::fill(work.begin(), work.end(), dim_local);
            uint8_t* nums = work.data();
            uint8_t* base = nums + num_sites;
            for (decltype(num_sites) site = 0; site < num_sites; site++)
                nums[site] = siteRead(props, site, orbital);
            dynamic_base_vec2num<uint8_t,uint64_t>(static_cast<MKL_INT>(num_sites), base, nums, res);
        }
        return res;
    }
    
    uint64_t mbasis_elem::label(const std::vector<basis_prop> &props,
                                std::vector<uint8_t> &work1, std::vector<uint64_t> &work2) const
    {
        if (props.size() == 1) {
            return label(props, 0, work1);
        } else {
            uint32_t N_orbs = props.size();
            if (work2.size() < N_orbs + N_orbs) work2.resize(N_orbs + N_orbs);
            uint64_t* nums = work2.data();
            uint64_t* base = nums + N_orbs;
            for (uint32_t orb = 0; orb < props.size(); orb++) {
                nums[orb] = label(props,orb, work1);
                base[orb] = int_pow<uint32_t, uint64_t>(static_cast<uint32_t>(props[orb].dim_local), props[orb].num_sites);
            }
            uint64_t res;
            dynamic_base_vec2num<uint64_t,uint64_t>(static_cast<MKL_INT>(N_orbs), base, nums, res);
            return res;
        }
    }
    
    void mbasis_elem::label_sub(const std::vector<basis_prop> &props, const uint32_t &orbital,
                                uint64_t &label1, uint64_t &label2, std::vector<uint8_t> &work) const
    {
        auto dim_local = props[orbital].dim_local;
        uint32_t num_sites = props[orbital].num_sites;
        uint32_t num_sites_sub1 = (num_sites + 1) / 2;
        uint32_t num_sites_sub2 = num_sites - num_sites_sub1;
        
        if (work.size() < num_sites + num_sites) work.resize(num_sites + num_sites);
        std::fill(work.begin(), work.end(), dim_local);
        uint8_t* nums_sub1 = work.data();
        uint8_t* nums_sub2 = nums_sub1 + num_sites_sub1;
        uint8_t* base_sub1 = nums_sub2 + num_sites_sub2;
        uint8_t* base_sub2 = base_sub1 + num_sites_sub1;
        
        for (uint32_t site = 0; site < num_sites_sub2; site++) {
            nums_sub1[site] = siteRead(props, site + site,     orbital);
            nums_sub2[site] = siteRead(props, site + site + 1, orbital);
        }
        if (num_sites_sub1 > num_sites_sub2) nums_sub1[num_sites_sub1 - 1] = siteRead(props, num_sites - 1, orbital);
        dynamic_base_vec2num<uint8_t,uint64_t>(static_cast<MKL_INT>(num_sites_sub1), base_sub1, nums_sub1, label1);
        dynamic_base_vec2num<uint8_t,uint64_t>(static_cast<MKL_INT>(num_sites_sub2), base_sub2, nums_sub2, label2);
    }
    
    void mbasis_elem::label_sub(const std::vector<basis_prop> &props,
                                uint64_t &label1, uint64_t &label2,
                                std::vector<uint8_t> &work1, std::vector<uint64_t> &work2) const
    {
        auto N_orbs = props.size();
        if (N_orbs == 1) {
            label_sub(props, 0, label1, label2, work1);
        } else {
            if (work2.size() < 4*N_orbs) work2.resize(4*N_orbs);
            uint64_t* base_sub1 = work2.data();
            uint64_t* base_sub2 = base_sub1 + N_orbs;
            uint64_t* nums_sub1 = base_sub2 + N_orbs;
            uint64_t* nums_sub2 = nums_sub1 + N_orbs;
            for (uint32_t orb = 0; orb < props.size(); orb++) {
                uint32_t num_sites = props[orb].num_sites;
                uint32_t num_sites_sub1 = (num_sites + 1) / 2;
                uint32_t num_sites_sub2 = num_sites - num_sites_sub1;
                uint32_t local_dim = static_cast<uint32_t>(props[orb].dim_local);
                label_sub(props, orb, nums_sub1[orb], nums_sub2[orb], work1);
                base_sub1[orb] = int_pow<uint32_t, uint64_t>(local_dim, num_sites_sub1);
                base_sub2[orb] = int_pow<uint32_t, uint64_t>(local_dim, num_sites_sub2);
            }
            dynamic_base_vec2num<uint64_t,uint64_t>(static_cast<MKL_INT>(N_orbs), base_sub1, nums_sub1, label1);
            dynamic_base_vec2num<uint64_t,uint64_t>(static_cast<MKL_INT>(N_orbs), base_sub2, nums_sub2, label2);
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
    
    void mbasis_elem::plot_states(const std::vector<basis_prop> &props, const lattice &latt,
                                  const std::string &filename) const
    {
        uint32_t dim = latt.dimension();
        std::ofstream fout(filename, std::ios::out);
        fout << "#(1)\t(2)\t(3)\t(4)\t(5)\t(6)" << std::endl;
        fout << "site\torb\tx\ty\tz\tbasis" << std::endl;
        fout << std::setprecision(10);
        
        std::vector<int> coor;
        int sub;
        std::vector<double> cart;
        for (uint32_t site = 0; site < latt.total_sites(); site++) {
            latt.site2coor(coor, sub, site);
            latt.coor2cart(coor, cart, sub);
            for (uint32_t orb = 0; orb < props.size(); orb++) {
                fout << site << "\t" << orb << "\t";
                for (uint32_t d = 0; d < dim; d++) fout << cart[d] << "\t";
                for (uint32_t d = dim; d < 3; d++) fout << "NA" << "\t";
                fout << static_cast<int>(siteRead(props, site, orb)) << std::endl;
            }
        }
        fout.close();
    }
    
    std::vector<double> mbasis_elem::center_pos(const std::vector<basis_prop> &props, const lattice &latt) const
    {
        uint32_t total_sites    = props[0].num_sites;
        uint32_t total_dim      = latt.dimension();
        assert(latt.total_sites() == total_sites);
        if (q_same_state_all_site(props)) return latt.center_pos();
        
        std::vector<uint32_t> site_list = {};
        for (uint32_t site = 0; site < total_sites; site++) {
            if (! q_zero_site(props, site)) site_list.push_back(site);
        }
        assert(site_list.size() > 0);
        
        std::vector<std::vector<double>> pos_sub = latt.sublattice_pos();
        std::vector<double> center(latt.dimension(), 0.0);
        std::vector<int> coor;
        int sub;
        for (uint32_t &site : site_list) {
            latt.site2coor(coor, sub, site);
            for (uint32_t d = 0; d < total_dim; d++) center[d] += coor[d] + pos_sub[sub][d];
        }
        for (uint32_t d = 0; d < total_dim; d++) center[d] /= static_cast<double>(site_list.size());
        return center;
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
    
    mbasis_elem &mbasis_elem::translate2center_OBC(const std::vector<basis_prop> &props,
                                                   const lattice &latt, std::vector<int> &disp_vec)
    {
        uint32_t total_sites    = props[0].num_sites;
        assert(latt.total_sites() == total_sites);
        if (q_same_state_all_site(props)) return *this;
        
        std::vector<double> center0 = latt.center_pos();                        // center of lattice
        std::vector<double> center1 = center_pos(props, latt);                  // center of basis
        
        disp_vec.resize(latt.dimension());
        for (uint32_t d = 0; d < latt.dimension(); d++) {
            disp_vec[d] = round2int(std::floor(center0[d] - center1[d] + 1e-12));
        }
        
        std::vector<uint32_t> scratch_plan;
        std::vector<int> scratch_coor, scratch_work;
        int sgn;
        latt.translation_plan(scratch_plan, disp_vec, scratch_coor, scratch_work);
        this->transform(props, scratch_plan, sgn);
        
        // check doing once more does not change state, delete later
        uint32_t dim = latt.dimension();
        center1 = center_pos(props, latt);
        for (uint32_t d = 0; d < dim; d++) {
            assert(round2int(std::floor(center0[d] - center1[d] + 1e-12)) == 0);
        }
        
        // check none of the nonvacuum state crosses boundary, delete later
        std::vector<int> scratch_coor2;
        int sub, sub2;
        for (uint32_t site = 0; site < total_sites; site++) {
            if (! q_zero_site(props, scratch_plan[site])) {
                latt.site2coor(scratch_coor, sub, site);
                latt.site2coor(scratch_coor2, sub2, scratch_plan[site]);
                assert(sub == sub2);
                for (uint32_t d = 0; d < dim; d++) {
                    assert(scratch_coor2[d] == scratch_coor[d] + disp_vec[d]);
                }
            }
        }
        
        return *this;
    }
    
    double mbasis_elem::diagonal_operator(const std::vector<basis_prop> &props, const opr<double>& lhs) const
    {
        assert((! lhs.fermion) && lhs.dim == props[lhs.orbital].dim_local);
        return lhs.mat[siteRead(props, lhs.site, lhs.orbital)];
    }
    
    std::complex<double> mbasis_elem::diagonal_operator(const std::vector<basis_prop> &props, const opr<std::complex<double>>& lhs) const
    {
        assert((! lhs.fermion) && lhs.dim == props[lhs.orbital].dim_local);
        return lhs.mat[siteRead(props, lhs.site, lhs.orbital)];
    }
    
    double mbasis_elem::diagonal_operator(const std::vector<basis_prop> &props, const opr_prod<double>& lhs) const
    {
        double res = lhs.coeff;
        for (const auto &op : lhs.mat_prod) {
            assert((! op.fermion) && op.dim == props[op.orbital].dim_local);
            res *= diagonal_operator(props, op);
            if (std::abs(res) < machine_prec) break;
        }
        return res;
    }
    
    std::complex<double> mbasis_elem::diagonal_operator(const std::vector<basis_prop> &props, const opr_prod<std::complex<double>>& lhs) const
    {
        std::complex<double> res = lhs.coeff;
        for (const auto &op : lhs.mat_prod) {
            assert((! op.fermion) && op.dim == props[op.orbital].dim_local);
            res *= diagonal_operator(props, op);
            if (std::abs(res) < machine_prec) break;
        }
        return res;
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
    

    // ------------------ operations on basis -------------------
    
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
        if (lhs.mbits == rhs.mbits) return true;
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
        assert(false);
        for (decltype(latt.dimension()) j = 1; j < latt.dimension(); j++) {
            assert(latt.boundary()[j-1] == latt.boundary()[j]); // relax in future
        }
        std::vector<uint32_t> scratch_plan;
        std::vector<int> scratch_coor, scratch_work;
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
            latt.translation_plan(scratch_plan, disp, scratch_coor, scratch_work);
            rhs_new.transform(props, scratch_plan, sgn);
            if (lhs == rhs_new) return true;
        }
        return false;
    }
    
    void zipper_basis(const std::vector<basis_prop> &props_parent,
                      const std::vector<basis_prop> &props_sub_a,
                      const std::vector<basis_prop> &props_sub_b,
                      const mbasis_elem &sub_a, const mbasis_elem &sub_b, mbasis_elem &parent)
    {
        uint32_t num_orb = props_parent.size();
        
        parent = mbasis_elem(props_parent);
        for (uint32_t orb = 0; orb < num_orb; orb++) {
            uint32_t num_sub_sites_a = props_sub_a[orb].num_sites;
            uint32_t num_sub_sites_b = props_sub_b[orb].num_sites;
            uint32_t num_sites = props_parent[orb].num_sites;
            assert(num_sub_sites_a + num_sub_sites_b == num_sites);
            assert(num_sub_sites_a >= num_sub_sites_b);
            for (uint32_t site = 0; site < num_sub_sites_b; site++) {
                parent.siteWrite(props_parent, site + site,     orb, sub_a.siteRead(props_sub_a, site, orb)); // from sub_a
                parent.siteWrite(props_parent, site + site + 1, orb, sub_b.siteRead(props_sub_b, site, orb)); // from sub_b
            }
            if (num_sub_sites_a > num_sub_sites_b) {
                assert(num_sub_sites_a == num_sub_sites_b + 1);
                parent.siteWrite(props_parent, num_sites - 1, orb, sub_a.siteRead(props_sub_a, num_sub_sites_b, orb)); // from sub_a
            }
        }
    }
    
    void unzipper_basis(const std::vector<basis_prop> &props_parent,
                        const std::vector<basis_prop> &props_sub_a,
                        const std::vector<basis_prop> &props_sub_b,
                        const mbasis_elem &parent,
                        mbasis_elem &sub_a, mbasis_elem &sub_b)
    {
        uint32_t num_orb = props_parent.size();
        
        sub_a = mbasis_elem(props_sub_a);
        sub_b = mbasis_elem(props_sub_b);
        for (uint32_t orb = 0; orb < num_orb; orb++) {
            uint32_t num_sub_sites_a = props_sub_a[orb].num_sites;
            uint32_t num_sub_sites_b = props_sub_b[orb].num_sites;
            uint32_t num_sites = props_parent[orb].num_sites;
            assert(num_sub_sites_a + num_sub_sites_b == num_sites);
            assert(num_sub_sites_a >= num_sub_sites_b);
            for (uint32_t site = 0; site < num_sub_sites_b; site++) {
                sub_a.siteWrite(props_sub_a, site, orb, parent.siteRead(props_parent, site + site,     orb)); // to sub_a
                sub_b.siteWrite(props_sub_b, site, orb, parent.siteRead(props_parent, site + site + 1, orb)); // to sub_b
            }
            if (num_sub_sites_a > num_sub_sites_b) {
                assert(num_sub_sites_a == num_sub_sites_b + 1);
                sub_a.siteWrite(props_sub_a, num_sub_sites_b, orb, parent.siteRead(props_parent, num_sites - 1, orb)); // to sub_a
            }
        }
    }
    
    template <typename T>
    void enumerate_basis(const std::vector<basis_prop> &props,
                         std::vector<qbasis::mbasis_elem> &basis,
                         std::vector<mopr<T>> conserve_lst,
                         std::vector<double> val_lst)
    {
        std::cout << "Enumerating basis with " << val_lst.size() << " conserved quantum numbers..." << std::endl;
        std::cout << "Quantum #s: ";
        for (decltype(val_lst.size()) cnt = 0; cnt < val_lst.size(); cnt++)
            std::cout << val_lst[cnt] << "\t";
        std::cout << std::endl;
        uint32_t n_sites = props[0].num_sites;
        assert(conserve_lst.size() == val_lst.size());
        
        std::list<std::vector<mbasis_elem>> basis_temp;
        auto GS = mbasis_elem(props);
        GS.reset();
        uint32_t n_orbs = props.size();
        uint32_t dim_local = 1;
        std::vector<uint32_t> dim_local_vec;
        for (decltype(props.size()) j = 0; j < props.size(); j++) {
            dim_local *= static_cast<uint32_t>(props[j].dim_local);
            dim_local_vec.push_back(static_cast<uint32_t>(props[j].dim_local));
        }
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        // Hilbert space size if without any symmetry
        MKL_INT dim_total = int_pow<MKL_INT,MKL_INT>(static_cast<MKL_INT>(dim_local), static_cast<MKL_INT>(n_sites));
        assert(dim_total > 0); // prevent overflow
        std::cout << "Nsite     = " << n_sites << std::endl;
        std::cout << "Dim_local = " << dim_local << std::endl;
        std::cout << "Hilbert space size **if** NO symmetry: " << dim_total << std::endl;
        
        // base[]: {  dim_orb0, dim_orb0, ..., dim_orb1, dim_orb1,..., ...  }
        std::vector<MKL_INT> base(n_sites * n_orbs);
        uint32_t pos = 0;
        for (uint32_t orb = 0; orb < n_orbs; orb++)  // low index orbitals considered last in comparison
            for (uint32_t site = 0; site < n_sites; site++) base[pos++] = dim_local_vec[orb];
        
        // array to help distributing jobs to different threads
        std::vector<MKL_INT> job_array;
        for (MKL_INT j = 0; j < dim_total; j+=10000) job_array.push_back(j);
        MKL_INT total_chunks = static_cast<MKL_INT>(job_array.size());
        job_array.push_back(dim_total);
        
        MKL_INT dim_full = 0;
        MKL_INT report = dim_total > 1000000 ? (total_chunks / 10) : total_chunks;
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT chunk = 0; chunk < total_chunks; chunk++) {
            if (chunk > 0 && chunk % report == 0) {
                std::cout << "progress: "
                << (static_cast<double>(chunk) / static_cast<double>(total_chunks) * 100.0) << "%" << std::endl;
            }
            std::vector<qbasis::mbasis_elem> basis_temp_job;
            
            // get a new starting basis element
            MKL_INT state_num = job_array[chunk];
            auto dist = dynamic_base(state_num, base);
            auto state_new = GS;
            MKL_INT pos = 0;
            for (uint32_t orb = 0; orb < n_orbs; orb++) // the order is important
                for (uint32_t site = 0; site < n_sites; site++) state_new.siteWrite(props, site, orb, dist[pos++]);
            
            while (state_num < job_array[chunk+1]) {
                // check if the symmetries are obeyed
                bool flag = true;
                auto it_opr = conserve_lst.begin();
                auto it_val = val_lst.begin();
                while (it_opr != conserve_lst.end()) {
                    auto temp = state_new.diagonal_operator(props, *it_opr);
                    if (std::abs(temp - *it_val) >= 1e-5) {
                        flag = false;
                        break;
                    }
                    it_opr++;
                    it_val++;
                }
                if (flag) basis_temp_job.push_back(state_new);
                state_num++;
                if (state_num < job_array[chunk+1]) state_new.increment(props);
            }
            if (basis_temp_job.size() > 0) {
                #pragma omp critical
                {
                    dim_full += basis_temp_job.size();
                    basis_temp.push_back(std::move(basis_temp_job));     // think how to make sure it is a move operation here
                }
            }
        }
        basis_temp.sort();
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl << std::endl;
        std::cout << "Hilbert space size with symmetry:      " << dim_full << std::endl;
        start = end;
        
        // pick the fruits
        basis.clear();
        std::cout << "memory performance not optimal in the following line, think about improvements." << std::endl;
        basis.reserve(dim_full);
        std::cout << "Moving temporary basis (" << basis_temp.size() << " pieces) to basis_full... ";
        for (auto it = basis_temp.begin(); it != basis_temp.end(); it++) {
            basis.insert(basis.end(), std::make_move_iterator(it->begin()), std::make_move_iterator(it->end()));
            it->erase(it->begin(), it->end()); // should I?
            it->shrink_to_fit();
        }
        assert(dim_full == static_cast<MKL_INT>(basis.size()));
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl << std::endl;
    }
    
    void sort_basis_normal_order(std::vector<qbasis::mbasis_elem> &basis)
    {
        
        bool sorted = true;
        MKL_INT dim = static_cast<MKL_INT>(basis.size());
        assert(dim > 0);
        for (MKL_INT j = 0; j < dim - 1; j++) {
            if (! (basis[j] < basis[j+1])) {
                sorted = false;
                break;
            }
        }
        if (! sorted) {
            std::chrono::time_point<std::chrono::system_clock> start, end;
            start = std::chrono::system_clock::now();
            std::cout << "sorting basis according to '<' comparison";
#ifdef use_gnu_parallel_sort
            std::cout << "(gnu_parallel)... " << std::flush;
            __gnu_parallel::sort(basis.begin(), basis.end());
#else
            std::cout << "(serial)... " << std::flush;
            std::sort(basis.begin(), basis.end());
#endif
            for (MKL_INT j = 0; j < dim - 1; j++) {
                assert(basis[j] < basis[j+1]);
            }
            
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << elapsed_seconds.count() << "s." << std::endl << std::endl;
        }
    }
    
    void sort_basis_Lin_order(const std::vector<basis_prop> &props, std::vector<qbasis::mbasis_elem> &basis)
    {
        std::vector<basis_prop> props_sub_a, props_sub_b;
        basis_props_split(props, props_sub_a, props_sub_b);
        
        bool sorted = true;
        MKL_INT dim = static_cast<MKL_INT>(basis.size());
        assert(dim > 0);
        for (MKL_INT j = 0; j < dim - 1; j++) {
            assert(basis[j] != basis[j+1]);
            mbasis_elem sub_a1, sub_b1, sub_a2, sub_b2;
            unzipper_basis(props, props_sub_a, props_sub_b, basis[j], sub_a1, sub_b1);
            unzipper_basis(props, props_sub_a, props_sub_b, basis[j+1], sub_a2, sub_b2);
            if (sub_b2 < sub_b1 || (sub_b2 == sub_b1 && sub_a2 < sub_a1)) {
                sorted = false;
                break;
            }
        }
        if (! sorted) {
            std::chrono::time_point<std::chrono::system_clock> start, end;
            start = std::chrono::system_clock::now();
            std::cout << "sorting basis according to Lin Table convention";
            auto cmp = [props, props_sub_a, props_sub_b](const mbasis_elem &j1, const mbasis_elem &j2){
                mbasis_elem sub_a1, sub_b1, sub_a2, sub_b2;
                unzipper_basis(props, props_sub_a, props_sub_b, j1, sub_a1, sub_b1);
                unzipper_basis(props, props_sub_a, props_sub_b, j2, sub_a2, sub_b2);
                if (sub_b1 == sub_b2) {
                    return sub_a1 < sub_a2;
                } else {
                    return sub_b1 < sub_b2;
                }};
#ifdef use_gnu_parallel_sort
            std::cout << "(gnu_parallel)... " << std::flush;
            __gnu_parallel::sort(basis.begin(), basis.end(),cmp);
#else
            std::cout << "(serial)... " << std::flush;
            std::sort(basis.begin(), basis.end(),cmp);
#endif
            for (MKL_INT j = 0; j < dim - 1; j++) {
                assert(cmp(basis[j], basis[j+1]));
            }
            
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << elapsed_seconds.count() << "s." << std::endl << std::endl;
        }
    }
    
    
    void fill_Lin_table(const std::vector<basis_prop> &props, const std::vector<qbasis::mbasis_elem> &basis,
                        std::vector<MKL_INT> &Lin_Ja, std::vector<MKL_INT> &Lin_Jb)
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        MKL_INT dim = static_cast<MKL_INT>(basis.size());
        std::vector<basis_prop> props_sub_a, props_sub_b;
        basis_props_split(props, props_sub_a, props_sub_b);
        
        uint32_t Nsites = props[0].num_sites;
        uint32_t Nsites_a = props_sub_a[0].num_sites;
        uint32_t Nsites_b = props_sub_b[0].num_sites;
        assert(Nsites_a >= Nsites_b && Nsites_a + Nsites_b == Nsites);
        
        std::cout << "Filling Lin Table (" << Nsites_a << "+" << Nsites_b <<" sites, dim=" << basis.size() << ")..." << std::endl;
        
        uint32_t local_dim = 1;
        for (decltype(props.size()) j = 0; j < props.size(); j++) local_dim *= props[j].dim_local;
        uint64_t dim_sub_a = int_pow<uint32_t, uint64_t>(local_dim, Nsites_a);
        uint64_t dim_sub_b = int_pow<uint32_t, uint64_t>(local_dim, Nsites_b);
        
        std::cout << "Basis size for sublattices (without any symmetry): " << dim_sub_a << " <-> " << dim_sub_b << std::endl;
        Lin_Ja = std::vector<MKL_INT>(dim_sub_a,-1);
        Lin_Jb = std::vector<MKL_INT>(dim_sub_b,-1);
        
        int num_threads = 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) num_threads = omp_get_num_threads();
        }
        std::vector<std::vector<uint8_t>> scratch_works1(num_threads);
        std::vector<std::vector<uint64_t>> scratch_works2(num_threads);
        
        // first loop over the basis to generate the list (Ia, Ib, J)
        // the element J may not be necessary, remove if not used
        std::cout << "building the (Ia,Ib,J) table...                    " << std::flush;
        std::vector<std::vector<MKL_INT>> table_pre(dim,std::vector<MKL_INT>(3));
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT j = 0; j < dim; j++) {
            int tid = omp_get_thread_num();
            uint64_t i_a, i_b;
            basis[j].label_sub(props, i_a, i_b, scratch_works1[tid], scratch_works2[tid]);
            // value of table_pre[j][2] will be fixed later
            table_pre[j][0] = static_cast<MKL_INT>(i_a);
            table_pre[j][1] = static_cast<MKL_INT>(i_b);
            table_pre[j][2] = j;
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl;
        start = end;
        
        std::cout << "checking if table_pre sorted via I_b               " << std::flush;
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT j = 0; j < dim - 1; j++) {
            if (table_pre[j][1] == table_pre[j+1][1]) {
                assert(table_pre[j][0] < table_pre[j+1][0]);
            } else {
                assert(table_pre[j][1] < table_pre[j+1][1]);
            }
        }
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl;
        start = end;
        
        // build a graph for the connectivity property of (Ia, Ib, J)
        // i.e., for 2 different Js with either same Ia or Ib, they are connected.
        // Also, 2 different Js connected to the same J are connected.
        // Thus, J form a graph, with several pieces disconnected
        ALGraph g(static_cast<uint64_t>(dim));
        std::cout << "Initializing graph vertex info...                  " << std::flush;
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT j = 0; j < dim; j++) {
            g[j].i_a = table_pre[j][0];
            g[j].i_b = table_pre[j][1];
        }
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl;
        start = end;
        
        // loop over the sorted table_pre, connect all horizontal edges
        std::cout << "building horizontal edges...                       " << std::flush;
        for (MKL_INT idx = 1; idx < dim; idx++) {
            assert(table_pre[idx][1] >= table_pre[idx-1][1]);
            if (table_pre[idx][1] == table_pre[idx-1][1]) { // same i_b, connected
                assert(table_pre[idx][0] > table_pre[idx-1][0]);
                g.add_edge(static_cast<uint64_t>(table_pre[idx-1][2]), static_cast<uint64_t>(table_pre[idx][2]));
            }
        }
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl;
        start = end;
        
        // sort table_pre according to ia, first connect all vertical edges
        std::cout << "sorting table_pre according to I_a...              " << std::flush;
        auto cmp = [](const std::vector<MKL_INT> &a, const std::vector<MKL_INT> &b){
            if (a[0] == b[0]) {
                return a[1] < b[1];
            } else {
                return a[0] < b[0];
            }};
#ifdef use_gnu_parallel_sort
        __gnu_parallel::sort(table_pre.begin(), table_pre.end(), cmp);
#else
        std::sort(table_pre.begin(), table_pre.end(), cmp);
#endif
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl;
        start = end;
        // loop over the sorted table_pre
        std::cout << "building vertical edges...                         " << std::flush;
        for (MKL_INT idx = 1; idx < dim; idx++) {
            if (table_pre[idx][0] == table_pre[idx-1][0]) { // same i_a, connected
                g.add_edge(static_cast<uint64_t>(table_pre[idx-1][2]), static_cast<uint64_t>(table_pre[idx][2]));
            }
        }
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl;
        start = end;
        
        //std::cout << "Num_edges = " << g.num_arcs() << std::endl;
        table_pre.clear();
        table_pre.shrink_to_fit();
        
        int fail = g.BSF_set_JaJb(Lin_Ja, Lin_Jb);
        if (fail) {
            // there is always a way to build, but need smarter ordering of the input basis
            std::cout << "Lin Table failed to build!!!" << std::endl;
            Lin_Ja.clear();
            Lin_Ja.shrink_to_fit();
            Lin_Jb.clear();
            Lin_Jb.shrink_to_fit();
        } else {
            // check with the original basis, delete later
            std::cout << "double checking Lin Table validity...              " << std::flush;
            #pragma omp parallel for schedule(dynamic,1)
            for (MKL_INT j = 0; j < dim; j++) {
                int tid = omp_get_thread_num();
                uint64_t i_a, i_b;
                basis[j].label_sub(props, i_a, i_b, scratch_works1[tid], scratch_works2[tid]);
                assert(Lin_Ja[i_a] + Lin_Jb[i_b] == j);
            }
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
            std::cout << elapsed_seconds.count() << "s." << std::endl << std::endl;
        }
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
        std::vector<uint32_t> scratch_plan(latt.total_sites());
        std::vector<int> scratch_coor(latt.dimension()), scratch_work(latt.dimension());
        
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
                    latt.translation_plan(scratch_plan, disp2, scratch_coor, scratch_work);
                    basis_temp.transform(props, scratch_plan, sgn);
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
        assert(is_sorted_norepeat(reps));
    }
    
    void classify_trans_rep2group(const std::vector<basis_prop> &props,
                                  const std::vector<mbasis_elem> &reps,
                                  const lattice &latt,
                                  const std::vector<bool> &trans_sym,
                                  const std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> &groups,
                                  std::vector<uint32_t> &omega_g,
                                  std::vector<uint32_t> &belong2group)
    {
        uint32_t dim = latt.dimension();
        auto L = latt.Linear_size();
        uint64_t dim_repr = reps.size();
        
        uint32_t num_groups = groups.size();
        
        belong2group.resize(dim_repr);
        std::fill(belong2group.begin(), belong2group.end(), num_groups+10);
        omega_g.resize(num_groups);
        for (uint32_t g = 0; g < num_groups; g++) omega_g[g] = groups[g].second;
        
        // for each representative, find its group
        #pragma omp parallel for schedule(dynamic,1)
        for (uint64_t j = 0; j < dim_repr; j++) {
            std::vector<uint32_t> scratch_plan(latt.total_sites());
            std::vector<int> scratch_coor(latt.dimension()), scratch_work(latt.dimension());
            
            // loop over groups, to check which one the repr belongs to
            for (uint32_t g = 0; g < num_groups; g++) {
                bool flag = true;
                for (uint32_t d = 0; d < dim; d++) {
                    if (! trans_sym[d]) continue;
                    std::vector<int> disp(dim);
                    int sgn;
                    for (uint32_t i = 0; i < dim; i++) disp[i] = groups[g].first[d][i];
                    auto basis_temp = reps[j];
                    latt.translation_plan(scratch_plan, disp, scratch_coor, scratch_work);
                    basis_temp.transform(props, scratch_plan, sgn);
                    if (basis_temp != reps[j]) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {                     // belongs to this group
                    belong2group[j] = g;
                    break;
                }
            }
            assert(belong2group[j] < num_groups);
        }
    }
    
    
    void log_Weisse_tables(const lattice &latt_parent,
                           const std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> &groups_parent,
                           const std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> &groups_sub,
                           const MltArray_PairVec &Weisse_e_lt,
                           const MltArray_PairVec &Weisse_e_eq,
                           const MltArray_PairVec &Weisse_e_gt,
                           const MltArray_uint32  &Weisse_w_lt,
                           const MltArray_uint32  &Weisse_w_eq,
                           const MltArray_uint32  &Weisse_w_gt)
    {
        auto latt_sub             = divide_lattice(latt_parent);
        uint32_t latt_sub_dim     = latt_sub.dimension();
        auto base_sub             = latt_sub.Linear_size();
        uint32_t num_groups       = groups_sub.size();
        
        std::ofstream fout("log_Weisse_table.txt", std::ios::out);
        
        fout << "print out e=" << std::endl;
        for (uint32_t ga = 0; ga < num_groups; ga++) {
            for (uint32_t gb = 0; gb < num_groups; gb++) {
                fout << "---------------" << std::endl;
                fout << "ga,    gb    = " << ga << ", " << gb << std::endl;
                std::vector<uint32_t> disp_ja(latt_sub_dim,0);
                while (! dynamic_base_overflow(disp_ja, base_sub)) {
                    std::vector<uint32_t> disp_jb(latt_sub_dim,0);
                    while (! dynamic_base_overflow(disp_jb, base_sub)) {
                        std::vector<uint64_t> pos{ga,gb};
                        pos.insert(pos.end(), disp_ja.begin(), disp_ja.end());
                        pos.insert(pos.end(), disp_jb.begin(), disp_jb.end());
                        auto res = Weisse_e_eq.index(pos);
                        if (res.first  != std::vector<uint32_t>(res.first.size(),999999999) ||
                            res.second != std::vector<uint32_t>(res.second.size(),999999999)) {
                            fout << "ja = ";
                            for (decltype(disp_ja.size()) j = 0; j < disp_ja.size(); j++) {
                                fout << disp_ja[j] << "\t";
                            }
                            fout << std::endl;
                            fout << "jb = ";
                            for (decltype(disp_jb.size()) j = 0; j < disp_jb.size(); j++) {
                                fout << disp_jb[j] << "\t";
                            }
                            fout << std::endl;
                            fout << "i  = ";
                            for (decltype(res.first.size()) j = 0; j < res.first.size(); j++) {
                                fout << res.first[j] << "\t";
                            }
                            fout << std::endl;
                            fout << "j  = ";
                            for (decltype(res.second.size()) j = 0; j < res.second.size(); j++) {
                                fout << res.second[j] << "\t";
                            }
                            fout << std::endl;
                            fout << std::endl;
                        }
                        disp_jb = dynamic_base_plus1(disp_jb, base_sub);
                    }
                    disp_ja = dynamic_base_plus1(disp_ja, base_sub);
                }
            }
        }
        
        fout << "print out e<" << std::endl;
        for (uint32_t ga = 0; ga < num_groups; ga++) {
            for (uint32_t gb = 0; gb < num_groups; gb++) {
                fout << "---------------" << std::endl;
                fout << "ga,    gb    = " << ga << ", " << gb << std::endl;
                std::vector<uint32_t> disp_ja(latt_sub_dim,0);
                while (! dynamic_base_overflow(disp_ja, base_sub)) {
                    std::vector<uint32_t> disp_jb(latt_sub_dim,0);
                    while (! dynamic_base_overflow(disp_jb, base_sub)) {
                        std::vector<uint64_t> pos{ga,gb};
                        pos.insert(pos.end(), disp_ja.begin(), disp_ja.end());
                        pos.insert(pos.end(), disp_jb.begin(), disp_jb.end());
                        auto res = Weisse_e_lt.index(pos);
                        if (res.first  != std::vector<uint32_t>(res.first.size(),999999999) ||
                            res.second != std::vector<uint32_t>(res.second.size(),999999999)) {
                            fout << "ja = ";
                            for (decltype(disp_ja.size()) j = 0; j < disp_ja.size(); j++) {
                                fout << disp_ja[j] << "\t";
                            }
                            fout << std::endl;
                            fout << "jb = ";
                            for (decltype(disp_jb.size()) j = 0; j < disp_jb.size(); j++) {
                                fout << disp_jb[j] << "\t";
                            }
                            fout << std::endl;
                            fout << "i  = ";
                            for (decltype(res.first.size()) j = 0; j < res.first.size(); j++) {
                                fout << res.first[j] << "\t";
                            }
                            fout << std::endl;
                            fout << "j  = ";
                            for (decltype(res.second.size()) j = 0; j < res.second.size(); j++) {
                                fout << res.second[j] << "\t";
                            }
                            fout << std::endl;
                            fout << std::endl;
                        }
                        disp_jb = dynamic_base_plus1(disp_jb, base_sub);
                    }
                    disp_ja = dynamic_base_plus1(disp_ja, base_sub);
                }
            }
        }
        
        fout << "print out e>" << std::endl;
        for (uint32_t ga = 0; ga < num_groups; ga++) {
            for (uint32_t gb = 0; gb < num_groups; gb++) {
                fout << "---------------" << std::endl;
                fout << "ga,    gb    = " << ga << ", " << gb << std::endl;
                std::vector<uint32_t> disp_ja(latt_sub_dim,0);
                while (! dynamic_base_overflow(disp_ja, base_sub)) {
                    std::vector<uint32_t> disp_jb(latt_sub_dim,0);
                    while (! dynamic_base_overflow(disp_jb, base_sub)) {
                        std::vector<uint64_t> pos{ga,gb};
                        pos.insert(pos.end(), disp_ja.begin(), disp_ja.end());
                        pos.insert(pos.end(), disp_jb.begin(), disp_jb.end());
                        auto res = Weisse_e_gt.index(pos);
                        if (res.first  != std::vector<uint32_t>(res.first.size(),999999999) ||
                            res.second != std::vector<uint32_t>(res.second.size(),999999999)) {
                            fout << "ja = ";
                            for (decltype(disp_ja.size()) j = 0; j < disp_ja.size(); j++) {
                                fout << disp_ja[j] << "\t";
                            }
                            fout << std::endl;
                            fout << "jb = ";
                            for (decltype(disp_jb.size()) j = 0; j < disp_jb.size(); j++) {
                                fout << disp_jb[j] << "\t";
                            }
                            fout << std::endl;
                            fout << "i  = ";
                            for (decltype(res.first.size()) j = 0; j < res.first.size(); j++) {
                                fout << res.first[j] << "\t";
                            }
                            fout << std::endl;
                            fout << "j  = ";
                            for (decltype(res.second.size()) j = 0; j < res.second.size(); j++) {
                                fout << res.second[j] << "\t";
                            }
                            fout << std::endl;
                            fout << std::endl;
                        }
                        disp_jb = dynamic_base_plus1(disp_jb, base_sub);
                    }
                    disp_ja = dynamic_base_plus1(disp_ja, base_sub);
                }
            }
        }
        
        fout << "print out w<" << std::endl;
        for (uint32_t ga = 0; ga < num_groups; ga++) {
            for (uint32_t gb = 0; gb < num_groups; gb++) {
                fout << "---------------" << std::endl;
                fout << "ga,    gb    = " << ga << ", " << gb << std::endl;
                std::vector<uint32_t> disp_j(latt_sub_dim,0);
                while (! dynamic_base_overflow(disp_j, base_sub)) {
                    std::vector<uint64_t> pos_w{ga,gb};
                    pos_w.insert(pos_w.end(), disp_j.begin(), disp_j.end());
                    auto g = Weisse_w_lt.index(pos_w);
                    if (g < groups_parent.size()) {
                        fout << "j = ";
                        for (decltype(disp_j.size()) j = 0; j < disp_j.size(); j++) {
                            fout << disp_j[j] << "\t";
                        }
                        fout << std::endl;
                        fout << "g(parent) = " << g << std::endl;
                        fout << "omega = " << groups_parent[g].second << std::endl;
                        fout << std::endl;
                    }
                    disp_j = dynamic_base_plus1(disp_j, base_sub);
                }
            }
        }
        
        fout << "print out w=" << std::endl;
        for (uint32_t ga = 0; ga < num_groups; ga++) {
            for (uint32_t gb = 0; gb < num_groups; gb++) {
                fout << "---------------" << std::endl;
                fout << "ga,    gb    = " << ga << ", " << gb << std::endl;
                std::vector<uint32_t> disp_j(latt_sub_dim,0);
                while (! dynamic_base_overflow(disp_j, base_sub)) {
                    std::vector<uint64_t> pos_w{ga,gb};
                    pos_w.insert(pos_w.end(), disp_j.begin(), disp_j.end());
                    auto g = Weisse_w_eq.index(pos_w);
                    if (g < groups_parent.size()) {
                        fout << "j = ";
                        for (decltype(disp_j.size()) j = 0; j < disp_j.size(); j++) {
                            fout << disp_j[j] << "\t";
                        }
                        fout << std::endl;
                        fout << "g(parent) = " << g << std::endl;
                        fout << "omega = " << groups_parent[g].second << std::endl;
                        fout << std::endl;
                    }
                    disp_j = dynamic_base_plus1(disp_j, base_sub);
                }
            }
        }
        
        if (Weisse_w_gt.size() == Weisse_w_lt.size()) {
            fout << "print out w>" << std::endl;
            for (uint32_t ga = 0; ga < num_groups; ga++) {
                for (uint32_t gb = 0; gb < num_groups; gb++) {
                    fout << "---------------" << std::endl;
                    fout << "ga,    gb    = " << ga << ", " << gb << std::endl;
                    std::vector<uint32_t> disp_j(latt_sub_dim,0);
                    while (! dynamic_base_overflow(disp_j, base_sub)) {
                        std::vector<uint64_t> pos_w{ga,gb};
                        pos_w.insert(pos_w.end(), disp_j.begin(), disp_j.end());
                        auto g = Weisse_w_gt.index(pos_w);
                        if (g < groups_parent.size()) {
                            fout << "j = ";
                            for (decltype(disp_j.size()) j = 0; j < disp_j.size(); j++) {
                                fout << disp_j[j] << "\t";
                            }
                            fout << std::endl;
                            fout << "g(parent) = " << g << std::endl;
                            fout << "omega = " << groups_parent[g].second << std::endl;
                            fout << std::endl;
                        }
                        disp_j = dynamic_base_plus1(disp_j, base_sub);
                    }
                }
            }
        }
        
        fout.close();
        
    }
    
    void classify_Weisse_tables(const std::vector<basis_prop> &props_parent,
                                const std::vector<basis_prop> &props_sub,
                                const std::vector<mbasis_elem> &basis_sub_repr,
                                const lattice &latt_parent,
                                const std::vector<bool> &trans_sym,
                                const std::vector<uint64_t> &belong2rep,
                                const std::vector<std::vector<int>> &dist2rep,
                                const std::vector<uint32_t> &belong2group,
                                const std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> &groups_parent,
                                const std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> &groups_sub,
                                MltArray_PairVec &Weisse_e_lt,
                                MltArray_PairVec &Weisse_e_eq,
                                MltArray_PairVec &Weisse_e_gt,
                                MltArray_uint32  &Weisse_w_lt,
                                MltArray_uint32  &Weisse_w_eq,
                                MltArray_uint32  &Weisse_w_gt)
    {
        assert(latt_parent.total_sites() % 2 == 0);
        auto latt_sub             = divide_lattice(latt_parent);
        uint64_t dim_repr         = basis_sub_repr.size();
        uint32_t num_groups       = groups_sub.size();
        uint32_t latt_sub_dim     = latt_sub.dimension();
        auto latt_sub_linear_size = latt_sub.Linear_size();
        auto base_parent          = latt_parent.Linear_size();
        auto base_sub             = latt_sub.Linear_size();
        bool flag_trans           = false;                                      // if translation symmetry exists
        for (uint32_t j = 0; j < latt_sub_dim; j++) {
            if (! trans_sym[j]) {
                base_parent[j] = 1;
                base_sub[j]    = 1;
            } else {
                flag_trans = true;
            }
        }
        assert(flag_trans);
        // check if the partitioned dimension involved in translation (if not, then the counting of repr has to be doubled)
        uint32_t dim_spec = latt_parent.dimension_spec();
        bool dim_spec_involved;
        if (dim_spec == latt_sub_dim) {
            assert(latt_parent.num_sublattice() % 2 == 0);
            dim_spec_involved = false;
        } else {
            dim_spec_involved = trans_sym[dim_spec];
        }
        std::vector<uint32_t> plan_parent(latt_parent.total_sites());
        std::vector<uint32_t> plan_sub(latt_sub.total_sites());
        std::vector<int> scratch_coor(latt_parent.dimension()), scratch_work(latt_parent.dimension());
        std::vector<uint8_t> scratch_work1;
        std::vector<uint64_t> scratch_work2;
        
        // gather a list of examples for different groups
        std::vector<std::vector<mbasis_elem>> examples(num_groups);
        for (uint64_t j = 0; j < dim_repr; j++) {
            auto group_label = belong2group[j];
            examples[group_label].push_back(basis_sub_repr[j]);
        }
        
        // automatically determines the shape of array_4D
        std::pair<std::vector<uint32_t>,std::vector<uint32_t>> default_value;
        assert(latt_parent.total_sites() < 999999999);
        default_value.first  = std::vector<uint32_t>(latt_sub_dim,999999999);
        default_value.second = default_value.first;
        
        std::vector<uint64_t> linear_size;
        linear_size.push_back(static_cast<uint64_t>(num_groups));
        linear_size.push_back(static_cast<uint64_t>(num_groups));
        for (uint32_t j = 0; j < latt_sub_dim; j++) {
            if (trans_sym[j]) {
                linear_size.push_back(static_cast<uint64_t>(latt_sub_linear_size[j]));
            } else {
                linear_size.push_back(1);
            }
        }
        Weisse_w_lt = MltArray_uint32(linear_size, groups_parent.size() + 10);
        Weisse_w_eq = MltArray_uint32(linear_size, groups_parent.size() + 10);
        Weisse_w_gt = MltArray_uint32(linear_size, groups_parent.size() + 10);
        if (dim_spec_involved) Weisse_w_gt = MltArray_uint32();                  // not used in this case
        for (uint32_t j = 0; j < latt_sub_dim; j++) {
            if (trans_sym[j]) {
                linear_size.push_back(static_cast<uint64_t>(latt_sub_linear_size[j]));
            } else {
                linear_size.push_back(1);
            }
        }
        Weisse_e_lt = MltArray_PairVec(linear_size, default_value);
        Weisse_e_eq = MltArray_PairVec(linear_size, default_value);
        Weisse_e_gt = MltArray_PairVec(linear_size, default_value);
        
        
        for (uint32_t ga = 0; ga < num_groups; ga++) {
            // change the assert to: if (size==0) continue
            assert(examples[ga].size() > 0);
            for (uint32_t gb = 0; gb < num_groups; gb++) {
                bool flag_lt, flag_eq, flag_gt;
                if (ga != gb) { // then it is impossible to pick ra == rb, table_eq not available
                    flag_eq = false;
                    if (examples[ga].back() < examples[gb].front()) {
                        flag_gt = false;
                        flag_lt = true;
                    } else if (examples[gb].back() < examples[ga].front()) {
                        flag_gt = true;
                        flag_lt = false;
                    } else {
                        flag_gt = true;
                        flag_lt = true;
                    }
                } else {
                    flag_eq = true;
                    if (examples[ga].size() > 1) {
                        flag_lt = true;
                        flag_gt = true;
                    } else {
                        flag_lt = false;
                        flag_gt = false;
                    }
                }
                // build table e<
                if (flag_lt) {
                    auto &ra = examples[ga].front();
                    auto &rb = examples[gb].back();
                    assert(ra < rb);
                    
                    std::vector<uint32_t> disp_j(latt_sub_dim,0);
                    while (! dynamic_base_overflow(disp_j, base_sub)) {
                        auto rb_new = rb;
                        std::vector<int> disp_j_int(latt_sub_dim);
                        for (uint32_t j = 0; j < latt_sub_dim; j++) disp_j_int[j] = static_cast<int>(disp_j[j]);
                        int sgn;
                        latt_sub.translation_plan(plan_sub, disp_j_int, scratch_coor, scratch_work);
                        rb_new.transform(props_sub, plan_sub, sgn);                               // Tj |rb>
                        auto rb_new_label = rb_new.label(props_sub, scratch_work1, scratch_work2);
                        assert(basis_sub_repr[belong2rep[rb_new_label]] == rb);
                        auto dist_to_rb = dist2rep[rb_new_label];
                        if (dist_to_rb != disp_j_int) {                                           // remove over-countings
                            disp_j = dynamic_base_plus1(disp_j, base_sub);
                            continue;
                        }
                        mbasis_elem ra_z_Tj_rb;
                        zipper_basis(props_parent, props_sub, props_sub, ra, rb_new, ra_z_Tj_rb); // |ra> z Tj |rb>
                        
                        std::vector<uint32_t> disp_i(latt_sub_dim,0);
                        while (! dynamic_base_overflow(disp_i, base_parent)) {
                            std::vector<int> disp_i_int(latt_sub_dim);
                            for (uint32_t j = 0; j < latt_sub_dim; j++) disp_i_int[j] = static_cast<int>(disp_i[j]);
                            auto Ti_ra_z_Tj_rb = ra_z_Tj_rb;
                            latt_parent.translation_plan(plan_parent, disp_i_int, scratch_coor, scratch_work);
                            Ti_ra_z_Tj_rb.transform(props_parent, plan_parent, sgn);                  // Ti (|ra> z Tj |rb>)
                            // now need find ja, jb
                            uint64_t state_sub1_label, state_sub2_label;
                            Ti_ra_z_Tj_rb.label_sub(props_parent, state_sub1_label, state_sub2_label,
                                                    scratch_work1, scratch_work2);                // |a>, |b>
                            
                            auto state_rep1_label = belong2rep[state_sub1_label];
                            auto state_rep2_label = belong2rep[state_sub2_label];
                            auto &state_rep1      = basis_sub_repr[state_rep1_label];             // |ra'>
                            auto &state_rep2      = basis_sub_repr[state_rep2_label];             // |rb'>
                            auto &dist2rep1       = dist2rep[state_sub1_label];                   // ja'
                            auto &dist2rep2       = dist2rep[state_sub2_label];                   // jb'
                            std::vector<uint64_t> pos{ga, gb};
                            if (state_rep1 < state_rep2) {                                        // ra' < rb' satisfied
                                assert(belong2group[state_rep1_label] == ga);
                                assert(belong2group[state_rep2_label] == gb);
                                assert(state_rep1 == ra && state_rep2 == rb);
                                pos.insert(pos.end(), dist2rep1.begin(), dist2rep1.end());        // ja'
                                pos.insert(pos.end(), dist2rep2.begin(), dist2rep2.end());        // jb'
                                assert(pos.size() == linear_size.size());
                                if ( disp_j < Weisse_e_lt.index(pos).second ||
                                    (disp_j == Weisse_e_lt.index(pos).second && disp_i < Weisse_e_lt.index(pos).first)) {
                                    Weisse_e_lt.index(pos).first  = disp_i;
                                    Weisse_e_lt.index(pos).second = disp_j;
                                }
                            }
                            disp_i = dynamic_base_plus1(disp_i, base_parent);
                        }
                        disp_j = dynamic_base_plus1(disp_j, base_sub);
                    }
                }
                // build table e>
                if (flag_gt) {
                    auto &ra = dim_spec_involved ? examples[gb].front() : examples[ga].back();
                    auto &rb = dim_spec_involved ? examples[ga].back() : examples[gb].front();
                    assert((dim_spec_involved && ra < rb) || ((! dim_spec_involved) && rb < ra));
                    
                    std::vector<uint32_t> disp_j(latt_sub_dim,0);
                    while (! dynamic_base_overflow(disp_j, base_sub)) {
                        auto rb_new = rb;
                        std::vector<int> disp_j_int(latt_sub_dim);
                        for (uint32_t j = 0; j < latt_sub_dim; j++) disp_j_int[j] = static_cast<int>(disp_j[j]);
                        int sgn;
                        latt_sub.translation_plan(plan_sub, disp_j_int, scratch_coor, scratch_work);
                        rb_new.transform(props_sub, plan_sub, sgn);                               // Tj |rb>
                        auto rb_new_label = rb_new.label(props_sub, scratch_work1, scratch_work2);
                        assert(basis_sub_repr[belong2rep[rb_new_label]] == rb);
                        auto dist_to_rb = dist2rep[rb_new_label];
                        if (dist_to_rb != disp_j_int) {                                           // remove over-countings
                            disp_j = dynamic_base_plus1(disp_j, base_sub);
                            continue;
                        }
                        mbasis_elem ra_z_Tj_rb;
                        zipper_basis(props_parent, props_sub, props_sub, ra, rb_new, ra_z_Tj_rb); // |ra> z Tj |rb>
                        
                        std::vector<uint32_t> disp_i(latt_sub_dim,0);
                        while (! dynamic_base_overflow(disp_i, base_parent)) {
                            std::vector<int> disp_i_int(latt_sub_dim);
                            for (uint32_t j = 0; j < latt_sub_dim; j++) disp_i_int[j] = static_cast<int>(disp_i[j]);
                            auto Ti_ra_z_Tj_rb = ra_z_Tj_rb;
                            latt_parent.translation_plan(plan_parent, disp_i_int, scratch_coor, scratch_work);
                            Ti_ra_z_Tj_rb.transform(props_parent, plan_parent, sgn);                  // Ti (|ra> z Tj |rb>)
                            // now need find ja, jb
                            uint64_t state_sub1_label, state_sub2_label;
                            Ti_ra_z_Tj_rb.label_sub(props_parent, state_sub1_label, state_sub2_label,
                                                    scratch_work1, scratch_work2);
                            auto state_rep1_label = belong2rep[state_sub1_label];
                            auto state_rep2_label = belong2rep[state_sub2_label];
                            auto &state_rep1      = basis_sub_repr[state_rep1_label];             // |ra'>
                            auto &state_rep2      = basis_sub_repr[state_rep2_label];             // |rb'>
                            auto &dist2rep1       = dist2rep[state_sub1_label];                   // ja'
                            auto &dist2rep2       = dist2rep[state_sub2_label];                   // jb'
                            std::vector<uint64_t> pos{ga, gb};
                            if (state_rep2 < state_rep1) {                                        // ra' > rb' satisfied
                                assert(belong2group[state_rep1_label] == ga);
                                assert(belong2group[state_rep2_label] == gb);
                                assert(state_rep1 == (dim_spec_involved?rb:ra));
                                assert(state_rep2 == (dim_spec_involved?ra:rb));
                                pos.insert(pos.end(), dist2rep1.begin(), dist2rep1.end());        // ja
                                pos.insert(pos.end(), dist2rep2.begin(), dist2rep2.end());        // jb
                                assert(pos.size() == linear_size.size());
                                if ( disp_j < Weisse_e_gt.index(pos).second ||
                                    (disp_j == Weisse_e_gt.index(pos).second && disp_i < Weisse_e_gt.index(pos).first)) {
                                    Weisse_e_gt.index(pos).first  = disp_i;
                                    Weisse_e_gt.index(pos).second = disp_j;
                                }
                            }
                            disp_i = dynamic_base_plus1(disp_i, base_parent);
                        }
                        disp_j = dynamic_base_plus1(disp_j, base_sub);
                    }
                }
                // build table e=
                if (flag_eq) {
                    auto &ra = examples[ga].front();
                    auto &rb = ra;
                    
                    std::vector<uint32_t> disp_j(latt_sub_dim,0);
                    while (! dynamic_base_overflow(disp_j, base_sub)) {
                        auto rb_new = rb;
                        std::vector<int> disp_j_int(latt_sub_dim);
                        for (uint32_t j = 0; j < latt_sub_dim; j++) disp_j_int[j] = static_cast<int>(disp_j[j]);
                        int sgn;
                        latt_sub.translation_plan(plan_sub, disp_j_int, scratch_coor, scratch_work);
                        rb_new.transform(props_sub, plan_sub, sgn);                               // Tj |rb>
                        auto rb_new_label = rb_new.label(props_sub, scratch_work1, scratch_work2);
                        assert(basis_sub_repr[belong2rep[rb_new_label]] == rb);
                        auto dist_to_rb = dist2rep[rb_new_label];
                        if (dist_to_rb != disp_j_int) {                                           // remove over-countings
                            disp_j = dynamic_base_plus1(disp_j, base_sub);
                            continue;
                        }
                        mbasis_elem ra_z_Tj_rb;
                        zipper_basis(props_parent, props_sub, props_sub, ra, rb_new, ra_z_Tj_rb); // |ra> z Tj |rb>
                        
                        std::vector<uint32_t> disp_i(latt_sub_dim,0);
                        while (! dynamic_base_overflow(disp_i, base_parent)) {
                            std::vector<int> disp_i_int(latt_sub_dim);
                            for (uint32_t j = 0; j < latt_sub_dim; j++) disp_i_int[j] = static_cast<int>(disp_i[j]);
                            auto Ti_ra_z_Tj_rb = ra_z_Tj_rb;
                            latt_parent.translation_plan(plan_parent, disp_i_int, scratch_coor, scratch_work);
                            Ti_ra_z_Tj_rb.transform(props_parent, plan_parent, sgn);              // Ti (|ra> z Tj |rb>)
                            // now need find ja, jb
                            uint64_t state_sub1_label, state_sub2_label;
                            Ti_ra_z_Tj_rb.label_sub(props_parent, state_sub1_label, state_sub2_label,
                                                    scratch_work1, scratch_work2);
                            assert(belong2rep[state_sub1_label] == belong2rep[state_sub2_label]);
                            auto &dist2rep1       = dist2rep[state_sub1_label];                   // ja
                            auto &dist2rep2       = dist2rep[state_sub2_label];                   // jb
                            std::vector<uint64_t> pos{ga, gb};
                            pos.insert(pos.end(), dist2rep1.begin(), dist2rep1.end());            // ja
                            pos.insert(pos.end(), dist2rep2.begin(), dist2rep2.end());            // jb
                            assert(pos.size() == linear_size.size());
                            if ( disp_j < Weisse_e_eq.index(pos).second ||
                                (disp_j == Weisse_e_eq.index(pos).second && disp_i < Weisse_e_eq.index(pos).first)) {
                                Weisse_e_eq.index(pos).first  = disp_i;
                                Weisse_e_eq.index(pos).second = disp_j;
                            }
                            disp_i = dynamic_base_plus1(disp_i, base_parent);
                        }
                        disp_j = dynamic_base_plus1(disp_j, base_sub);
                    }
                }
            }
        }
        
        
        for (uint32_t ga = 0; ga < num_groups; ga++) {
            auto ra = examples[ga].front();
            for (uint32_t gb = 0; gb < num_groups; gb++) {
                std::vector<uint32_t> disp_i(latt_sub_dim,0);  // fixed to ja=0
                std::vector<uint32_t> disp_j(latt_sub_dim,0);  // now also serves the job of jb
                while (! dynamic_base_overflow(disp_j, base_sub)) {
                    std::vector<uint64_t> pos_e{ga,gb};
                    std::vector<uint64_t> pos_w{ga,gb};
                    pos_e.insert(pos_e.end(), disp_i.begin(), disp_i.end());
                    pos_e.insert(pos_e.end(), disp_j.begin(), disp_j.end());
                    pos_w.insert(pos_w.end(), disp_j.begin(), disp_j.end());
                    auto res_lt = Weisse_e_lt.index(pos_e);
                    auto res_eq = Weisse_e_eq.index(pos_e);
                    // build table w<
                    if (res_lt.first == disp_i && res_lt.second == disp_j) {
                        std::vector<int> disp_j_int(latt_sub_dim);
                        for (uint32_t j = 0; j < latt_sub_dim; j++) disp_j_int[j] = static_cast<int>(disp_j[j]);
                        auto rb_new = examples[gb].back();
                        assert(ra < rb_new);
                        int sgn;
                        latt_sub.translation_plan(plan_sub, disp_j_int, scratch_coor, scratch_work);
                        rb_new.transform(props_sub, plan_sub, sgn);                               // Tj |rb>
                        mbasis_elem ra_z_Tj_rb;
                        zipper_basis(props_parent, props_sub, props_sub, ra, rb_new, ra_z_Tj_rb); // |ra> z Tj |rb>
                        // loop over groups, to check which one the repr belongs to
                        for (uint32_t g = 0; g < groups_parent.size(); g++) {
                            bool flag = true;
                            for (uint32_t d = 0; d < latt_sub_dim; d++) {
                                if (! trans_sym[d]) continue;
                                std::vector<int> disp(latt_sub_dim);
                                for (uint32_t i = 0; i < latt_sub_dim; i++) disp[i] = groups_parent[g].first[d][i];
                                auto temp = ra_z_Tj_rb;
                                latt_parent.translation_plan(plan_parent, disp, scratch_coor, scratch_work);
                                temp.transform(props_parent, plan_parent, sgn);
                                if (temp != ra_z_Tj_rb) {
                                    flag = false;
                                    break;
                                }
                            }
                            if (flag) {                                                           // group found
                                Weisse_w_lt.index(pos_w) = g;
                                break;
                            }
                        }
                        assert(Weisse_w_lt.index(pos_w) < groups_parent.size());
                    }
                    // build table w=
                    if (res_eq.first == disp_i && res_eq.second == disp_j) {
                        std::vector<int> disp_j_int(latt_sub_dim);
                        for (uint32_t j = 0; j < latt_sub_dim; j++) disp_j_int[j] = static_cast<int>(disp_j[j]);
                        auto rb_new = examples[gb].front();
                        assert(ra == rb_new);
                        int sgn;
                        latt_sub.translation_plan(plan_sub, disp_j_int, scratch_coor, scratch_work);
                        rb_new.transform(props_sub, plan_sub, sgn);                               // Tj |rb>
                        mbasis_elem ra_z_Tj_rb;
                        zipper_basis(props_parent, props_sub, props_sub, ra, rb_new, ra_z_Tj_rb); // |ra> z Tj |rb>
                        // loop over groups, to check which one the repr belongs to
                        for (uint32_t g = 0; g < groups_parent.size(); g++) {
                            bool flag = true;
                            for (uint32_t d = 0; d < latt_sub_dim; d++) {
                                if (! trans_sym[d]) continue;
                                std::vector<int> disp(latt_sub_dim);
                                for (uint32_t i = 0; i < latt_sub_dim; i++) disp[i] = groups_parent[g].first[d][i];
                                auto temp = ra_z_Tj_rb;
                                latt_parent.translation_plan(plan_parent, disp, scratch_coor, scratch_work);
                                temp.transform(props_parent, plan_parent, sgn);
                                if (temp != ra_z_Tj_rb) {
                                    flag = false;
                                    break;
                                }
                            }
                            if (flag) {                                                           // group found
                                Weisse_w_eq.index(pos_w) = g;
                                break;
                            }
                        }
                        assert(Weisse_w_eq.index(pos_w) < groups_parent.size());
                    }
                    disp_j = dynamic_base_plus1(disp_j, base_sub);
                }
            }
            
            // build table w>
            if (Weisse_w_gt.size() == 0) continue;
            ra = examples[ga].back();
            for (uint32_t gb = 0; gb < num_groups; gb++) {
                std::vector<uint32_t> disp_i(latt_sub_dim,0);  // fixed to ja=0
                std::vector<uint32_t> disp_j(latt_sub_dim,0);  // now also serves the job of jb
                while (! dynamic_base_overflow(disp_j, base_sub)) {
                    std::vector<uint64_t> pos_e{ga,gb};
                    std::vector<uint64_t> pos_w{ga,gb};
                    pos_e.insert(pos_e.end(), disp_i.begin(), disp_i.end());
                    pos_e.insert(pos_e.end(), disp_j.begin(), disp_j.end());
                    pos_w.insert(pos_w.end(), disp_j.begin(), disp_j.end());
                    auto res_gt = Weisse_e_gt.index(pos_e);
                    
                    if (res_gt.first == disp_i && res_gt.second == disp_j) {
                        std::vector<int> disp_j_int(latt_sub_dim);
                        for (uint32_t j = 0; j < latt_sub_dim; j++) disp_j_int[j] = static_cast<int>(disp_j[j]);
                        auto rb_new = examples[gb].front();
                        assert(rb_new < ra);
                        int sgn;
                        latt_sub.translation_plan(plan_sub, disp_j_int, scratch_coor, scratch_work);
                        rb_new.transform(props_sub, plan_sub, sgn);                               // Tj |rb>
                        mbasis_elem ra_z_Tj_rb;
                        zipper_basis(props_parent, props_sub, props_sub, ra, rb_new, ra_z_Tj_rb); // |ra> z Tj |rb>
                        // loop over groups, to check which one the repr belongs to
                        for (uint32_t g = 0; g < groups_parent.size(); g++) {
                            bool flag = true;
                            for (uint32_t d = 0; d < latt_sub_dim; d++) {
                                if (! trans_sym[d]) continue;
                                std::vector<int> disp(latt_sub_dim);
                                for (uint32_t i = 0; i < latt_sub_dim; i++) disp[i] = groups_parent[g].first[d][i];
                                auto temp = ra_z_Tj_rb;
                                latt_parent.translation_plan(plan_parent, disp, scratch_coor, scratch_work);
                                temp.transform(props_parent, plan_parent, sgn);
                                if (temp != ra_z_Tj_rb) {
                                    flag = false;
                                    break;
                                }
                            }
                            if (flag) {                                                           // group found
                                Weisse_w_gt.index(pos_w) = g;
                                break;
                            }
                        }
                        assert(Weisse_w_gt.index(pos_w) < groups_parent.size());
                    }
                    
                    disp_j = dynamic_base_plus1(disp_j, base_sub);
                }
            }
            
            
        }
        
        log_Weisse_tables(latt_parent, groups_parent, groups_sub,
                          Weisse_e_lt, Weisse_e_eq, Weisse_e_gt,
                          Weisse_w_lt, Weisse_w_eq, Weisse_w_gt);
    }
    
    
    double norm_trans_repr(const std::vector<basis_prop> &props, const mbasis_elem &repr,
                           const lattice &latt_parent, const std::pair<std::vector<std::vector<uint32_t>>,uint32_t> &group_parent,
                           const std::vector<int> &momentum)
    {
        uint32_t dim = latt_parent.dimension();
        uint32_t N   = latt_parent.total_sites();
        auto L       = latt_parent.Linear_size();
        std::vector<uint32_t> plan_parent(latt_parent.total_sites());
        std::vector<int> scratch_work(latt_parent.dimension()), scratch_coor(latt_parent.dimension());
        
        std::vector<uint32_t> zerovec(dim,0);
        assert(std::any_of(group_parent.first.begin(), group_parent.first.end(), [zerovec](std::vector<uint32_t> i){ return i != zerovec; }));
        assert(momentum.size() == dim);
        auto momentum2 = momentum;
        for (uint32_t d = 0; d < dim; d++) {
            while (momentum2[d] < 0) momentum2[d] += static_cast<int>(L[d]);
            momentum2[d] %= static_cast<int>(L[d]);
        }
        
        bool bosonic = q_bosonic(props);
        bool flag_nonzero = true;
        for (uint32_t d_ou = 0; d_ou < dim; d_ou++) {
            auto &xyz = group_parent.first[d_ou];
            if (xyz == zerovec) continue;
            
            uint32_t numerator = 0;
            for (uint32_t d_in = 0; d_in < dim; d_in++) {
                numerator += static_cast<uint32_t>(momentum2[d_in]) * xyz[d_in] * N / L[d_in];
            }
            
            if (! bosonic) {
                std::vector<int> disp(dim);
                for (uint32_t d_in = 0; d_in < dim; d_in++) disp[d_in] = static_cast<int>(xyz[d_in]);
                int sgn;
                auto basis_temp = repr;
                latt_parent.translation_plan(plan_parent, disp, scratch_coor, scratch_work);
                basis_temp.transform(props, plan_parent, sgn);
                numerator += static_cast<uint32_t>(sgn % 2) * N / 2;
            }
            
            if (numerator % N != 0) {
                flag_nonzero = false;
                break;
            }
        }
        double nu = 1.0;
        if (flag_nonzero) {
            nu = static_cast<double>(group_parent.second);
        } else {
            nu = 0.0;
        }
        
        // the following lines are only for double checking purpose, should be removed in future
        //static int cnt = 0;
        //if (cnt++ == 0) std::cout << "(**remove**) ";
        
        
        uint32_t dim_trans = 0;
        for (uint32_t j = 0; j < dim; j++) {
            if (group_parent.first[j] != zerovec) dim_trans++;
        }
        
        double denominator = 1.0;
        for (uint32_t d = 0; d < dim; d++) {
            denominator *= (group_parent.first[d] == zerovec ? 1.0 : static_cast<double>(L[d]));
        }
        std::complex<double> nu_inv_check = 1.0;  // <r|P_k|r>
        auto num_sub = latt_parent.num_sublattice();
        for (uint32_t site = num_sub; site < latt_parent.total_sites(); site += num_sub) {
            std::vector<int> disp;
            int sub, sgn;
            latt_parent.site2coor(disp, sub, site);
            
            auto basis_temp = repr;
            latt_parent.translation_plan(plan_parent, disp, scratch_coor, scratch_work);
            basis_temp.transform(props, plan_parent, sgn);
            if (basis_temp != repr) continue;
            double exp_coef = 0.0;
            for (uint32_t d = 0; d < latt_parent.dimension(); d++) {
                if (group_parent.first[d] != zerovec) {
                    exp_coef += momentum2[d] * disp[d] / static_cast<double>(L[d]);
                }
            }
            auto coef = std::exp(std::complex<double>(0.0, 2.0 * pi * exp_coef));
            if (sgn % 2 == 1) coef *= std::complex<double>(-1.0, 0.0);
            nu_inv_check += coef;
        }
        nu_inv_check /= denominator;
        //std::cout << "repr: " << std::endl;
        //repr.prt_bits(props);
        assert(std::abs(std::imag(nu_inv_check)) < lanczos_precision);
        if (std::abs(nu) > lanczos_precision) {
            assert(std::abs(std::real(nu_inv_check) - 1.0/nu) < lanczos_precision);
        } else {
            assert(std::abs(std::real(nu_inv_check)) < lanczos_precision);
        }
        
        return nu;
    }
    
    // ----------------- implementation of wavefunction ------------------
    template <typename T>
    wavefunction<T>::wavefunction(const std::vector<basis_prop> &props, const MKL_INT &capacity) : bgn(0), end(0)
    {
        std::pair<mbasis_elem, T> gs(mbasis_elem(props), static_cast<T>(0));
        total_bytes = static_cast<int>(gs.first.mbits[0] * 256) + static_cast<int>(gs.first.mbits[1]);
        ele = std::vector<std::pair<mbasis_elem,T>>(capacity, gs);
    }
    
    template <typename T>
    wavefunction<T>::wavefunction(const mbasis_elem &old, const MKL_INT &capacity) : bgn(0), end(1)
    {
        std::pair<mbasis_elem, T> gs(old,static_cast<T>(1.0));
        total_bytes = static_cast<int>(old.mbits[0] * 256) + static_cast<int>(old.mbits[1]);
        ele = std::vector<std::pair<mbasis_elem,T>>(capacity, gs);
    }
    
    template <typename T>
    wavefunction<T>::wavefunction(mbasis_elem &&old, const MKL_INT &capacity) : bgn(0), end(1)
    {
        std::pair<mbasis_elem, T> gs(old,static_cast<T>(1.0));
        total_bytes = static_cast<int>(old.mbits[0] * 256) + static_cast<int>(old.mbits[1]);
        ele = std::vector<std::pair<mbasis_elem,T>>(capacity, gs);
    }
    
    template <typename T>
    bool wavefunction<T>::q_sorted() const
    {
        if (size() == 0 || size() == 1) return true;
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        bool check = true;
        for (MKL_INT j = (bgn + 1) % capacity; j != end; j = (j + 1) % capacity) {
            if (ele[j].first < ele[(j + capacity - 1) % capacity].first) {
                check = false;
                break;
            }
        }
        return check;
    }
    
    template <typename T>
    bool wavefunction<T>::q_sorted_fully() const
    {
        if (size() == 0 || size() == 1) return true;
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        bool check = true;
        for (MKL_INT j = (bgn + 1) % capacity; j != end; j = (j + 1) % capacity) {
            if ( !(ele[(j + capacity - 1) % capacity].first < ele[j].first) ) {
                check = false;
                break;
            }
        }
        return check;
    }
    
    template <typename T>
    double wavefunction<T>::amplitude() const
    {
        assert(q_sorted_fully());
        if (q_empty()) return 0.0;
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        double res = 0.0;
        for (MKL_INT j = bgn; j != end; j = (j + 1) % capacity) {
            res += std::norm(ele[j].second);
        }
        return res;
    }
    
    template <typename T>
    void wavefunction<T>::prt_bits(const std::vector<basis_prop> &props) const
    {
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        for (MKL_INT j = bgn; j != end; j = (j + 1) % capacity) {
            std::cout << "coeff: " << ele[j].second << std::endl;
            ele[j].first.prt_bits(props);
            std::cout << std::endl;
        }
    }
    
    template <typename T>
    void wavefunction<T>::prt_states(const std::vector<basis_prop> &props) const
    {
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        for (MKL_INT j = bgn; j != end; j = (j + 1) % capacity) {
            std::cout << "coeff: " << ele[j].second << std::endl;
            ele[j].first.prt_states(props);
            std::cout << std::endl;
        }
    }
    
    template <typename T>
    std::pair<mbasis_elem, T> &wavefunction<T>::operator[](MKL_INT n)
    {
        assert(n >= 0 && n < size());
        assert(! q_empty());
        return ele[(bgn + n) % static_cast<MKL_INT>(ele.size())];
    }
    
    template <typename T>
    const std::pair<mbasis_elem, T> &wavefunction<T>::operator[](MKL_INT n) const
    {
        assert(n >= 0 && n < size());
        assert(! q_empty());
        return ele[(bgn + n) % static_cast<MKL_INT>(ele.size())];
    }
    
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::reserve(const MKL_INT& capacity_new)
    {
        assert(capacity_new > 0);
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        if (capacity_new <= capacity) return *this;
        
        auto gs = ele[bgn].first;
        wavefunction<T> wv_new(gs, capacity_new);
        wv_new.copy(*this);
        
        using std::swap;
        swap(wv_new, *this);
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::clear()
    {
        bgn = 0;
        end = 0;
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::add(const mbasis_elem &rhs)
    {
        assert(ele[0].first.mbits[0] == rhs.mbits[0] && ele[0].first.mbits[1] == rhs.mbits[1]);
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        if (size() + 1 >= capacity) {
            capacity += capacity;
            reserve(capacity);
        }
        ele[end].second = static_cast<T>(1.0);
        std::memcpy(ele[end].first.mbits, rhs.mbits, total_bytes);
        end = (end + 1) % capacity;
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::copy(const mbasis_elem &rhs)
    {
        bgn = 0;
        end = 0;
        add(rhs);
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::add(const std::pair<mbasis_elem, T> &rhs)
    {
        assert(ele[0].first.mbits[0] == rhs.first.mbits[0] && ele[0].first.mbits[1] == rhs.first.mbits[1]);
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        if (size() + 1 >= capacity) {
            capacity += capacity;
            reserve(capacity);
        }
        ele[end].second = rhs.second;
        std::memcpy(ele[end].first.mbits, rhs.first.mbits, total_bytes);
        end = (end + 1) % capacity;
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::copy(const std::pair<mbasis_elem, T> &rhs)
    {
        bgn = 0;
        end = 0;
        add(rhs);
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::add(const wavefunction<T> &rhs)
    {
        if (rhs.q_empty()) return *this;
        assert(ele[0].first.mbits[0] == rhs.ele[0].first.mbits[0] && ele[0].first.mbits[1] == rhs.ele[0].first.mbits[1]);
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        MKL_INT cnt_total = rhs.size();
        if (size() + cnt_total >= capacity) {
            capacity += capacity;
            reserve(capacity);
        }
        
        for (MKL_INT cnt = 0; cnt < cnt_total; cnt++) {
            if (std::abs(rhs.ele[rhs.bgn+cnt].second) > machine_prec) {
                ele[end].second = rhs.ele[rhs.bgn+cnt].second;
                std::memcpy(ele[end].first.mbits, rhs.ele[rhs.bgn+cnt].first.mbits, total_bytes);
                end = (end + 1) % capacity;
            }
        }
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::copy(const wavefunction<T> &rhs)
    {
        bgn = 0;
        end = 0;
        add(rhs);
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::operator+=(mbasis_elem ele_new)
    {
        add(ele_new);
        return *this;
    }
    
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::operator+=(std::pair<mbasis_elem, T> ele_new)
    {
        add(ele_new);
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::operator+=(wavefunction<T> rhs)
    {
        add(rhs);
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::operator*=(const T &rhs)
    {
        if (q_empty()) return *this;            // itself zero
        if (std::abs(rhs) < machine_prec) {     // multiply by zero
            bgn = 0;
            end = 0;
            return *this;
        }
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        for (MKL_INT j = bgn; j != end; j = (j + 1) % capacity) ele[j].second *= rhs;
        return *this;
    }
    
    template <typename T>
    wavefunction<T> &wavefunction<T>::simplify()
    {
        if (q_empty()) return *this;            // itself zero
        MKL_INT size_old = size();
        MKL_INT capacity = static_cast<MKL_INT>(ele.size());
        
        // sqeeze into a single chunk
        using std::swap;
        if (bgn < end) {
            if (bgn > 0) {
                for (MKL_INT j = 0; j < size_old; j++) swap(ele[j], ele[bgn+j]);
            }
        } else {
            for (MKL_INT j = bgn; j < capacity; j++) swap(ele[j], ele[end + (j - bgn)]);
        }
        bgn = 0;
        end = size_old;
        
        // combine terms
        std::sort(ele.begin(), ele.begin() + size_old,
                  [](const std::pair<mbasis_elem, T> &lhs, const std::pair<mbasis_elem, T> &rhs){return lhs.first < rhs.first; });
        MKL_INT pos = bgn;
        while (pos < end) {
            MKL_INT pos_next = pos + 1;
            while (pos_next < end && ele[pos].first == ele[pos_next].first) pos_next++;
            for (MKL_INT j = pos + 1; j < pos_next; j++) {
                ele[pos].second += ele[j].second;
                ele[j].second = static_cast<T>(0.0);
            }
            pos = pos_next;
        }
        
        // delete zeros
        MKL_INT pos_zero = bgn;
        while (pos_zero < end && std::abs(ele[pos_zero].second) > opr_precision) pos_zero++;
        while (pos_zero < end) {
            MKL_INT pos_next = pos_zero + 1;
            while (pos_next < end && std::abs(ele[pos_next].second) < opr_precision) pos_next++;
            if (pos_next < end) {
                swap(ele[pos_zero], ele[pos_next]);
            } else {
                break;
            }
            while (pos_zero < end && std::abs(ele[pos_zero].second) > opr_precision) pos_zero++;
        }
        end = pos_zero;
        for (MKL_INT j = 0; j + 1 < end; j++) {
            assert(ele[j].first < ele[j+1].first);
            assert(std::abs(ele[j].second) > opr_precision);
        }
        if (bgn < end) assert(std::abs(ele[end - 1].second) > opr_precision);
        
        return *this;
    }
    
    template <typename T>
    void swap(wavefunction<T> &lhs, wavefunction<T> &rhs)
    {
        using std::swap;
        swap(lhs.bgn, rhs.bgn);
        swap(lhs.end, rhs.end);
        swap(lhs.total_bytes, rhs.total_bytes);
        swap(lhs.ele, rhs.ele);
    }

    template <typename T>
    T inner_product(const wavefunction<T> &lhs, const wavefunction<T> &rhs)
    {
        if (lhs.size() == 0 || rhs.size() == 0) return static_cast<T>(0.0);
        assert(lhs.ele[lhs.bgn].first.mbits[0] == rhs.ele[rhs.bgn].first.mbits[0] &&
               lhs.ele[lhs.bgn].first.mbits[1] == rhs.ele[rhs.bgn].first.mbits[1]);

        MKL_INT size_lhs = lhs.size();
        MKL_INT size_rhs = rhs.size();

        T res = 0.0;
        for (MKL_INT i = 0; i < size_lhs; i++) {
            auto &phi_i = lhs[i];
            for (MKL_INT j = 0; j < size_rhs; j++) {
                auto &phi_j = rhs[j];
                if (phi_i.first == phi_j.first) {
                    res += conjugate(phi_i.second) * phi_j.second;
                }
            }
        }
        return res;
    }
    
    template <typename T>
    wavefunction<T> operator+(const wavefunction<T> &lhs, const wavefunction<T> &rhs)
    {
        wavefunction<T> sum(lhs);
        sum.add(rhs);
        return sum;
    }
    
    template <typename T>
    wavefunction<T> operator*(const mbasis_elem &lhs, const T &rhs)
    {
        wavefunction<T> prod(lhs);
        prod *= rhs;
        return prod;
    }
    
    template <typename T>
    wavefunction<T> operator*(const T &lhs, const mbasis_elem &rhs)
    {
        wavefunction<T> prod(rhs);
        prod *= lhs;
        return prod;
    }
    
    template <typename T>
    wavefunction<T> operator*(const wavefunction<T> &lhs, const T &rhs)
    {
        wavefunction<T> prod(lhs);
        prod *= rhs;
        return prod;
    }
    
    template <typename T>
    wavefunction<T> operator*(const T &lhs, const wavefunction<T> &rhs)
    {
        wavefunction<T> prod(rhs);
        prod *= lhs;
        return prod;
    }
    
    // ----------------- implementation of operator * wavefunction ------------

    // example of sign count (spin fermion model):
    // site 0: one fermion, site 1: 0 fermion, site 2: one fermion, site 3: one fermion
    // |psi> = f_0^\dagger f_2^\dagger f_3^\dagger |0>
    // f_1^\dagger |psi> = - f_0^\dagger f_1^\dagger f_2^\dagger f_3^\dagger |0>
    template <typename T>
    void oprXphi(const opr<T> &lhs, const std::vector<basis_prop> &props, wavefunction<T> &res, const bool &append)
    {
        if (res.q_empty()) return;
        
        MKL_INT capacity = static_cast<MKL_INT>(res.ele.size());
        MKL_INT cnt_total = res.size();
        
        // if diagonal operator, implementation is simple
        if (lhs.diagonal) {
            if (append) {
                if (cnt_total + cnt_total >= capacity) {
                    capacity *= 2;
                    res.reserve(capacity);
                }
                for (MKL_INT cnt = 0; cnt < cnt_total; cnt++) {
                    auto &rhs = res.ele[res.bgn + cnt];
                    if (std::abs(rhs.second) < machine_prec) continue;
                    auto coeff_new = rhs.second * rhs.first.diagonal_operator(props,lhs);
                    if (std::abs(coeff_new) < machine_prec) continue;
                    std::memcpy(res.ele[res.end].first.mbits, rhs.first.mbits, res.total_bytes);
                    res.ele[res.end].second = coeff_new;
                    res.end = (res.end + 1) % capacity;
                }
            } else {
                for (MKL_INT j = res.bgn; j != res.end; j = (j+1) % capacity) {
                    if (std::abs(res.ele[j].second) > machine_prec)
                        res.ele[j].second *= res.ele[j].first.diagonal_operator(props, lhs);
                }
            }
            return;
        }
        
        // if operator is not diagonal
        auto dim = props[lhs.orbital].dim_local;
        assert(lhs.dim == dim);
        for (MKL_INT cnt = 0; cnt < cnt_total; cnt++) {                          // need operate on res cnt_total times
            if (res.size() + static_cast<int>(dim) >= capacity) {
                capacity *= 2;
                res.reserve(capacity);
            }
            
            // check if the particular element in original wavefunction equals == 0
            auto &rhs = append ? res.ele[res.bgn+cnt] : res.ele[res.bgn];
            if (std::abs(rhs.second) < machine_prec) {
                if (! append) res.bgn = (res.bgn + 1) % capacity;
                continue;
            }
            
            uint32_t col = rhs.first.siteRead(props, lhs.site, lhs.orbital);     // actually col <= 255
            uint32_t displacement = col * dim;
            
            // check if the full column == 0
            bool flag = true;
            for (uint8_t row = 0; row < dim; row++) {
                if (std::abs(lhs.mat[row + displacement]) > machine_prec) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                if (! append) res.bgn = (res.bgn + 1) % capacity;
                continue;
            }
            
            // count # of fermions traversed by this operator
            int sgn = 0;
            if (lhs.fermion) {
                for (uint32_t orb_cnt = 0; orb_cnt < lhs.orbital; orb_cnt++) {
                    if (props[orb_cnt].q_fermion()) {
                        for (uint32_t site_cnt = 0; site_cnt < props[orb_cnt].num_sites; site_cnt++) {
                            sgn = (sgn + props[orb_cnt].Nfermion_map[rhs.first.siteRead(props, site_cnt, orb_cnt)]) % 2;
                        }
                    }
                }
                assert(props[lhs.orbital].q_fermion());
                for (uint32_t site_cnt = 0; site_cnt < lhs.site; site_cnt++) {
                    sgn = (sgn + props[lhs.orbital].Nfermion_map[rhs.first.siteRead(props, site_cnt, lhs.orbital)]) % 2;
                }
            }
            
            // write down new elements in wavefunction
            for (uint8_t row = 0; row < dim; row++) {
                auto coeff = (sgn == 0 ? lhs.mat[row + displacement] : (-lhs.mat[row + displacement]));
                coeff *= rhs.second;
                if (std::abs(coeff) > machine_prec) {
                    res.ele[res.end].second = coeff;
                    std::memcpy(res.ele[res.end].first.mbits, rhs.first.mbits, res.total_bytes);
                    res.ele[res.end].first.siteWrite(props, lhs.site, lhs.orbital, row);
                    res.end = (res.end + 1) % capacity;
                }
            }
            if (! append) res.bgn = (res.bgn + 1) % capacity;
        }
    }
    
    
    template <typename T>
    void oprXphi(const opr<T> &lhs, const std::vector<basis_prop> &props, wavefunction<T> &res, mbasis_elem rhs, const bool &append)
    {
        if (! append) {
            res.bgn = 0;
            res.end = 0;
        }
        assert(res.ele[0].first.mbits[0] == rhs.mbits[0] && res.ele[0].first.mbits[1] == rhs.mbits[1]);
        MKL_INT capacity = static_cast<MKL_INT>(res.ele.size());
        
        auto dim = props[lhs.orbital].dim_local;
        assert(lhs.dim == dim);
        
        if (res.size() + static_cast<MKL_INT>(dim) >= capacity) {
            capacity *= 2;
            res.reserve(capacity);
        }
        
        uint32_t col = rhs.siteRead(props, lhs.site, lhs.orbital); // actually col <= 255
        if (lhs.diagonal) {
            assert(! lhs.fermion);
            if (std::abs(lhs.mat[col]) > machine_prec) {
                res.ele[res.end].second = lhs.mat[col];
                std::memcpy(res.ele[res.end].first.mbits, rhs.mbits, res.total_bytes);
                res.end = (res.end + 1) % capacity;
            }
        } else {
            uint32_t displacement = col * lhs.dim;
            bool flag = true;
            for (uint8_t row = 0; row < dim; row++) {
                if (std::abs(lhs.mat[row + displacement]) > machine_prec) {
                    flag = false;
                    break;
                }
            }
            if (flag) return;                                     // the full column == 0
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
                if (std::abs(coeff) > machine_prec) {
                    res.ele[res.end].second = coeff;
                    std::memcpy(res.ele[res.end].first.mbits, rhs.mbits, res.total_bytes);
                    res.ele[res.end].first.siteWrite(props, lhs.site, lhs.orbital, row);
                    res.end = (res.end + 1) % capacity;
                }
            }
        }
    }
    
    template <typename T>
    void oprXphi(const opr<T> &lhs, const std::vector<basis_prop> &props, wavefunction<T> &res, wavefunction<T> phi0, const bool &append)
    {
        if (! append) {
            res.bgn = 0;
            res.end = 0;
        }
        if (phi0.q_empty()) return;
        
        MKL_INT capacity0 = static_cast<MKL_INT>(phi0.ele.size());
        for (MKL_INT j = phi0.bgn; j != phi0.end; j = (j+1) % capacity0) {
            if (std::abs(phi0.ele[j].second) < machine_prec) continue;
            oprXphi(phi0.ele[j].second * lhs, props, res, phi0.ele[j].first, true);
        }
    }
    
    
    template <typename T>
    void oprXphi(const opr_prod<T> &lhs, const std::vector<basis_prop> &props, wavefunction<T> &res)
    {
        if (res.q_empty()) return;
        
        for (auto rit = lhs.mat_prod.rbegin(); rit != lhs.mat_prod.rend(); rit++) {
            oprXphi((*rit), props, res, false);
            res.simplify();
        }
        res *= lhs.coeff;
    }
    
    template <typename T>
    void oprXphi(const opr_prod<T> &lhs, const std::vector<basis_prop> &props, wavefunction<T> &res, mbasis_elem rhs, const bool &append)
    {
        if (lhs.q_zero()) {
            if (! append) {
                res.bgn = 0;
                res.end = 0;
            }
        } else {
            if (append) {
                wavefunction<T> temp(rhs);
                oprXphi(lhs, props, temp);
                res.add(temp);
            } else {
                res.bgn = 0;
                res.end = 1;
                res.ele[res.bgn].second = static_cast<T>(1.0);
                std::memcpy(res.ele[res.bgn].first.mbits, rhs.mbits, res.total_bytes);
                oprXphi(lhs, props, res);
            }
        }
    }
    
    template <typename T>
    void oprXphi(const opr_prod<T> &lhs, const std::vector<basis_prop> &props, wavefunction<T> &res, wavefunction<T> phi0, const bool &append)
    {
        if (! append) {
            res.bgn = 0;
            res.end = 0;
        }
        oprXphi(lhs, props, phi0);
        res.add(phi0);
    }
    
    template <typename T>
    void oprXphi(const mopr<T> &lhs, const std::vector<basis_prop> &props, wavefunction<T> &res, mbasis_elem rhs, const bool &append)
    {
        if (! append) {
            res.bgn = 0;
            res.end = 0;
        }
        if (lhs.q_zero()) return;                                // zero operator
        
        wavefunction<T> temp(rhs);
        for (auto it = lhs.mats.begin(); it != lhs.mats.end(); it++) {
            temp.copy(rhs);
            oprXphi((*it), props, temp);
            res.add(temp);
        }
    }
    
    template <typename T>
    void oprXphi(const mopr<T> &lhs, const std::vector<basis_prop> &props, wavefunction<T> &res, wavefunction<T> phi0, const bool &append)
    {
        if (! append) {
            res.bgn = 0;
            res.end = 0;
        }
        if (lhs.q_zero() || phi0.q_empty()) return;
        
        wavefunction<T> temp(phi0);
        for (auto it = lhs.mats.begin(); it != lhs.mats.end(); it++) {
            temp.copy(phi0);
            oprXphi((*it), props, temp);
            res.add(temp);
        }
    }
    
    template <typename T> void gen_mbasis_by_mopr(const mopr<T> &Ham, std::list<mbasis_elem> &basis,
                                                  const std::vector<basis_prop> &props,
                                                  std::vector<mopr<T>> conserve_lst, std::vector<double> val_lst)
    {
        assert(conserve_lst.size() == val_lst.size());
        int num_threads = 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) num_threads = omp_get_num_threads();
        }
        // prepare intermediates in advance
        std::vector<wavefunction<T>> intermediate_states(num_threads, {props});
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::list<mbasis_elem> basis_new;
        
        #pragma omp parallel for schedule(dynamic,256)
        for (decltype(basis.size()) j = 0; j < basis.size(); j++) {
            int tid = omp_get_thread_num();
            
            std::list<mbasis_elem> temp;
            auto it_basis = basis.begin();
            std::advance(it_basis, j);                                           // can be faster, if using stragy like in basis enumeration
         
            oprXphi(Ham, props, intermediate_states[tid], *it_basis);
            intermediate_states[tid].simplify();
            for (int cnt = 0; cnt < intermediate_states[tid].size(); cnt++) {
                auto &ele = intermediate_states[tid][cnt].first;
                if (ele != *it_basis) {
                    bool flag = true;
                    auto it_opr = conserve_lst.begin();
                    auto it_val = val_lst.begin();
                    while (it_opr != conserve_lst.end()) {
                        if (std::abs(ele.diagonal_operator(props, *it_opr) - *it_val) >= 1e-5) {
                            flag = false;
                            break;
                        }
                        it_opr++;
                        it_val++;
                    }
                    if (flag) temp.push_back(ele);
                }
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
    
    void rm_mbasis_dulp_trans(const lattice &latt, std::list<mbasis_elem> &basis, const std::vector<basis_prop> &props)
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        std::chrono::duration<double> elapsed_seconds;
        start = std::chrono::system_clock::now();
        std::cout << "Moving states to translational equivalents... " << std::flush;
        
        std::vector<int> disp_vec;
        for (auto it = basis.begin(); it != basis.end(); it++)
            it->translate2center_OBC(props, latt, disp_vec);
        end   = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl;
        
        start = std::chrono::system_clock::now();
        std::cout << "Resorting basis... ";
        basis.sort();
        end   = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl;
        
        start = std::chrono::system_clock::now();
        std::cout << "Removing translational dulplicates... ";
        auto it = basis.begin();
        auto it_prev = it++;
        while (it != basis.end()) {
            if (*it == *it_prev) {
                it = basis.erase(it);
            } else {
                it_prev = it++;
            }
        }
        end   = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "After removing translational dulplicates: " << basis.size() << std::endl;
    }
    
    
    // Explicit instantiation
    template void enumerate_basis(const std::vector<basis_prop> &props, std::vector<qbasis::mbasis_elem> &basis,
                                  std::vector<mopr<double>> conserve_lst, std::vector<double> val_lst);
    template void enumerate_basis(const std::vector<basis_prop> &props, std::vector<qbasis::mbasis_elem> &basis,
                                  std::vector<mopr<std::complex<double>>> conserve_lst, std::vector<double> val_lst);
    
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
    
    template void oprXphi(const opr<double>&,               const std::vector<basis_prop>&, wavefunction<double>&, const bool&);
    template void oprXphi(const opr<std::complex<double>>&, const std::vector<basis_prop>&, wavefunction<std::complex<double>>&, const bool&);
    
    template void oprXphi(const opr<double>&,               const std::vector<basis_prop>&, wavefunction<double>&,               mbasis_elem, const bool&);
    template void oprXphi(const opr<std::complex<double>>&, const std::vector<basis_prop>&, wavefunction<std::complex<double>>&, mbasis_elem, const bool&);
    
    template void oprXphi(const opr<double>&,               const std::vector<basis_prop>&, wavefunction<double>&,               wavefunction<double>, const bool&);
    template void oprXphi(const opr<std::complex<double>>&, const std::vector<basis_prop>&, wavefunction<std::complex<double>>&, wavefunction<std::complex<double>>, const bool&);
    
    template void oprXphi(const opr_prod<double>&,               const std::vector<basis_prop>&, wavefunction<double>&);
    template void oprXphi(const opr_prod<std::complex<double>>&, const std::vector<basis_prop>&, wavefunction<std::complex<double>>&);
    
    template void oprXphi(const opr_prod<double>&,               const std::vector<basis_prop>&, wavefunction<double>&, mbasis_elem, const bool&);
    template void oprXphi(const opr_prod<std::complex<double>>&, const std::vector<basis_prop>&, wavefunction<std::complex<double>>&, mbasis_elem, const bool&);
    
    template void oprXphi(const opr_prod<double>&,               const std::vector<basis_prop>&, wavefunction<double>&,               wavefunction<double>, const bool&);
    template void oprXphi(const opr_prod<std::complex<double>>&, const std::vector<basis_prop>&, wavefunction<std::complex<double>>&, wavefunction<std::complex<double>>, const bool&);
    
    template void oprXphi(const mopr<double>&,               const std::vector<basis_prop>&, wavefunction<double>&,               mbasis_elem, const bool&);
    template void oprXphi(const mopr<std::complex<double>>&, const std::vector<basis_prop>&, wavefunction<std::complex<double>>&, mbasis_elem, const bool&);
    
    template void oprXphi(const mopr<double>&,               const std::vector<basis_prop>&, wavefunction<double>&,               wavefunction<double>, const bool&);
    template void oprXphi(const mopr<std::complex<double>>&, const std::vector<basis_prop>&, wavefunction<std::complex<double>>&, wavefunction<std::complex<double>>, const bool&);

    template double inner_product(const wavefunction<double>&, const wavefunction<double>&);
    template std::complex<double> inner_product(const wavefunction<std::complex<double>>&, const wavefunction<std::complex<double>>&);

    template void gen_mbasis_by_mopr(const mopr<double>&, std::list<mbasis_elem>&, const std::vector<basis_prop>&,
                                     std::vector<mopr<double>> conserve_lst, std::vector<double> val_lst);
    template void gen_mbasis_by_mopr(const mopr<std::complex<double>>&, std::list<mbasis_elem>&, const std::vector<basis_prop>&,
                                     std::vector<mopr<std::complex<double>>> conserve_lst, std::vector<double> val_lst);
    
}
