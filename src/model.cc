#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <regex>
#include "qbasis.h"
#include "graph.h"

namespace qbasis {
    
    void ckpt_lanczos_clean();
    void ckpt_CG_clean();
    
    template <typename T>
    model<T>::model(const double &fake_pos_):
                    matrix_free(true), nconv(0),
                    sec_mat(0),
                    dim_full({0,0}), dim_repr({0,0}),
                    fake_pos(fake_pos_)
    {
        momenta.resize(2);
        basis_full.resize(2);
        basis_repr.resize(2);
        norm_repr.resize(2);
        Lin_Ja_full.resize(2);
        Lin_Jb_full.resize(2);
        Lin_Ja_repr.resize(2);
        Lin_Jb_repr.resize(2);
        HamMat_csr_full.resize(2);
        HamMat_csr_repr.resize(2);
        basis_belong_deprec.resize(2);
        basis_coeff_deprec.resize(2);
        basis_repr_deprec.resize(2);
    }
    
    template <typename T>
    uint32_t model<T>::local_dimension() const
    {
        uint32_t res = 1;
        for (decltype(props.size()) j = 0; j < props.size(); j++) res *= props[j].dim_local;
        return res;
    }
    
    template <typename T>
    void model<T>::switch_sec(const uint32_t &sec_mat_)
    {
        sec_mat = sec_mat_;
    }
    
    template <typename T>
    void model<T>::check_translation()
    {
        std::cout << "Checking translational symmetry (NOT a serious check at this moment)." << std::endl;
        trans_sym.clear();
        auto bc = latt_parent.boundary();
        for (uint32_t j = 0; j < latt_parent.dimension(); j++) {
            if (bc[j] == "pbc" || bc[j] == "PBC") {
                trans_sym.push_back(true);
            } else {
                trans_sym.push_back(false);
            }
        }
        
        uint32_t dim_spec = latt_parent.dimension_spec();
        if (dim_spec == latt_parent.dimension()) {
            assert(latt_parent.num_sublattice() % 2 == 0);
            dim_spec_involved = false;
        } else {
            dim_spec_involved = trans_sym[dim_spec];
        }
        
        std::cout << std::endl;
    }
    
    template <typename T>
    void model<T>::fill_Weisse_table(const lattice &latt)
    {
        latt_parent = latt;
        latt_sub = divide_lattice(latt_parent);
        
        check_translation();
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        auto props_sub = props_sub_a;
        
        groups_sub    = latt_sub.trans_subgroups(trans_sym);
        groups_parent = latt.trans_subgroups(trans_sym);
        
        std::cout << "------------------------------------" << std::endl;
        std::cout << "Generating sublattice full basis... " << std::endl;
        std::vector<mbasis_elem> basis_sub_full;
        enumerate_basis<T>(props_sub, basis_sub_full);
        sort_basis_normal_order(basis_sub_full);                                // has to be sorted in the normal way
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Elapsed time for generating sublattice full basis: " << elapsed_seconds.count() << "s." << std::endl << std::endl;
        start = end;
        
        std::cout << "------------------------------------" << std::endl;
        std::cout << "Classifying sublattice basis... " << std::flush;
        classify_trans_full2rep(props_sub, basis_sub_full, latt_sub, trans_sym, basis_sub_repr, belong2rep_sub, dist2rep_sub);
        classify_trans_rep2group(props_sub, basis_sub_repr, latt_sub, trans_sym, groups_sub, omega_g_sub, belong2group_sub);
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl << std::endl;
        start = end;
        
        // double checking correctness
        uint64_t check_dim_sub_full = 0;
        for (decltype(basis_sub_repr.size()) j = 0; j < basis_sub_repr.size(); j++) check_dim_sub_full += omega_g_sub[belong2group_sub[j]];
        assert(check_dim_sub_full == static_cast<uint64_t>(basis_sub_full.size()));
        
        std::cout << "------------------------------------" << std::endl;
        std::cout << "Generating maps (ga,gb,ja,jb) -> (i,j) and (ga,gb,j) -> (w) ... " << std::flush;
        classify_Weisse_tables(props, props_sub, basis_sub_repr, latt, trans_sym,
                               belong2rep_sub, dist2rep_sub, belong2group_sub, groups_parent, groups_sub,
                               Weisse_e_lt, Weisse_e_eq, Weisse_e_gt, Weisse_w_lt, Weisse_w_eq, Weisse_w_gt);
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl;
        std::cout << std::endl;
    }
    
    
    // need further optimization! (for example, special treatment of dilute limit; special treatment of quantum numbers; quick sort of sign)
    template <typename T>
    void model<T>::enumerate_basis_full(std::vector<mopr<T>> conserve_lst,
                                        std::vector<double> val_lst,
                                        const uint32_t &sec_full)
    {
        enumerate_basis<T>(props, basis_full[sec_full], conserve_lst, val_lst);
        
        dim_full[sec_full] = static_cast<MKL_INT>(basis_full[sec_full].size());
        
        sort_basis_Lin_order(props, basis_full[sec_full]);
        
        fill_Lin_table(props, basis_full[sec_full], Lin_Ja_full[sec_full], Lin_Jb_full[sec_full]);
        
        if (Lin_Ja_full[sec_full].size() == 0 || Lin_Jb_full[sec_full].size() == 0) {
            std::cout << "Due to faliure of Lin Table construction, fall back to bisection index of basis." << std::endl;
            sort_basis_normal_order(basis_full[sec_full]);
        }
        
        nconv = 0;
    }
    
    
    template <typename T>
    void model<T>::enumerate_basis_repr(const std::vector<int> &momentum,
                                        std::vector<mopr<T>> conserve_lst,
                                        std::vector<double> val_lst,
                                        const uint32_t &sec_repr)
    {
        assert(latt_parent.dimension() == static_cast<uint32_t>(momentum.size()));
        assert(conserve_lst.size() == val_lst.size());
        assert(Weisse_e_lt.size() > 0);
        assert(basis_sub_repr.size() > 0);   // should be already generated when filling Weisse Tables
        
        momenta[sec_repr] = momentum;
        
        if (dim_spec_involved) {
            assert(Weisse_w_gt.size() == 0);
        } else {
            assert(Weisse_w_lt.size() == Weisse_w_gt.size());
        }
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        auto L        = latt_parent.Linear_size();
        auto base_sub = latt_sub.Linear_size();
        std::cout << "Momentum: (" << std::flush;
        for (uint32_t j = 0; j < momentum.size(); j++) {
            if (trans_sym[j]) {
                std::cout << momentum[j] << "\t";
            } else {
                std::cout << "NA\t";
            }
        }
        std::cout << "):" << std::endl;
        
        // now start enumerating representatives, if not generated before (or generated but already destroyed)
        if (dim_repr[sec_repr] <= 0 || static_cast<MKL_INT>(basis_repr[sec_repr].size()) != dim_repr[sec_repr]) {
            std::cout << "Enumerating basis_repr..." << std::endl;
            basis_repr[sec_repr].clear();
            std::list<std::vector<mbasis_elem>> basis_temp;
            dim_repr[sec_repr] = 0;
            auto report = basis_sub_repr.size() > 100 ? (basis_sub_repr.size() / 10) : basis_sub_repr.size();
            #pragma omp parallel for schedule(dynamic,1)
            for (decltype(basis_sub_repr.size()) ra = 0; ra < basis_sub_repr.size(); ra++) {
                if (ra > 0 && ra % report == 0) {
                    std::cout << "progress: "
                    << (static_cast<double>(ra) / static_cast<double>(basis_sub_repr.size()) * 100.0) << "%" << std::endl;
                }
                std::vector<qbasis::mbasis_elem> basis_temp_job;
                auto ga = belong2group_sub[ra];
                int sgn;
                for (decltype(ra) rb = (dim_spec_involved?ra:0); rb < basis_sub_repr.size(); rb++) {
                    auto gb = belong2group_sub[rb];
                    std::vector<uint32_t> disp_j(latt_sub.dimension(),0);
                    std::vector<int> disp_j_int(disp_j.size());
                    while (! dynamic_base_overflow(disp_j, base_sub)) {
                        auto pos = std::vector<uint64_t>{ga,gb};
                        pos.insert(pos.end(), disp_j.begin(), disp_j.end());
                        uint32_t omega;
                        if (ra < rb) {
                            omega = Weisse_w_lt.index(pos);
                        } else if (ra == rb) {
                            omega = Weisse_w_eq.index(pos);
                        } else {
                            omega = Weisse_w_gt.index(pos);
                        }
                        
                        if (omega < groups_parent.size()) {  // valid representative
                            mbasis_elem rb_new = basis_sub_repr[rb];
                            for (uint32_t j = 0; j < latt_sub.dimension(); j++) disp_j_int[j] = static_cast<int>(disp_j[j]);
                            rb_new.translate(props_sub_b, latt_sub, disp_j_int, sgn);
                            mbasis_elem ra_z_Tj_rb;
                            zipper_basis(props, props_sub_a, props_sub_b, basis_sub_repr[ra], rb_new, ra_z_Tj_rb);
                            // check if the symmetries are obeyed
                            bool flag = true;
                            auto it_opr = conserve_lst.begin();
                            auto it_val = val_lst.begin();
                            while (it_opr != conserve_lst.end()) {
                                auto temp = ra_z_Tj_rb.diagonal_operator(props, *it_opr);
                                if (std::abs(temp - *it_val) >= 1e-5) {
                                    flag = false;
                                    break;
                                }
                                it_opr++;
                                it_val++;
                            }
                            if (flag) basis_temp_job.push_back(ra_z_Tj_rb);
                        }
                        disp_j = dynamic_base_plus1(disp_j, base_sub);
                    }
                }
                
                if (basis_temp_job.size() > 0) {
                    #pragma omp critical
                    {
                        dim_repr[sec_repr] += static_cast<MKL_INT>(basis_temp_job.size());
                        basis_temp.push_back(std::move(basis_temp_job));
                    }
                }
                
            }
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
            start = end;
            std::cout << "Hilbert space size with symmetry:      " << dim_repr[sec_repr] << std::endl;
            
            basis_repr[sec_repr].reserve(dim_repr[sec_repr]);
            std::cout << "Moving temporary basis (" << basis_temp.size() << " pieces) to basis_repr... ";
            for (auto it = basis_temp.begin(); it != basis_temp.end(); it++) {
                basis_repr[sec_repr].insert(basis_repr[sec_repr].end(), std::make_move_iterator(it->begin()), std::make_move_iterator(it->end()));
                it->erase(it->begin(), it->end());
                it->shrink_to_fit();
            }
            assert(dim_repr[sec_repr] == static_cast<MKL_INT>(basis_repr[sec_repr].size()));
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
            std::cout << elapsed_seconds.count() << "s." << std::endl << std::endl;
            start = end;
            
            sort_basis_Lin_order(props, basis_repr[sec_repr]);
            
            fill_Lin_table(props, basis_repr[sec_repr], Lin_Ja_repr[sec_repr], Lin_Jb_repr[sec_repr]);
            
            if (Lin_Ja_repr[sec_repr].size() == 0 || Lin_Jb_repr[sec_repr].size() == 0) {
                std::cout << "Due to faliure of Lin Table construction, fall back to bisection index of basis." << std::endl;
                sort_basis_normal_order(basis_repr[sec_repr]);
                assert(is_sorted_norepeat(basis_repr[sec_repr]));
            }
            
            nconv = 0;
        }
        
        // calculate normalization factors
        std::cout << "Calculating normalization factors (a much faster version already written, should be turned on in future)..." << std::endl;
        start = std::chrono::system_clock::now();
        std::cout << "dim_repr = " << dim_repr[sec_repr] << " - " << std::flush;
        MKL_INT extra = 0;
        norm_repr[sec_repr].clear();
        norm_repr[sec_repr].resize(dim_repr[sec_repr]);
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT j = 0; j < dim_repr[sec_repr]; j++) {
            uint64_t state_sub1_label, state_sub2_label;
            basis_repr[sec_repr][j].label_sub(props, state_sub1_label, state_sub2_label);
            auto &ra_label = belong2rep_sub[state_sub1_label];
            auto &rb_label = belong2rep_sub[state_sub2_label];
            auto &ga = belong2group_sub[ra_label];
            auto &gb = belong2group_sub[rb_label];
            std::vector<uint64_t> pos_w{ga, gb};
            pos_w.insert(pos_w.end(), dist2rep_sub[state_sub2_label].begin(), dist2rep_sub[state_sub2_label].end());
            uint32_t g_label;
            if (ra_label < rb_label) {
                g_label = Weisse_w_lt.index(pos_w);
            } else if (ra_label == rb_label) {
                g_label = Weisse_w_eq.index(pos_w);
            } else {
                g_label = Weisse_w_gt.index(pos_w);
            }
            
            norm_repr[sec_repr][j] = norm_trans_repr(props, basis_repr[sec_repr][j], latt_parent, groups_parent[g_label], momentum);
            if (std::abs(norm_repr[sec_repr][j]) < lanczos_precision) {
                #pragma omp atomic
                extra++;
            }
        }
        std::cout << extra << " = " << (dim_repr[sec_repr] - extra) << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl << std::endl;
    }
    
    template <typename T>
    void model<T>::generate_Ham_sparse_full(const bool &upper_triangle)
    {
        if (matrix_free) matrix_free = false;
        assert(dim_full[sec_mat] > 0);
        
        MKL_INT dim = dim_full[sec_mat];
        auto &basis = basis_full[sec_mat];
        
        int num_threads = 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) num_threads = omp_get_num_threads();
        }
        // prepare intermediates in advance
        std::vector<wavefunction<T>> intermediate_states(num_threads, {basis[0]});
        
        std::cout << "Generating LIL Hamiltonian matrix (full)..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        lil_mat<T> matrix_lil(dim, upper_triangle);
        #pragma omp parallel for schedule(dynamic,16)
        for (MKL_INT i = 0; i < dim; i++) {
            int tid = omp_get_thread_num();
            // diagonal part:
            for (auto it = Ham_diag.mats.begin(); it != Ham_diag.mats.end(); it++) {
                matrix_lil.add(i, i, basis[i].diagonal_operator(props, *it));
            }
            
            // non-diagonal part:
            uint64_t i_a, i_b;
            MKL_INT j;
            for (auto it = Ham_off_diag.mats.begin(); it != Ham_off_diag.mats.end(); it++) {
                
                
                
                
                intermediate_states[tid].copy(basis[i]);
                oprXphi(*it, props, intermediate_states[tid]);
                
                /*
                if (i == 6041) {
                    std::cout << "---state---" << std::endl;
                    basis[i].prt_states(props);
                    std::cout << std::endl;
                    
                    std::cout << "---operator---" << std::endl;
                    it->prt();
                    std::cout << std::endl;
                    
                    std::cout << "--- new state ---" << std::endl;
                    for (int cnt = 0; cnt < intermediate_states[tid].size(); cnt++) {
                        std::cout << "cnt = " << cnt << std::endl;
                        intermediate_states[tid][cnt].first.prt_states(props);
                        std::cout << std::endl;
                    }
                }
                */
                
                for (int cnt = 0; cnt < intermediate_states[tid].size(); cnt++) {
                    auto &ele_new = intermediate_states[tid][cnt];
                    if (std::abs(ele_new.second) < machine_prec) continue;
                    if (Lin_Ja_full[sec_mat].size() > 0 && Lin_Jb_full[sec_mat].size() > 0) {
                        ele_new.first.label_sub(props, i_a, i_b);
                        j = Lin_Ja_full[sec_mat][i_a] + Lin_Jb_full[sec_mat][i_b];
                    } else {
                        j = binary_search<mbasis_elem,MKL_INT>(basis, ele_new.first, 0, dim);
                    }
                    if (j < 0 || j >= dim) continue;
                    if (upper_triangle) {
                        if (i <= j) matrix_lil.add(i, j, conjugate(ele_new.second));
                    } else {
                        matrix_lil.add(i, j, conjugate(ele_new.second));
                    }
                }
                
            }
        }
        HamMat_csr_full[sec_mat] = csr_mat<T>(matrix_lil);
        std::cout << "Hamiltonian CSR matrix (full) generated." << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    template <typename T>
    void model<T>::generate_Ham_sparse_repr(const bool &upper_triangle)
    {
        if (matrix_free) matrix_free = false;
        MKL_INT dim = dim_repr[sec_mat];
        auto &basis = basis_repr[sec_mat];
        assert(dim > 0);
        if (dim_spec_involved) {
            assert(Weisse_w_gt.size() == 0);
        } else {
            assert(Weisse_w_lt.size() == Weisse_w_gt.size());
        }
        int num_threads = 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) num_threads = omp_get_num_threads();
        }
        // prepare intermediates in advance
        std::vector<wavefunction<T>> intermediate_states(num_threads, {basis[0]});
        auto dim_latt = latt_parent.dimension();
        auto L = latt_parent.Linear_size();
        bool bosonic = q_bosonic(props);
        
        std::cout << "Generating LIL Hamiltonian Matrix (repr)..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        lil_mat<std::complex<double>> matrix_lil(dim, upper_triangle);
        #pragma omp parallel for schedule(dynamic,16)
        for (MKL_INT i = 0; i < dim; i++) {
            double nu_i = norm_repr[sec_mat][i];                                // normalization factor for repr i
            if (std::abs(nu_i) < lanczos_precision) {
                matrix_lil.add(i, i, static_cast<T>(fake_pos + static_cast<double>(i)/static_cast<double>(dim)));
                continue;
            }
            
            int tid = omp_get_thread_num();
            
            // diagonal part:
            for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)
                matrix_lil.add(i, i, basis[i].diagonal_operator(props,Ham_diag[cnt]));
            
            // non-diagonal part:
            uint64_t state_sub1_label, state_sub2_label;
            std::vector<uint32_t> disp_i(dim_latt), disp_j(dim_latt);
            std::vector<int> disp_i_int(dim_latt), disp_j_int(dim_latt);
            int sgn;
            mbasis_elem state_sub_new1, state_sub_new2, ra_z_Tj_rb;
            uint64_t i_a, i_b;
            MKL_INT j;
            for (auto it = Ham_off_diag.mats.begin(); it != Ham_off_diag.mats.end(); it++) {
                intermediate_states[tid].copy(basis[i]);
                oprXphi(*it, props, intermediate_states[tid]);
                
                for (int cnt = 0; cnt < intermediate_states[tid].size(); cnt++) {
                    auto &ele_new = intermediate_states[tid][cnt];
                    // use Weisse Tables to find the representative |ra,rb,j>
                    ele_new.first.label_sub(props, state_sub1_label, state_sub2_label);
                    auto &state_rep1_label = belong2rep_sub[state_sub1_label];       // ra
                    auto &state_rep2_label = belong2rep_sub[state_sub2_label];       // rb
                    auto &ga               = belong2group_sub[state_rep1_label];     // ga
                    auto &gb               = belong2group_sub[state_rep2_label];     // gb
                    std::vector<uint64_t> pos_e{ga, gb};
                    pos_e.insert(pos_e.end(), dist2rep_sub[state_sub1_label].begin(), dist2rep_sub[state_sub1_label].end());
                    pos_e.insert(pos_e.end(), dist2rep_sub[state_sub2_label].begin(), dist2rep_sub[state_sub2_label].end());
                    if (state_rep1_label < state_rep2_label) {                          // ra < rb
                        disp_i = Weisse_e_lt.index(pos_e).first;
                        disp_j = Weisse_e_lt.index(pos_e).second;
                    } else if (state_rep2_label < state_rep1_label) {                   // ra > rb
                        disp_i = Weisse_e_gt.index(pos_e).first;
                        disp_j = Weisse_e_gt.index(pos_e).second;
                    } else {                                                            // ra == rb
                        disp_i = Weisse_e_eq.index(pos_e).first;
                        disp_j = Weisse_e_eq.index(pos_e).second;
                    }
                    for (uint32_t j = 0; j < disp_j.size(); j++) {
                        disp_i_int[j] = static_cast<int>(disp_i[j]);
                        disp_j_int[j] = static_cast<int>(disp_j[j]);
                    }
                    
                    if (state_rep2_label < state_rep1_label && dim_spec_involved) {
                        state_sub_new1 = basis_sub_repr[state_rep2_label];
                        state_sub_new2 = basis_sub_repr[state_rep1_label];
                    } else {
                        state_sub_new1 = basis_sub_repr[state_rep1_label];
                        state_sub_new2 = basis_sub_repr[state_rep2_label];
                    }
                    
                    state_sub_new2.translate(props_sub_b, latt_sub, disp_j_int, sgn);   // T_j |rb>
                    zipper_basis(props, props_sub_a, props_sub_b, state_sub_new1, state_sub_new2, ra_z_Tj_rb); // |ra> z T_j |rb>
                    
                    if (Lin_Ja_repr[sec_mat].size() > 0 && Lin_Jb_repr[sec_mat].size() > 0) {
                        i_a = state_sub_new1.label(props_sub_a);                    // use Lin Tables
                        i_b = state_sub_new2.label(props_sub_b);
                        j = Lin_Ja_repr[sec_mat][i_a] + Lin_Jb_repr[sec_mat][i_b];
                    } else {
                        j = binary_search<mbasis_elem,MKL_INT>(basis, ra_z_Tj_rb, 0, dim);
                    }
                    if (j < 0 || j >= dim) continue;
                    assert(ra_z_Tj_rb == basis[j]);
                    double nu_j = norm_repr[sec_mat][j];
                    if (std::abs(nu_j) < lanczos_precision) continue;
                    
                    double exp_coef = 0.0;
                    for (uint32_t d = 0; d < latt_parent.dimension(); d++) {
                        if (trans_sym[d]) {
                            exp_coef += momenta[sec_mat][d] * disp_i_int[d] / static_cast<double>(L[d]);
                        }
                    }
                    auto coef = std::sqrt(nu_i / nu_j) * conjugate(ele_new.second) * std::exp(std::complex<double>(0.0, 2.0 * pi * exp_coef));
                    if (! bosonic) {
                        ra_z_Tj_rb.translate(props, latt_parent, disp_i_int, sgn);          // to get sgn
                        assert(ra_z_Tj_rb == ele_new.first);
                        if (sgn % 2 == 1) coef *= std::complex<double>(-1.0, 0.0);
                    }
                    
                    if (upper_triangle) {
                        if (i <= j) matrix_lil.add(i, j, coef);
                    } else {
                        matrix_lil.add(i, j, coef);
                    }
                }
            }
        }
        
        HamMat_csr_repr[sec_mat] = csr_mat<std::complex<double>>(matrix_lil);
        std::cout << "Hamiltonian CSR matrix (repr) generated." << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    
    
    template <typename T>
    std::vector<std::complex<double>> model<T>::to_dense()
    {
        std::cout << "Fall back to generate matrix explicitly!" << std::endl;
        if (sec_sym == 0) {
            generate_Ham_sparse_full();
            return HamMat_csr_full[sec_mat].to_dense();
        } else {
            generate_Ham_sparse_repr();
            return HamMat_csr_full[sec_mat].to_dense();
        }
    }
    
    
    template <typename T>
    void model<T>::MultMv2(const T *x, T *y) const
    {
        assert(matrix_free);
        MKL_INT dim = (sec_sym == 0) ? dim_full[sec_mat] : dim_repr[sec_mat];
        auto &basis = (sec_sym == 0) ? basis_full[sec_mat] : basis_repr[sec_mat];
        int num_threads = 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) num_threads = omp_get_num_threads();
        }
        // prepare intermediates in advance
        std::vector<wavefunction<T>> intermediate_states(num_threads, {basis[0]});
        
        std::cout << "*" << std::flush;
        if (sec_sym == 0) {
            #pragma omp parallel for schedule(dynamic,16)
            for (MKL_INT i = 0; i < dim; i++) {
                int tid = omp_get_thread_num();
                
                // diagonal part
                if (std::abs(x[i]) > machine_prec) {
                    for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)
                        y[i] += x[i] * basis[i].diagonal_operator(props, Ham_diag[cnt]);
                }
                
                // non-diagonal part
                uint64_t i_a, i_b;
                MKL_INT j;
                for (auto it = Ham_off_diag.mats.begin(); it != Ham_off_diag.mats.end(); it++) {
                    intermediate_states[tid].copy(basis[i]);
                    oprXphi(*it, props, intermediate_states[tid]);
                    for (int cnt = 0; cnt < intermediate_states[tid].size(); cnt++) {
                        auto &ele_new = intermediate_states[tid][cnt];
                        if (std::abs(ele_new.second) < machine_prec) continue;
                        if (Lin_Ja_full[sec_mat].size() > 0 && Lin_Jb_full[sec_mat].size() > 0) {
                            ele_new.first.label_sub(props, i_a, i_b);
                            j = Lin_Ja_full[sec_mat][i_a] + Lin_Jb_full[sec_mat][i_b];
                        } else {
                            j = binary_search<mbasis_elem,MKL_INT>(basis, ele_new.first, 0, dim);
                        }
                        if (j < 0 || j >= dim) continue;
                        if (std::abs(x[j]) > machine_prec) y[i] += (x[j] * conjugate(ele_new.second));
                    }
                }
            }
        } else {
            auto dim_latt = latt_parent.dimension();
            auto L        = latt_parent.Linear_size();
            bool bosonic  = q_bosonic(props);
            
            #pragma omp parallel for schedule(dynamic,16)
            for (MKL_INT i = 0; i < dim; i++) {
                double nu_i = norm_repr[sec_mat][i];                             // normalization factor for repr i
                if (std::abs(nu_i) < lanczos_precision) {
                    y[i] += x[i] * static_cast<T>(fake_pos + static_cast<double>(i)/static_cast<double>(dim));
                    continue;
                }
                
                int tid = omp_get_thread_num();
                
                // diagonal part
                if (std::abs(x[i]) > machine_prec) {
                    for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)          // diagonal part:
                        y[i] += x[i] * basis[i].diagonal_operator(props,Ham_diag[cnt]);
                }
                
                // non-diagonal part
                uint64_t state_sub1_label, state_sub2_label;
                std::vector<uint32_t> disp_i(dim_latt), disp_j(dim_latt);
                std::vector<int> disp_i_int(dim_latt), disp_j_int(dim_latt);
                int sgn;
                mbasis_elem state_sub_new1, state_sub_new2, ra_z_Tj_rb;
                uint64_t i_a, i_b;
                MKL_INT j;
                for (auto it = Ham_off_diag.mats.begin(); it != Ham_off_diag.mats.end(); it++) {
                    intermediate_states[tid].copy(basis[i]);
                    oprXphi(*it, props, intermediate_states[tid]);
                    
                    for (int cnt = 0; cnt < intermediate_states[tid].size(); cnt++) {
                        auto &ele_new = intermediate_states[tid][cnt];
                        // use Weisse Tables to find the representative |ra,rb,j>
                        ele_new.first.label_sub(props, state_sub1_label, state_sub2_label);
                        auto &state_rep1_label = belong2rep_sub[state_sub1_label];       // ra
                        auto &state_rep2_label = belong2rep_sub[state_sub2_label];       // rb
                        auto &ga               = belong2group_sub[state_rep1_label];     // ga
                        auto &gb               = belong2group_sub[state_rep2_label];     // gb
                        std::vector<uint64_t> pos_e{ga, gb};
                        pos_e.insert(pos_e.end(), dist2rep_sub[state_sub1_label].begin(), dist2rep_sub[state_sub1_label].end());
                        pos_e.insert(pos_e.end(), dist2rep_sub[state_sub2_label].begin(), dist2rep_sub[state_sub2_label].end());
                        if (state_rep1_label < state_rep2_label) {                          // ra < rb
                            disp_i = Weisse_e_lt.index(pos_e).first;
                            disp_j = Weisse_e_lt.index(pos_e).second;
                        } else if (state_rep2_label < state_rep1_label) {                   // ra > rb
                            disp_i = Weisse_e_gt.index(pos_e).first;
                            disp_j = Weisse_e_gt.index(pos_e).second;
                        } else {                                                            // ra == rb
                            disp_i = Weisse_e_eq.index(pos_e).first;
                            disp_j = Weisse_e_eq.index(pos_e).second;
                        }
                        for (uint32_t j = 0; j < disp_j.size(); j++) {
                            disp_i_int[j] = static_cast<int>(disp_i[j]);
                            disp_j_int[j] = static_cast<int>(disp_j[j]);
                        }
                        
                        if (state_rep2_label < state_rep1_label && dim_spec_involved) {
                            state_sub_new1 = basis_sub_repr[state_rep2_label];
                            state_sub_new2 = basis_sub_repr[state_rep1_label];
                        } else {
                            state_sub_new1 = basis_sub_repr[state_rep1_label];
                            state_sub_new2 = basis_sub_repr[state_rep2_label];
                        }
                        
                        state_sub_new2.translate(props_sub_b, latt_sub, disp_j_int, sgn);   // T_j |rb>
                        zipper_basis(props, props_sub_a, props_sub_b, state_sub_new1, state_sub_new2, ra_z_Tj_rb); // |ra> z T_j |rb>
                        if (Lin_Ja_repr[sec_mat].size() > 0 && Lin_Jb_repr[sec_mat].size() > 0) {
                            i_a = state_sub_new1.label(props_sub_a);               // use Lin Tables
                            i_b = state_sub_new2.label(props_sub_b);
                            j = Lin_Ja_repr[sec_mat][i_a] + Lin_Jb_repr[sec_mat][i_b];
                        } else {
                            j = binary_search<mbasis_elem,MKL_INT>(basis, ra_z_Tj_rb, 0, dim);
                        }
                        if (j < 0 || j >= dim) continue;
                        assert(ra_z_Tj_rb == basis[j]);
                        if (std::abs(x[j]) < machine_prec) continue;
                        double nu_j = norm_repr[sec_mat][j];
                        if (std::abs(nu_j) < lanczos_precision) continue;
                        
                        double exp_coef = 0.0;
                        for (uint32_t d = 0; d < latt_parent.dimension(); d++) {
                            if (trans_sym[d]) {
                                exp_coef += momenta[sec_mat][d] * disp_i_int[d] / static_cast<double>(L[d]);
                            }
                        }
                        auto coef = std::sqrt(nu_i / nu_j) * conjugate(ele_new.second) * std::exp(std::complex<double>(0.0, 2.0 * pi * exp_coef));
                        if (! bosonic) {
                            ra_z_Tj_rb.translate(props, latt_parent, disp_i_int, sgn);          // to get sgn
                            assert(ra_z_Tj_rb == ele_new.first);
                            if (sgn % 2 == 1) coef *= std::complex<double>(-1.0, 0.0);
                        }
                        
                        y[i] += (x[j] * coef);
                    }
                }
            }
        }
    }
    
    template <typename T>
    void model<T>::MultMv(T *x, T *y)
    {
        T zero = static_cast<T>(0.0);
        if (sec_sym == 0) {
            for (MKL_INT j = 0; j < dim_full[sec_mat]; j++) y[j] = zero;
        } else {
            for (MKL_INT j = 0; j < dim_repr[sec_mat]; j++) y[j] = zero;
        }
        MultMv2(x, y);
    }
    
    template <typename T>
    void model<T>::locate_E0_lanczos(const uint32_t &sec_sym_, const MKL_INT &nev, const MKL_INT &ncv, MKL_INT maxit)
    {
        std::cout << "Locating E0 with Lanczos (sec_sym = " << sec_sym_ << ")..." << std::endl;
        assert(nev > 0 && nev <= 2 && ncv >= nev - 1 && ncv <= nev);
        uint32_t seed   = 1;
        sec_sym         = sec_sym_;
        MKL_INT dim     = (sec_sym == 0) ? dim_full[sec_mat] : dim_repr[sec_mat];
        auto &HamMat    = (sec_sym == 0) ? HamMat_csr_full[sec_mat] : HamMat_csr_repr[sec_mat];
        auto &eigenvals = (sec_sym == 0) ? eigenvals_full : eigenvals_repr;
        auto &eigenvecs = (sec_sym == 0) ? eigenvecs_full : eigenvecs_repr;
        assert(sec_sym < 2);
        assert(dim > 0);
        
        using std::swap;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        std::chrono::duration<double> elapsed_seconds;
        eigenvecs.clear();
        std::vector<double> hessenberg(2*maxit, 0.0), ritz, s;
        std::vector<T> v;
        if (ncv > 0) {
            v.resize(dim*4);
        } else {
            v.resize(dim*2);
        }
        vec_randomize(dim, v.data(), seed);
        
        bool E0_done = false;
        bool V0_done = false;
        bool E1_done = false;
        bool V1_done = false;
        ckpt_lczsE0_init(E0_done, V0_done, E1_done, V1_done, v);
        
        if (! E0_done) {
            std::cout << "Calculating ground state energy (simple Lanczos)..." << std::endl;
            start = std::chrono::system_clock::now();
            MKL_INT m = 0;
            if (matrix_free) {
                lanczos(0, maxit-1, maxit, m, dim, *this,  v.data(), hessenberg.data(), "sr_val0");
            } else {
                lanczos(0, maxit-1, maxit, m, dim, HamMat, v.data(), hessenberg.data(), "sr_val0");
            }
            assert(m < maxit);
            hess_eigen(hessenberg.data(), maxit, m, "sr", ritz, s);
            eigenvals.resize(1);
            eigenvals[0] = ritz[0];
            E0 = eigenvals[0];
            nconv = 0;
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
            std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
            std::cout << "Lanczos steps: " << m << std::endl;
            std::cout << "Lanczos accuracy: " << std::abs(hessenberg[m] * s[m-1]) << std::endl;
            std::cout << "E0   = " << E0 << std::endl << std::endl;
            
            E0_done = true;
            ckpt_lczsE0_updt(E0_done, V0_done, E1_done, V1_done);
        }
        
        if (ncv == 0) {
            // clean up ckpt
            return;
        }
        
        // obtain ground state eigenvector
        if (! V0_done) {
            start = std::chrono::system_clock::now();
            std::cout << "Calculating ground state eigenvector with CG..." << std::endl;
            
            vec_randomize(dim, v.data() + 2 * dim, seed);
            double accuracy;
            MKL_INT m = 0;
            if (matrix_free) {
                eigenvec_CG(dim, maxit, m, *this,  static_cast<T>(E0), accuracy,
                            v.data() + 2 * dim, v.data(), v.data() + dim, v.data() + 3 * dim);
            } else {
                eigenvec_CG(dim, maxit, m, HamMat, static_cast<T>(E0), accuracy,
                            v.data() + 2 * dim, v.data(), v.data() + dim, v.data() + 3 * dim);
            }
            assert(m >= 0 && m < maxit);
            assert(accuracy < lanczos_precision);
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
            std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
            std::cout << "CG steps:     " << m << std::endl;
            std::cout << "Accuracy:     " << accuracy << std::endl << std::endl;
            
            V0_done = true;
            nconv = 1;
            ckpt_lczsE0_updt(E0_done, V0_done, E1_done, V1_done);
        }
        
        // postpone writing down ground state eigenvector, if gap needed
        if (nev == 2 && ! E1_done) {
            start = std::chrono::system_clock::now();
            std::cout << "Calculating 1st excited state energy (simple Lanczos)..." << std::endl;
            vec_randomize(dim, v.data(), seed);
            auto alpha = dotc(dim, v.data() + 2 * dim, 1, v.data(), 1);          // (phi0, v0)
            axpy(dim, -alpha, v.data() + 2 * dim, 1, v.data(), 1);               // v0 -= alpha * phi0
            double rnorm = nrm2(dim, v.data(), 1);
            scal(dim, 1.0 / rnorm, v.data(), 1);                                 // normalize v0
            
            MKL_INT m = 0;
            if (matrix_free) {
                lanczos(0, maxit-1, maxit, m, dim, *this,  v.data(), hessenberg.data(), "sr_val1");
            } else {
                lanczos(0, maxit-1, maxit, m, dim, HamMat, v.data(), hessenberg.data(), "sr_val1");
            }
            assert(m < maxit);
            hess_eigen(hessenberg.data(), maxit, m, "sr", ritz, s);
            eigenvals.resize(2);
            eigenvals[1] = ritz[0];
            E1  = eigenvals[1];
            gap = eigenvals[1] - eigenvals[0];
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
            std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
            std::cout << "Lanczos steps: " << m << std::endl;
            std::cout << "Lanczos accuracy: " << std::abs(hessenberg[m] * s[m-1]) << std::endl;
            std::cout << "E1   = " << E1 << std::endl;
            std::cout << "gap  = " << gap << std::endl << std::endl;
            
            E1_done = true;
            ckpt_lczsE0_updt(E0_done, V0_done, E1_done, V1_done);
        }
        
        if (ncv == 1) {
            copy(dim, v.data() + 2 * dim, 1, v.data(), 1);                       // copy eigenvec to head of v
            v.resize(dim);
            swap(eigenvecs,v);
            // clean ckpt
            return;
        }
        
        if (! V1_done) {
            start = std::chrono::system_clock::now();
            std::cout << "Calculate 1st excited state eigenvector with CG..." << std::endl;
            
            v.resize(5*dim);
            vec_randomize(dim, v.data() + 3 * dim, seed + 7);
            double accuracy;
            MKL_INT m = 0;
            if (matrix_free) {
                eigenvec_CG(dim, maxit, m, *this,  static_cast<T>(E1), accuracy,
                            v.data() + 3 * dim, v.data(), v.data() + dim, v.data() + 4 * dim);
            } else {
                eigenvec_CG(dim, maxit, m, HamMat, static_cast<T>(E1), accuracy,
                            v.data() + 3 * dim, v.data(), v.data() + dim, v.data() + 4 * dim);
            }
            assert(m >= 0 && m <= maxit);
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
            std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
            std::cout << "CG steps:     " << m << std::endl;
            std::cout << "Accuracy:     " << accuracy << std::endl;
            
            V1_done = true;
            nconv = 2;
            copy(2*dim, v.data() + 2 * dim, 1, v.data(), 1);                     // copy eigenvec to head of v
            v.resize(2*dim);
            swap(eigenvecs,v);
            if (gap < lanczos_precision) {                                       // orthogonalize the two degenerate states
                std::cout << "Orthogonalizing degenerate ground states..." << std::endl;
                T alpha = dotc(dim, eigenvecs.data(), 1, eigenvecs.data() + dim, 1); // (v0,v1)
                axpy(dim, -alpha, eigenvecs.data(), 1, eigenvecs.data() + dim, 1);
                double rnorm = nrm2(dim, eigenvecs.data() + dim, 1);
                scal(dim, 1.0 / rnorm, eigenvecs.data() + dim, 1);
            }
            std::cout << std::endl;
            
            ckpt_lczsE0_updt(E0_done, V0_done, E1_done, V1_done);
        }
    }
    
    
    template <typename T>
    void model<T>::locate_E0_full(const MKL_INT &nev, const MKL_INT &ncv, MKL_INT maxit)
    {
        assert(nev > 0);
        assert(ncv > nev + 1);
        if (maxit <= 0) maxit = nev * 100; // arpack default
        sec_sym = 0;                       // work with dim_full
        
        std::cout << "Calculating ground state (full)..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<T> v0(dim_full[sec_mat], 1.0);
        eigenvals_full.resize(nev);
        eigenvecs_full.resize(dim_full[sec_mat] * nev);
        if (matrix_free) {
            iram(dim_full[sec_mat], *this, v0.data(), nev, ncv, maxit, "sr", nconv, eigenvals_full.data(), eigenvecs_full.data());
        } else {
            iram(dim_full[sec_mat], HamMat_csr_full[sec_mat], v0.data(), nev, ncv, maxit, "sr", nconv, eigenvals_full.data(), eigenvecs_full.data());
        }
        assert(nconv > 0);
        E0 = eigenvals_full[0];
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "E0   = " << E0 << std::endl;
        if (nconv > 1) {
            gap = eigenvals_full[1] - eigenvals_full[0];
            std::cout << "Gap  = " << gap << std::endl;
        }
    }
    
    template <typename T>
    void model<T>::locate_Emax_full(const MKL_INT &nev, const MKL_INT &ncv, MKL_INT maxit)
    {
        assert(ncv > nev + 1);
        if (maxit <= 0) maxit = nev * 100; // arpack default
        sec_sym = 0;                       // work with dim_full
        std::cout << "Calculating highest energy state (full)..." << std::endl;
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<T> v0(HamMat_csr_full[sec_mat].dimension(), 1.0);
        eigenvals_full.resize(nev);
        eigenvecs_full.resize(HamMat_csr_full[sec_mat].dimension() * nev);
        if (matrix_free) {
            iram(dim_full[sec_mat], *this, v0.data(), nev, ncv, maxit, "lr", nconv, eigenvals_full.data(), eigenvecs_full.data());
        } else {
            iram(dim_full[sec_mat], HamMat_csr_full[sec_mat], v0.data(), nev, ncv, maxit, "lr", nconv, eigenvals_full.data(), eigenvecs_full.data());
        }
        assert(nconv > 0);
        Emax = eigenvals_full[0];
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "Emax = " << Emax << std::endl;
    }
    
    template <typename T>
    void model<T>::locate_E0_repr(const MKL_INT &nev, const MKL_INT &ncv, MKL_INT maxit)
    {
        assert(ncv > nev + 1);
        assert(dim_repr[sec_mat] > 0);
        if (maxit <= 0) maxit = nev * 100; // arpack default
        sec_sym = 1;                       // work with dim_repr
        std::cout << "Calculating ground state (repr)..." << std::endl;
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        std::vector<std::complex<double>> v0(dim_repr[sec_mat], 1.0);
        eigenvals_repr.resize(nev);
        eigenvecs_repr.resize(dim_repr[sec_mat] * nev);
        
        if (matrix_free) {
            iram(dim_repr[sec_mat], *this, v0.data(), nev, ncv, maxit, "sr", nconv, eigenvals_repr.data(), eigenvecs_repr.data());
        } else {
            iram(dim_repr[sec_mat], HamMat_csr_repr[sec_mat], v0.data(), nev, ncv, maxit, "sr", nconv, eigenvals_repr.data(), eigenvecs_repr.data());
        }
        
        assert(nconv > 1);
        E0 = eigenvals_repr[0];
        gap = eigenvals_repr[1] - eigenvals_repr[0];
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "E0   = " << E0 << std::endl;
        std::cout << "Gap  = " << gap << std::endl;
    }
    
    template <typename T>
    void model<T>::locate_Emax_repr(const MKL_INT &nev, const MKL_INT &ncv, MKL_INT maxit)
    {
        assert(ncv > nev + 1);
        assert(dim_repr[sec_mat] > 0);
        if (maxit <= 0) maxit = nev * 100; // arpack default
        sec_sym = 1;                       // work with dim_repr
        std::cout << "Calculating highest energy state (repr)..." << std::endl;
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<std::complex<double>> v0(dim_repr[sec_mat], 1.0);
        eigenvals_repr.resize(nev);
        eigenvecs_repr.resize(dim_repr[sec_mat] * nev);
        
        if (matrix_free) {
            iram(dim_repr[sec_mat], *this, v0.data(), nev, ncv, maxit, "lr", nconv, eigenvals_repr.data(), eigenvecs_repr.data());
        } else {
            iram(dim_repr[sec_mat], HamMat_csr_repr[sec_mat], v0.data(), nev, ncv, maxit, "lr", nconv, eigenvals_repr.data(), eigenvecs_repr.data());
        }
        
        assert(nconv > 1);
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "Emax(maybe fake) = " << eigenvals_repr[0] << std::endl;
        Emax = eigenvals_repr[0];  // if we use a parameter extra in normalization calculation, we can know how many faked
    }
    
    template <typename T>
    void model<T>::moprXvec_full(const mopr<T> &lhs, const uint32_t &sec_old, const uint32_t &sec_new,
                                 const T* vec_old, T* vec_new)
    {
        // note: vec_new has size dim_full[sec_target]
        assert(dim_full[sec_old] > 0 && dim_full[sec_new] > 0);
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        
        int num_threads = 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) num_threads = omp_get_num_threads();
        }
        // prepare intermediates in advance
        std::vector<wavefunction<T>> intermediate_states(num_threads, {props});
        
        std::cout << "mopr * vec (s = " << sec_old << ", t = " << sec_new << ")... " << std::endl;
        for (MKL_INT j = 0; j < dim_full[sec_new]; j++) vec_new[j] = 0.0;
        
        #pragma omp parallel for schedule(dynamic,16)
        for (MKL_INT j = 0; j < dim_full[sec_old]; j++) {
            int tid = omp_get_thread_num();
            
            auto sj = vec_old[j];
            if (std::abs(sj) < lanczos_precision) continue;
            
            MKL_INT i;
            uint64_t i_a, i_b;
            std::vector<std::pair<MKL_INT, T>> values;
            for (auto it = lhs.mats.begin(); it != lhs.mats.end(); it++) {
                if (it->q_diagonal() && (sec_old == sec_new)) {
                    values.push_back(std::pair<MKL_INT, T>(j,sj * basis_full[sec_old][j].diagonal_operator(props,*it)));
                } else {
                    intermediate_states[tid].copy(basis_full[sec_old][j]);
                    oprXphi(*it, props, intermediate_states[tid]);
                    for (int cnt = 0; cnt < intermediate_states[tid].size(); cnt++) {
                        auto &ele = intermediate_states[tid][cnt];
                        if (Lin_Ja_full[sec_new].size() > 0 && Lin_Jb_full[sec_new].size() > 0) {
                            ele.first.label_sub(props, i_a, i_b);
                            i = Lin_Ja_full[sec_new][i_a] + Lin_Jb_full[sec_new][i_b];
                        } else {
                            i = binary_search<mbasis_elem,MKL_INT>(basis_full[sec_new], ele.first, 0, dim_full[sec_new]);
                        }
                        if (i < 0 || i >= dim_full[sec_new]) continue;
                        assert(basis_full[sec_new][i] == ele.first);
                        values.push_back(std::pair<MKL_INT, T>(i, sj * ele.second));
                    }
                }
            }
            #pragma omp critical
            {
                for (decltype(values.size()) cnt = 0; cnt < values.size(); cnt++)
                    vec_new[values[cnt].first] += values[cnt].second;
            }
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    
    template <typename T>
    void model<T>::moprXeigenvec_full(const mopr<T> &lhs, const uint32_t &sec_old, const uint32_t &sec_new,
                                      const MKL_INT &which_col, T* vec_new)
    {
        assert(which_col >= 0 && which_col < nconv);
        T* vec_old = eigenvecs_full.data() + dim_full[sec_old] * which_col;
        moprXvec_full(lhs, sec_old, sec_new, vec_old, vec_new);
    }
    
    
    template <typename T>
    T model<T>::measure_full(const mopr<T> &lhs, const uint32_t &sec_full, const MKL_INT &which_col)
    {
        assert(which_col >= 0 && which_col < nconv);
        MKL_INT base = dim_full[sec_full] * which_col;
        std::vector<T> vec_new(dim_full[sec_full]);
        moprXeigenvec_full(lhs, sec_full, sec_full, which_col, vec_new.data());
        return dotc(dim_full[sec_full], eigenvecs_full.data() + base, 1, vec_new.data(), 1);
    }
    
    
    template <typename T>
    T model<T>::measure_full(const std::vector<mopr<T>> &lhs, const std::vector<uint32_t> &sec_old_list, const MKL_INT &which_col)
    {
        assert(which_col >= 0 && which_col < nconv);
        assert(sec_old_list.size() > 0 && sec_old_list.size() == lhs.size());
        MKL_INT base = dim_full[sec_old_list[0]] * which_col;
        
        std::vector<uint32_t> sec_new_list(sec_old_list.size());
        for (decltype(sec_old_list.size()) j = 1; j < sec_old_list.size(); j++)sec_new_list[j-1] = sec_old_list[j];
        sec_new_list.back() = sec_old_list.front();
        
        std::vector<T> vec_new(dim_full[sec_new_list[0]]);
        moprXeigenvec_full(lhs[0], sec_old_list[0], sec_new_list[0], which_col, vec_new.data());
        std::vector<T> vec_temp;
        for (decltype(lhs.size()) j = 1; j < lhs.size(); j++) {
            vec_temp.resize(dim_full[sec_new_list[j]]);
            moprXvec_full(lhs[j], sec_old_list[j], sec_new_list[j], vec_new.data(), vec_temp.data());
            vec_new = vec_temp;
        }
        return dotc(dim_full[sec_old_list[0]], eigenvecs_full.data() + base, 1, vec_new.data(), 1);
    }
    
    
    template <typename T>
    void model<T>::moprXvec_repr(const mopr<T> &lhs, const uint32_t &sec_old, const uint32_t &sec_new,
                                 const T* vec_old, T* vec_new)
    {
        // note: vec_new has size dim_repr[sec_target]
        auto dim_latt = latt_parent.dimension();
        auto L        = latt_parent.Linear_size();
        bool bosonic  = q_bosonic(props);
        assert(dim_repr[sec_old] > 0 && dim_repr[sec_new] > 0);
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        int num_threads = 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) num_threads = omp_get_num_threads();
        }
        // prepare intermediates in advance
        std::vector<wavefunction<T>> intermediate_states(num_threads, {props});
        
        std::cout << "mopr * vec (s = " << sec_old << ", t = " << sec_new << ")... " << std::endl;
        for (MKL_INT j = 0; j < dim_repr[sec_new]; j++) vec_new[j] = 0.0;
        
        #pragma omp parallel for schedule(dynamic,16)
        for (MKL_INT j = 0; j < dim_repr[sec_old]; j++) {
            auto sj     = vec_old[j];
            double nu_j = norm_repr[sec_old][j];
            if (std::abs(sj) < lanczos_precision || std::abs(nu_j) < lanczos_precision) continue;
            
            int tid = omp_get_thread_num();
            
            std::vector<std::pair<MKL_INT, T>> values;
            for (auto it = lhs.mats.begin(); it != lhs.mats.end(); it++) {
                if (it->q_diagonal()) {                                                    // only momentum changes
                    double nu_i = norm_repr[sec_new][j];
                    if (std::abs(nu_i) > lanczos_precision)
                        values.push_back(std::pair<MKL_INT, T>(j, std::sqrt(nu_j/nu_i) * sj * basis_repr[sec_old][j].diagonal_operator(props,*it)));
                } else {
                    intermediate_states[tid].copy(basis_repr[sec_old][j]);
                    oprXphi(*it, props, intermediate_states[tid]);
                    uint64_t state_sub1_label, state_sub2_label;
                    std::vector<uint32_t> disp_i(dim_latt), disp_j(dim_latt);
                    std::vector<int> disp_i_int(dim_latt), disp_j_int(dim_latt);
                    int sgn;
                    mbasis_elem state_sub_new1, state_sub_new2, ra_z_Tj_rb;
                    
                    for (int cnt = 0; cnt < intermediate_states[tid].size(); cnt++) {
                        auto &ele_new = intermediate_states[tid][cnt];
                        ele_new.first.label_sub(props, state_sub1_label, state_sub2_label);
                        auto &state_rep1_label = belong2rep_sub[state_sub1_label];       // ra
                        auto &state_rep2_label = belong2rep_sub[state_sub2_label];       // rb
                        auto &ga               = belong2group_sub[state_rep1_label];     // ga
                        auto &gb               = belong2group_sub[state_rep2_label];     // gb
                        std::vector<uint64_t> pos_e{ga, gb};
                        pos_e.insert(pos_e.end(), dist2rep_sub[state_sub1_label].begin(), dist2rep_sub[state_sub1_label].end());
                        pos_e.insert(pos_e.end(), dist2rep_sub[state_sub2_label].begin(), dist2rep_sub[state_sub2_label].end());
                        if (state_rep1_label < state_rep2_label) {                          // ra < rb
                            disp_i = Weisse_e_lt.index(pos_e).first;
                            disp_j = Weisse_e_lt.index(pos_e).second;
                        } else if (state_rep2_label < state_rep1_label) {                   // ra > rb
                            disp_i = Weisse_e_gt.index(pos_e).first;
                            disp_j = Weisse_e_gt.index(pos_e).second;
                        } else {                                                            // ra == rb
                            disp_i = Weisse_e_eq.index(pos_e).first;
                            disp_j = Weisse_e_eq.index(pos_e).second;
                        }
                        for (uint32_t j = 0; j < disp_j.size(); j++) {
                            disp_i_int[j] = static_cast<int>(disp_i[j]);
                            disp_j_int[j] = static_cast<int>(disp_j[j]);
                        }
                        
                        if (state_rep2_label < state_rep1_label && dim_spec_involved) {
                            state_sub_new1 = basis_sub_repr[state_rep2_label];
                            state_sub_new2 = basis_sub_repr[state_rep1_label];
                        } else {
                            state_sub_new1 = basis_sub_repr[state_rep1_label];
                            state_sub_new2 = basis_sub_repr[state_rep2_label];
                        }
                        
                        state_sub_new2.translate(props_sub_b, latt_sub, disp_j_int, sgn);   // T_j |rb>
                        zipper_basis(props, props_sub_a, props_sub_b, state_sub_new1, state_sub_new2, ra_z_Tj_rb); // |ra> z T_j |rb>
                        MKL_INT i;
                        if (Lin_Ja_repr[sec_new].size() > 0 && Lin_Jb_repr[sec_new].size() > 0) {
                            uint64_t i_a = state_sub_new1.label(props_sub_a);               // use Lin Tables
                            uint64_t i_b = state_sub_new2.label(props_sub_b);
                            i = Lin_Ja_repr[sec_new][i_a] + Lin_Jb_repr[sec_new][i_b];
                        } else {
                            i = binary_search<mbasis_elem,MKL_INT>(basis_repr[sec_new], ra_z_Tj_rb, 0, dim_repr[sec_new]);
                        }
                        if (i < 0 || i >= dim_repr[sec_new]) continue;
                        assert(ra_z_Tj_rb == basis_repr[sec_new][i]);
                        double nu_i = norm_repr[sec_new][i];
                        if (std::abs(nu_i) < lanczos_precision) continue;
                        
                        double exp_coef = 0.0;
                        for (uint32_t d = 0; d < dim_latt; d++) {
                            if (trans_sym[d]) {
                                exp_coef += momenta[sec_new][d] * disp_i_int[d] / static_cast<double>(L[d]);
                            }
                        }
                        auto coef = std::sqrt(nu_j / nu_i) * sj * ele_new.second * std::exp(std::complex<double>(0.0, -2.0 * pi * exp_coef));
                        if (! bosonic) {
                            ra_z_Tj_rb.translate(props, latt_parent, disp_i_int, sgn);          // to get sgn
                            assert(ra_z_Tj_rb == ele_new.first);
                            if (sgn % 2 == 1) coef *= std::complex<double>(-1.0, 0.0);
                        }
                        values.push_back(std::pair<MKL_INT, T>(i, coef));
                    }
                }
            }
            #pragma omp critical
            {
                for (decltype(values.size()) cnt = 0; cnt < values.size(); cnt++)
                    vec_new[values[cnt].first] += values[cnt].second;
            }
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    
    template <typename T>
    void model<T>::moprXeigenvec_repr(const mopr<T> &lhs, const uint32_t &sec_old, const uint32_t &sec_new,
                                      const MKL_INT &which_col, T* vec_new)
    {
        assert(which_col >= 0 && which_col < nconv);
        T* vec_old = eigenvecs_repr.data() + dim_repr[sec_old] * which_col;
        moprXvec_repr(lhs, sec_old, sec_new, vec_old, vec_new);
    }
    
    
    template <typename T>
    T model<T>::measure_repr(const mopr<T> &lhs, const uint32_t &sec_repr, const MKL_INT &which_col)
    {
        double denominator = 1.0;
        auto L = latt_parent.Linear_size();
        std::vector<uint32_t> base;
        for (uint32_t d = 0; d < latt_parent.dimension(); d++) {
            if (trans_sym[d]) {
                denominator *= L[d];
                base.push_back(L[d]);
            } else {
                base.push_back(1);
            }
        }
        
        qbasis::mopr<T> opr_trans;                                               // O_t = (1/N) \sum_R T(R) O T(-R)
        std::vector<uint32_t> disp(base.size(),0);
        while (! dynamic_base_overflow(disp, base)) {
            std::vector<int> disp_int(base.size());
            for (uint32_t d = 0; d < disp.size(); d++) disp_int[d] = static_cast<int>(disp[d]);
            auto plan = latt_parent.translation_plan(disp_int);
            auto opr_temp = lhs;
            opr_temp.transform(plan);
            opr_trans += static_cast<T>(1.0/denominator) * opr_temp;
            disp = dynamic_base_plus1(disp, base);
        }
        opr_trans.simplify();
        
        std::vector<T> vec_new(dim_repr[sec_repr]);
        moprXeigenvec_repr(opr_trans, sec_repr, sec_repr, which_col, vec_new.data());
        return dotc(dim_repr[sec_repr], eigenvecs_repr.data() + dim_repr[sec_repr] * which_col, 1, vec_new.data(), 1);
    }
    
    
//     ---------------------------- deprecated ---------------------------------
//     ---------------------------- deprecated ---------------------------------
    
    template <typename T>
    void model<T>::basis_init_repr_deprecated(const lattice &latt, const std::vector<int> &momentum,
                                              const uint32_t &sec_full, const uint32_t &sec_repr)
    {
        latt_parent = latt;
        assert(latt_parent.dimension() == static_cast<uint32_t>(momentum.size()));
        assert(dim_full[sec_full] > 0 && dim_full[sec_full] == static_cast<MKL_INT>(basis_full[sec_full].size()));
        
        momenta[sec_repr] = momentum;
        
        check_translation();
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::cout << "Classifying basis_repr according to momentum (deprecated method): (";
        for (uint32_t j = 0; j < momentum.size(); j++) {
            if (trans_sym[j]) {
                std::cout << momentum[j] << "\t";
            } else {
                std::cout << "NA\t";
            }
        }
        std::cout << ")..." << std::endl;
        
        auto num_sub = latt_parent.num_sublattice();
        auto L = latt_parent.Linear_size();
        basis_belong_deprec[sec_repr].resize(dim_full[sec_full]);
        std::fill(basis_belong_deprec[sec_repr].begin(), basis_belong_deprec[sec_repr].end(), -1);
        basis_coeff_deprec[sec_repr].resize(dim_full[sec_full]);
        std::fill(basis_coeff_deprec[sec_repr].begin(), basis_coeff_deprec[sec_repr].end(), std::complex<double>(0.0,0.0));
        basis_repr_deprec[sec_repr].resize(0);
        
        for (MKL_INT i = 0; i < dim_full[sec_full]; i++) {
            if (basis_belong_deprec[sec_repr][i] != -1) continue;
            basis_belong_deprec[sec_repr][i] = i;
            basis_repr_deprec[sec_repr].push_back(i);
            basis_coeff_deprec[sec_repr][i] = std::complex<double>(1.0, 0.0);
            #pragma omp parallel for schedule(dynamic,1)
            for (uint32_t site = num_sub; site < latt_parent.total_sites(); site += num_sub) {
                std::vector<int> disp;
                int sub, sgn;
                latt_parent.site2coor(disp, sub, site);
                bool flag = false;
                for (uint32_t d = 0; d < latt_parent.dimension(); d++) {
                    if (!trans_sym[d] && disp[d] != 0) {
                        flag = true;
                        break;
                    }
                }
                if (flag) continue;            // such translation forbidden
                auto basis_temp = basis_full[sec_full][i];
                basis_temp.translate(props, latt_parent, disp, sgn);
                MKL_INT j;
                if (Lin_Ja_full[sec_full].size() > 0 && Lin_Jb_full[sec_full].size() > 0) {
                    uint64_t i_a, i_b;
                    basis_temp.label_sub(props, i_a, i_b);
                    j = Lin_Ja_full[sec_full][i_a] + Lin_Jb_full[sec_full][i_b];
                } else {
                    j = binary_search<mbasis_elem,MKL_INT>(basis_full[sec_full], basis_temp, 0, dim_full[sec_full]);
                }
                assert(basis_full[sec_full][j] == basis_temp);
                
                double exp_coef = 0.0;
                for (uint32_t d = 0; d < latt_parent.dimension(); d++) {
                    if (trans_sym[d]) {
                        exp_coef += momentum[d] * disp[d] / static_cast<double>(L[d]);
                    }
                }
                auto coef = std::exp(std::complex<double>(0.0, 2.0 * pi * exp_coef));
                if (sgn % 2 == 1) coef *= std::complex<double>(-1.0, 0.0);
                #pragma omp critical
                {
                    basis_belong_deprec[sec_repr][j] = i;
                    basis_coeff_deprec[sec_repr][j] += coef;
                }
            }
        }
        assert(is_sorted_norepeat(basis_repr_deprec[sec_repr]));
        if (dim_repr[sec_repr] > 0 && dim_repr[sec_repr] == static_cast<MKL_INT>(basis_repr[sec_repr].size())) {
            assert(dim_repr[sec_repr] == static_cast<MKL_INT>(basis_repr_deprec[sec_repr].size()));
        } else {
            dim_repr[sec_repr] = basis_repr_deprec[sec_repr].size();
        }
        std::cout << "dim_repr = " << dim_repr[sec_repr] << std::flush;
        
        MKL_INT extra = 0;
        for (MKL_INT j = 0; j < dim_repr[sec_repr]; j++) {
            if (std::abs(basis_coeff_deprec[sec_repr][basis_repr_deprec[sec_repr][j]]) < lanczos_precision) extra++;
        }
        std::cout << " - " << extra << " = " << (dim_repr[sec_repr] - extra) << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    
    template <typename T>
    void model<T>::generate_Ham_sparse_repr_deprecated(const bool &upper_triangle)
    {
        if (matrix_free) matrix_free = false;
        assert(dim_repr[sec_mat] > 0);
        int num_threads = 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) num_threads = omp_get_num_threads();
        }
        // prepare intermediates in advance
        std::vector<wavefunction<T>> intermediate_states(num_threads, {props});
        
        std::cout << "Generating LIL Hamiltonian Matrix (repr) (deprecated)..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        lil_mat<std::complex<double>> matrix_lil(dim_repr[sec_mat], upper_triangle);
        
        #pragma omp parallel for schedule(dynamic,16)
        for (MKL_INT i = 0; i < dim_repr[sec_mat]; i++) {
            int tid = omp_get_thread_num();
            
            auto repr_i = basis_repr_deprec[sec_mat][i];
            if (std::abs(basis_coeff_deprec[sec_mat][repr_i]) < lanczos_precision) {
                matrix_lil.add(i, i, static_cast<T>(fake_pos + static_cast<double>(i)/static_cast<double>(dim_repr[sec_mat])));
                continue;
            }
            // diagonal part:
            for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)
                matrix_lil.add(i, i, basis_full[sec_mat][repr_i].diagonal_operator(props,Ham_diag[cnt]));
            
            // non-diagonal part:
            for (auto it = Ham_off_diag.mats.begin(); it != Ham_off_diag.mats.end(); it++) {
                intermediate_states[tid].copy(basis_full[sec_mat][repr_i]);
                oprXphi(*it, props, intermediate_states[tid]);
                
                for (int cnt = 0; cnt < intermediate_states[tid].size(); cnt++) {
                    auto &ele_new = intermediate_states[tid][cnt];
                    MKL_INT state_j;
                    if (Lin_Ja_full[sec_mat].size() > 0 && Lin_Jb_full[sec_mat].size() > 0) {
                        uint64_t i_a, i_b;
                        ele_new.first.label_sub(props, i_a, i_b);
                        state_j = Lin_Ja_full[sec_mat][i_a] + Lin_Jb_full[sec_mat][i_b];
                    } else {
                        state_j = binary_search<mbasis_elem,MKL_INT>(basis_full[sec_mat], ele_new.first, 0, dim_full[sec_mat]);
                    }
                    if (state_j < 0 || state_j >= dim_full[sec_mat]) continue;
                    assert(state_j >= 0 && state_j < dim_full[sec_mat]);
                    auto repr_j = basis_belong_deprec[sec_mat][state_j];
                    if (std::abs(basis_coeff_deprec[sec_mat][repr_j]) < lanczos_precision) continue;
                    
                    //MKL_INT j = Lin_Ja_repr[sec_repr][i_a] + Lin_Jb_repr[sec_repr][i_b];
                    auto j = binary_search<MKL_INT,MKL_INT>(basis_repr_deprec[sec_mat], repr_j, 0, dim_repr[sec_mat]);  // < j |P'_k H | i > obtained
                    //if (j < 0 || j >= dim_repr[sec_repr] ) continue;
                    auto coeff = basis_coeff_deprec[sec_mat][state_j]/std::sqrt(std::real(basis_coeff_deprec[sec_mat][repr_i] * basis_coeff_deprec[sec_mat][repr_j]));
                    
                    if (upper_triangle) {
                        if (i <= j) matrix_lil.add(i, j, conjugate(ele_new.second) * coeff);
                    } else {
                        matrix_lil.add(i, j, conjugate(ele_new.second) * coeff);
                    }
                }
            }
        }
        HamMat_csr_repr[sec_mat] = csr_mat<std::complex<double>>(matrix_lil);
        std::cout << "Hamiltonian generated." << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    
    
    
    template <typename T>
    void model<T>::ckpt_lczsE0_init(bool &E0_done, bool &V0_done, bool &E1_done, bool &V1_done, std::vector<T> &v)
    {
        if (! enable_ckpt) return;
        fs::path outdir("out_Qckpt");
        if (fs::exists(outdir)) {
            if (! fs::is_directory(outdir)) {
                fs::remove_all(outdir);
                fs::create_directory(outdir);
            }
        } else {
            fs::create_directory(outdir);
        }
        
        MKL_INT dim     = (sec_sym == 0) ? dim_full[sec_mat] : dim_repr[sec_mat];
        auto &eigenvals = (sec_sym == 0) ? eigenvals_full : eigenvals_repr;
        auto &eigenvecs = (sec_sym == 0) ? eigenvecs_full : eigenvecs_repr;
        
        std::ofstream fout("out_Qckpt/log_lczs_E0_ckpt.txt", std::ios::out | std::ios::app);
        fout << std::setprecision(10);
        fout << std::endl << "Log start (ckpt_lczsE0_init): " << date_and_time() << std::endl;
        fout << "Initializing lczs_E0" << std::endl;
        
        auto filesize_ideal = 4 * sizeof(bool) + sizeof(MKL_INT) + 3 * sizeof(double);
        
        std::string filename0 = "out_Qckpt/lczs_E0_sym" + std::to_string(sec_sym) + "_sec" + std::to_string(sec_mat);
        if (sec_sym == 1) {
            filename0 += "_K";
            for (auto &k : momenta[sec_mat]) filename0 += std::to_string(k);
        }
        filename0 += ".Qckpt";
        std::string filename1 = filename0 + "1";
        std::string filename2 = filename0 + "2";
        
        if ((! fs::exists(fs::path(filename0)) && ! fs::exists(fs::path(filename1))) ||
            (fs::exists(fs::path(filename0)) && fs::file_size(fs::path(filename0)) != filesize_ideal)) {  // un-initialized
            fout << "Initializing checkpoint for lczs_E0." << std::endl;
            std::ofstream ftemp(filename0, std::ios::out | std::ios::binary);
            ftemp.write(reinterpret_cast<const char*>(&E0_done), sizeof(bool));
            ftemp.write(reinterpret_cast<const char*>(&V0_done), sizeof(bool));
            ftemp.write(reinterpret_cast<const char*>(&E1_done), sizeof(bool));
            ftemp.write(reinterpret_cast<const char*>(&V1_done), sizeof(bool));
            ftemp.write(reinterpret_cast<char*>(&nconv), sizeof(MKL_INT));
            ftemp.write(reinterpret_cast<char*>(&E0), sizeof(double));
            ftemp.write(reinterpret_cast<char*>(&E1), sizeof(double));
            ftemp.write(reinterpret_cast<char*>(&gap), sizeof(double));
            ftemp.close();
        } else {
            bool updating = (fs::exists(fs::path(filename1)) &&
                             fs::file_size(fs::path(filename1)) == filesize_ideal) ? true : false;
            fout << "Resuming from an interrupted update? " << updating << std::endl;
            if (updating && fs::exists(fs::path(filename2))) {
                fout << "New data finished writing." << std::endl;
                fs::remove(fs::path(filename0));
                fs::copy(fs::path(filename1), fs::path(filename0));
                ckpt_lanczos_clean();
                ckpt_CG_clean();
            }
            fs::remove(fs::path(filename1));
            fs::remove(fs::path(filename2));
            fout << "loading data from " << filename0 << std::endl;
            
            std::ifstream ftemp(filename0, std::ios::in | std::ios::binary);
            ftemp.read(reinterpret_cast<char*>(&E0_done), sizeof(bool));
            fout << "E0_done: " << E0_done << std::endl;
            ftemp.read(reinterpret_cast<char*>(&V0_done), sizeof(bool));
            fout << "V0_done: " << V0_done << std::endl;
            ftemp.read(reinterpret_cast<char*>(&E1_done), sizeof(bool));
            fout << "E1_done: " << E1_done << std::endl;
            ftemp.read(reinterpret_cast<char*>(&V1_done), sizeof(bool));
            fout << "V1_done: " << V1_done << std::endl;
            ftemp.read(reinterpret_cast<char*>(&nconv), sizeof(MKL_INT));
            fout << "nconv: " << nconv << std::endl;
            ftemp.read(reinterpret_cast<char*>(&E0), sizeof(double));
            if (E0_done) fout << "E0: " << E0 << std::endl;
            ftemp.read(reinterpret_cast<char*>(&E1), sizeof(double));
            if (E1_done) fout << "E1: " << E1 << std::endl;
            ftemp.read(reinterpret_cast<char*>(&gap), sizeof(double));
            if (E1_done) fout << "gap: " << gap << std::endl;
            ftemp.close();
        }
        
        if (E1_done) {
            eigenvals.resize(2);
            eigenvals[0] = E0;
            eigenvals[1] = E1;
        } else if (E0_done) {
            eigenvals.resize(1);
            eigenvals[0] = E0;
        }
        
        if (E0_done && V0_done && (! V1_done)) {
            fout << "Reading eigenvec0 from disk." << std::endl;
            std::string filename = "out_Qckpt/eigenvec0_sym" + std::to_string(sec_sym) + "_sec" + std::to_string(sec_mat);
            if (sec_sym == 1) {
                filename += "_K";
                for (auto &k : momenta[sec_mat]) filename += std::to_string(k);
            }
            filename += ".dat";
            assert(fs::exists(fs::path(filename)));
            assert(static_cast<MKL_INT>(v.size()) >= 3 * dim);
            vec_disk_read(filename, dim, v.data() + 2 * dim);
        } else if (V1_done) {
            fout << "Reading eigenvec0/1 from disk." << std::endl;
            std::string flnm0 = "out_Qckpt/eigenvec0_sym" + std::to_string(sec_sym) + "_sec" + std::to_string(sec_mat);
            std::string flnm1 = "out_Qckpt/eigenvec1_sym" + std::to_string(sec_sym) + "_sec" + std::to_string(sec_mat);
            if (sec_sym == 1) {
                flnm0 += "_K";
                flnm1 += "_K";
                for (auto &k : momenta[sec_mat]) {
                    flnm0 += std::to_string(k);
                    flnm1 += std::to_string(k);
                }
            }
            flnm0 += ".dat";
            flnm1 += ".dat";
            assert(fs::exists(fs::path(flnm0)));
            assert(fs::exists(fs::path(flnm1)));
            v.resize(0);
            v.shrink_to_fit();
            
            eigenvecs.resize(2*dim);
            vec_disk_read(flnm0, dim, eigenvecs.data());
            vec_disk_read(flnm1, dim, eigenvecs.data() + dim);
        }
        
        fout << "Log end (ckpt_lczsE0_init): " << date_and_time() << std::endl << std::endl;
        fout.close();
    }
    
    template <typename T>
    void model<T>::ckpt_lczsE0_updt(const bool &E0_done, const bool &V0_done, const bool &E1_done, const bool &V1_done)
    {
        if (! enable_ckpt) return;
        
        MKL_INT dim     = (sec_sym == 0) ? dim_full[sec_mat] : dim_repr[sec_mat];
        auto &eigenvecs = (sec_sym == 0) ? eigenvecs_full : eigenvecs_repr;
        
        std::string filename0 = "out_Qckpt/lczs_E0_sym" + std::to_string(sec_sym) + "_sec" + std::to_string(sec_mat);
        if (sec_sym == 1) {
            filename0 += "_K";
            for (auto &k : momenta[sec_mat]) filename0 += std::to_string(k);
        }
        filename0 += ".Qckpt";
        std::string filename1 = filename0 + "1";
        std::string filename2 = filename0 + "2";
        assert(fs::exists(fs::path(filename0)));
        
        std::ofstream fout("out_Qckpt/log_lczs_E0_ckpt.txt", std::ios::out | std::ios::app);
        fout << std::setprecision(10);
        fout << std::endl << "Log start (ckpt_lczsE0_updt): " << date_and_time() << std::endl;
        fout << "Updating lczs_E0" << std::endl;
        
        fs::remove(fs::path(filename1));
        fs::remove(fs::path(filename2));
        
        std::ofstream ftemp(filename1, std::ios::out | std::ios::binary);
        fout << "E0_done: " << E0_done << std::endl;
        ftemp.write(reinterpret_cast<const char*>(&E0_done), sizeof(bool));
        fout << "V0_done: " << V0_done << std::endl;
        ftemp.write(reinterpret_cast<const char*>(&V0_done), sizeof(bool));
        fout << "E1_done: " << E1_done << std::endl;
        ftemp.write(reinterpret_cast<const char*>(&E1_done), sizeof(bool));
        fout << "V1_done: " << V1_done << std::endl;
        ftemp.write(reinterpret_cast<const char*>(&V1_done), sizeof(bool));
        fout << "nconv: " << nconv << std::endl;
        ftemp.write(reinterpret_cast<char*>(&nconv), sizeof(MKL_INT));
        if (E0_done) fout << "E0: " << E0 << std::endl;
        ftemp.write(reinterpret_cast<char*>(&E0), sizeof(double));
        if (E1_done) fout << "E1: " << E0 << std::endl;
        ftemp.write(reinterpret_cast<char*>(&E1), sizeof(double));
        if (E1_done) fout << "gap: " << gap << std::endl;
        ftemp.write(reinterpret_cast<char*>(&gap), sizeof(double));
        ftemp.close();
        
        std::string flnm0 = "out_Qckpt/eigenvec0_sym" + std::to_string(sec_sym) + "_sec" + std::to_string(sec_mat);
        std::string flnm1 = "out_Qckpt/eigenvec1_sym" + std::to_string(sec_sym) + "_sec" + std::to_string(sec_mat);
        if (sec_sym == 1) {
            flnm0 += "_K";
            flnm1 += "_K";
            for (auto &k : momenta[sec_mat]) {
                flnm0 += std::to_string(k);
                flnm1 += std::to_string(k);
            }
        }
        flnm0 += ".dat";
        flnm1 += ".dat";
        
        if (E0_done && V0_done && (! E1_done) && (! V1_done)) {                  // record eigenvec0
            for (auto &p : fs::directory_iterator("out_Qckpt"))
            {
                if (std::regex_match(p.path().filename().string(), std::regex("CG_V[[:digit:]]+\\.dat")))
                {
                    fs::remove(fs::path(flnm0));
                    fs::copy(fs::path(p.path()), fs::path(flnm0));
                }
            }
        }
        if (V1_done) {                                                           // record eigenvec0/1
            assert(static_cast<MKL_INT>(eigenvecs.size()) == 2 * dim);
            if (! fs::exists(fs::path(flnm0))) vec_disk_write(flnm0, dim, eigenvecs.data());
            vec_disk_write(flnm1, dim, eigenvecs.data() + dim);
        }
        
        // before/after this point, have to use old/new data
        fs::copy(fs::path(filename1), fs::path(filename2));
        
        fs::remove(fs::path(filename0));                                         // at this moment, filename0 disappear
        fs::copy(fs::path(filename1), fs::path(filename0));
        ckpt_lanczos_clean();
        ckpt_CG_clean();
        fs::remove(fs::path(filename1));
        fs::remove(fs::path(filename2));
        fout << "Log end (ckpt_lczsE0_updt): " << date_and_time() << std::endl << std::endl;
        fout.close();
    }
    
    
    // Explicit instantiation
    //template class model<double>;
    template class model<std::complex<double>>;

}
