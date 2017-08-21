#include <iostream>
#include <iomanip>
#include <random>
#include "qbasis.h"
#include "graph.h"

namespace qbasis {
    template <typename T>
    model<T>::model(const double &fake_pos_, const double &fake_incr_):
                    matrix_free(true), nconv(0),
                    sec_mat(0),
                    dim_full({0,0}), dim_repr({0,0}),
                    fake_pos(fake_pos_), fake_incr(fake_incr_)
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
        classify_Weisse_tables(props, props_sub, basis_sub_full, basis_sub_repr, latt, trans_sym,
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
        
        std::cout << "Generating LIL Hamiltonian matrix (full)..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        lil_mat<T> matrix_lil(dim_full[sec_mat], upper_triangle);
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_full[sec_mat]; i++) {
            for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)                                       // diagonal part:
                matrix_lil.add(i, i, basis_full[sec_mat][i].diagonal_operator(props, Ham_diag[cnt]));
            qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_full[sec_mat][i], props);  // non-diagonal part:
            for (decltype(intermediate_state.size()) cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
                if (std::abs(ele_new.second) < machine_prec) continue;
                MKL_INT j;
                if (Lin_Ja_full[sec_mat].size() > 0 && Lin_Jb_full[sec_mat].size() > 0) {
                    uint64_t i_a, i_b;
                    ele_new.first.label_sub(props, i_a, i_b);
                    j = Lin_Ja_full[sec_mat][i_a] + Lin_Jb_full[sec_mat][i_b];
                } else {
                    j = binary_search<mbasis_elem,MKL_INT>(basis_full[sec_mat], ele_new.first, 0, dim_full[sec_mat]);
                }
                assert(j >= 0 && j < dim_full[sec_mat]);
                if (upper_triangle) {
                    if (i <= j) matrix_lil.add(i, j, conjugate(ele_new.second));
                } else {
                    matrix_lil.add(i, j, conjugate(ele_new.second));
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
        assert(dim_repr[sec_mat] > 0);
        
        if (dim_spec_involved) {
            assert(Weisse_w_gt.size() == 0);
        } else {
            assert(Weisse_w_lt.size() == Weisse_w_gt.size());
        }
        
        std::cout << "Generating LIL Hamiltonian Matrix (repr)..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        auto dim_latt = latt_parent.dimension();
        auto L = latt_parent.Linear_size();
        lil_mat<std::complex<double>> matrix_lil(dim_repr[sec_mat], upper_triangle);
        
        double faked = fake_pos;
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_repr[sec_mat]; i++) {
            double nu_i = norm_repr[sec_mat][i];                                // normalization factor for repr i
            if (std::abs(nu_i) < lanczos_precision) {
                matrix_lil.add(i, i, static_cast<T>(faked));
                #pragma omp atomic
                faked += fake_incr;
                continue;
            }
            for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)                 // diagonal part:
                matrix_lil.add(i, i, basis_repr[sec_mat][i].diagonal_operator(props,Ham_diag[cnt]));
            
            qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_repr[sec_mat][i], props);
            uint64_t state_sub1_label, state_sub2_label;
            std::vector<uint32_t> disp_i(dim_latt), disp_j(dim_latt);
            std::vector<int> disp_i_int(dim_latt), disp_j_int(dim_latt);
            int sgn;
            mbasis_elem state_sub_new1, state_sub_new2, ra_z_Tj_rb;
            
            for (uint32_t cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
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
                MKL_INT j;
                if (Lin_Ja_repr[sec_mat].size() > 0 && Lin_Jb_repr[sec_mat].size() > 0) {
                    uint64_t i_a = state_sub_new1.label(props_sub_a);               // use Lin Tables
                    uint64_t i_b = state_sub_new2.label(props_sub_b);
                    j = Lin_Ja_repr[sec_mat][i_a] + Lin_Jb_repr[sec_mat][i_b];
                } else {
                    j = binary_search<mbasis_elem,MKL_INT>(basis_repr[sec_mat], ra_z_Tj_rb, 0, dim_repr[sec_mat]);
                }
                assert(j >= 0 && j < dim_repr[sec_mat]);
                assert(ra_z_Tj_rb == basis_repr[sec_mat][j]);
                
                double nu_j = norm_repr[sec_mat][j];
                if (std::abs(nu_j) < lanczos_precision) continue;
                ra_z_Tj_rb.translate(props, latt_parent, disp_i_int, sgn);         // remove this line in future
                assert(ra_z_Tj_rb == ele_new.first);
                
                double exp_coef = 0.0;
                for (uint32_t d = 0; d < latt_parent.dimension(); d++) {
                    if (trans_sym[d]) {
                        exp_coef += momenta[sec_mat][d] * disp_i_int[d] / static_cast<double>(L[d]);
                    }
                }
                auto coef = std::sqrt(nu_i / nu_j) * conjugate(ele_new.second) * std::exp(std::complex<double>(0.0, 2.0 * pi * exp_coef));
                if (sgn % 2 == 1) coef *= std::complex<double>(-1.0, 0.0);

                if (upper_triangle) {
                    if (i <= j) matrix_lil.add(i, j, coef);
                } else {
                    matrix_lil.add(i, j, coef);
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
        std::cout << "Fall back to use matrix explicitly." << std::endl;
        if (sec_sym == 0) {
            generate_Ham_sparse_full();
            return HamMat_csr_full[sec_mat].to_dense();
        } else {
            generate_Ham_sparse_repr();
            return HamMat_csr_full[sec_mat].to_dense();
        }
    }
    
    template <typename T>
    void model<T>::MultMv(const T *x, T *y) const
    {
        assert(matrix_free);
        std::cout << "*" << std::flush;
        if (sec_sym == 0) {
            #pragma omp parallel for schedule(dynamic,1)
            for (MKL_INT i = 0; i < dim_full[sec_mat]; i++) {
                y[i] = static_cast<T>(0.0);
                if (std::abs(x[i]) > machine_prec) {
                    for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)
                        y[i] += x[i] * basis_full[sec_mat][i].diagonal_operator(props, Ham_diag[cnt]);
                }
                qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_full[sec_mat][i], props);
                for (decltype(intermediate_state.size()) cnt = 0; cnt < intermediate_state.size(); cnt++) {
                    auto &ele_new = intermediate_state[cnt];
                    if (std::abs(ele_new.second) < machine_prec) continue;
                    MKL_INT j;
                    if (Lin_Ja_full[sec_mat].size() > 0 && Lin_Jb_full[sec_mat].size() > 0) {
                        uint64_t i_a, i_b;
                        ele_new.first.label_sub(props, i_a, i_b);
                        j = Lin_Ja_full[sec_mat][i_a] + Lin_Jb_full[sec_mat][i_b];
                    } else {
                        j = binary_search<mbasis_elem,MKL_INT>(basis_full[sec_mat], ele_new.first, 0, dim_full[sec_mat]);
                    }
                    assert(j >= 0 && j < dim_full[sec_mat]);
                    if (std::abs(x[j]) > machine_prec) y[i] += (x[j] * conjugate(ele_new.second));
                }
            }
        } else {
            auto dim_latt = latt_parent.dimension();
            auto L        = latt_parent.Linear_size();
            
            double faked = fake_pos;
            #pragma omp parallel for schedule(dynamic,1)
            for (MKL_INT i = 0; i < dim_repr[sec_mat]; i++) {
                y[i] = static_cast<T>(0.0);
                
                double nu_i = norm_repr[sec_mat][i];                             // normalization factor for repr i
                if (std::abs(nu_i) < lanczos_precision) {
                    y[i] += x[i] * static_cast<T>(faked);
                    #pragma omp atomic
                    faked += fake_incr;
                    continue;
                }
                
                if (std::abs(x[i]) > machine_prec) {
                    for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)          // diagonal part:
                        y[i] += x[i] * basis_repr[sec_mat][i].diagonal_operator(props,Ham_diag[cnt]);
                }
                
                qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_repr[sec_mat][i], props);
                uint64_t state_sub1_label, state_sub2_label;
                std::vector<uint32_t> disp_i(dim_latt), disp_j(dim_latt);
                std::vector<int> disp_i_int(dim_latt), disp_j_int(dim_latt);
                int sgn;
                mbasis_elem state_sub_new1, state_sub_new2, ra_z_Tj_rb;
                
                for (uint32_t cnt = 0; cnt < intermediate_state.size(); cnt++) {
                    auto &ele_new = intermediate_state[cnt];
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
                    MKL_INT j;
                    if (Lin_Ja_repr[sec_mat].size() > 0 && Lin_Jb_repr[sec_mat].size() > 0) {
                        uint64_t i_a = state_sub_new1.label(props_sub_a);               // use Lin Tables
                        uint64_t i_b = state_sub_new2.label(props_sub_b);
                        j = Lin_Ja_repr[sec_mat][i_a] + Lin_Jb_repr[sec_mat][i_b];
                    } else {
                        j = binary_search<mbasis_elem,MKL_INT>(basis_repr[sec_mat], ra_z_Tj_rb, 0, dim_repr[sec_mat]);
                    }
                    assert(j >= 0 && j < dim_repr[sec_mat]);
                    assert(ra_z_Tj_rb == basis_repr[sec_mat][j]);
                    if (std::abs(x[j]) < machine_prec) continue;
                    
                    double nu_j = norm_repr[sec_mat][j];
                    if (std::abs(nu_j) < lanczos_precision) continue;
                    ra_z_Tj_rb.translate(props, latt_parent, disp_i_int, sgn);         // remove this line in future
                    assert(ra_z_Tj_rb == ele_new.first);
                    
                    double exp_coef = 0.0;
                    for (uint32_t d = 0; d < latt_parent.dimension(); d++) {
                        if (trans_sym[d]) {
                            exp_coef += momenta[sec_mat][d] * disp_i_int[d] / static_cast<double>(L[d]);
                        }
                    }
                    auto coef = std::sqrt(nu_i / nu_j) * conjugate(ele_new.second) * std::exp(std::complex<double>(0.0, 2.0 * pi * exp_coef));
                    if (sgn % 2 == 1) coef *= std::complex<double>(-1.0, 0.0);
                    y[i] += (x[j] * coef);
                }
            }
        }
    }
    
    template <typename T>
    void model<T>::MultMv(T *x, T *y)
    {
        assert(matrix_free);
        std::cout << "*" << std::flush;
        if (sec_sym == 0) {
            #pragma omp parallel for schedule(dynamic,1)
            for (MKL_INT i = 0; i < dim_full[sec_mat]; i++) {
                y[i] = static_cast<T>(0.0);
                if (std::abs(x[i]) > machine_prec) {
                    for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)
                        y[i] += x[i] * basis_full[sec_mat][i].diagonal_operator(props, Ham_diag[cnt]);
                }
                qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_full[sec_mat][i], props);
                for (decltype(intermediate_state.size()) cnt = 0; cnt < intermediate_state.size(); cnt++) {
                    auto &ele_new = intermediate_state[cnt];
                    if (std::abs(ele_new.second) < machine_prec) continue;
                    MKL_INT j;
                    if (Lin_Ja_full[sec_mat].size() > 0 && Lin_Jb_full[sec_mat].size() > 0) {
                        uint64_t i_a, i_b;
                        ele_new.first.label_sub(props, i_a, i_b);
                        j = Lin_Ja_full[sec_mat][i_a] + Lin_Jb_full[sec_mat][i_b];
                    } else {
                        j = binary_search<mbasis_elem,MKL_INT>(basis_full[sec_mat], ele_new.first, 0, dim_full[sec_mat]);
                    }
                    assert(j >= 0 && j < dim_full[sec_mat]);
                    if (std::abs(x[j]) > machine_prec) y[i] += (x[j] * conjugate(ele_new.second));
                }
            }
        } else {
            auto dim_latt = latt_parent.dimension();
            auto L        = latt_parent.Linear_size();
            
            double faked = fake_pos;
            #pragma omp parallel for schedule(dynamic,1)
            for (MKL_INT i = 0; i < dim_repr[sec_mat]; i++) {
                y[i] = static_cast<T>(0.0);
                
                double nu_i = norm_repr[sec_mat][i];                             // normalization factor for repr i
                if (std::abs(nu_i) < lanczos_precision) {
                    y[i] += x[i] * static_cast<T>(faked + fake_incr);
                    #pragma omp atomic
                    faked += fake_incr;
                    continue;
                }
                
                if (std::abs(x[i]) > machine_prec) {
                    for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)          // diagonal part:
                        y[i] += x[i] * basis_repr[sec_mat][i].diagonal_operator(props,Ham_diag[cnt]);
                }
                
                qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_repr[sec_mat][i], props);
                uint64_t state_sub1_label, state_sub2_label;
                std::vector<uint32_t> disp_i(dim_latt), disp_j(dim_latt);
                std::vector<int> disp_i_int(dim_latt), disp_j_int(dim_latt);
                int sgn;
                mbasis_elem state_sub_new1, state_sub_new2, ra_z_Tj_rb;
                
                for (uint32_t cnt = 0; cnt < intermediate_state.size(); cnt++) {
                    auto &ele_new = intermediate_state[cnt];
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
                    MKL_INT j;
                    if (Lin_Ja_repr[sec_mat].size() > 0 && Lin_Jb_repr[sec_mat].size() > 0) {
                        uint64_t i_a = state_sub_new1.label(props_sub_a);               // use Lin Tables
                        uint64_t i_b = state_sub_new2.label(props_sub_b);
                        j = Lin_Ja_repr[sec_mat][i_a] + Lin_Jb_repr[sec_mat][i_b];
                    } else {
                        j = binary_search<mbasis_elem,MKL_INT>(basis_repr[sec_mat], ra_z_Tj_rb, 0, dim_repr[sec_mat]);
                    }
                    assert(j >= 0 && j < dim_repr[sec_mat]);
                    assert(ra_z_Tj_rb == basis_repr[sec_mat][j]);
                    if (std::abs(x[j]) < machine_prec) continue;
                    
                    double nu_j = norm_repr[sec_mat][j];
                    if (std::abs(nu_j) < lanczos_precision) continue;
                    ra_z_Tj_rb.translate(props, latt_parent, disp_i_int, sgn);         // remove this line in future
                    assert(ra_z_Tj_rb == ele_new.first);
                    
                    double exp_coef = 0.0;
                    for (uint32_t d = 0; d < latt_parent.dimension(); d++) {
                        if (trans_sym[d]) {
                            exp_coef += momenta[sec_mat][d] * disp_i_int[d] / static_cast<double>(L[d]);
                        }
                    }
                    auto coef = std::sqrt(nu_i / nu_j) * conjugate(ele_new.second) * std::exp(std::complex<double>(0.0, 2.0 * pi * exp_coef));
                    if (sgn % 2 == 1) coef *= std::complex<double>(-1.0, 0.0);
                    y[i] += (x[j] * coef);
                }
            }
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
    void model<T>::locate_E0_full_lanczos()
    {
        assert(false);
        sec_sym = 0;                       // work with dim_full
        std::cout << "Calculating ground state (full, with simple Lanczos)..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-1.0,1.0);
        std::vector<T> resid(dim_full[sec_mat]), v(dim_full[sec_mat]*3);
        for (MKL_INT j = 0; j < dim_full[sec_mat]; j++) resid[j] = static_cast<T>(distribution(generator));
        double rnorm = nrm2(dim_full[sec_mat], resid.data(), 1);
        scal(dim_full[sec_mat], 1.0 / rnorm, resid.data(), 1);
        MKL_INT ldh = 2000;                                     // at most 2000 steps
        std::vector<double> hessenberg(ldh, 0.0), ritz(ldh), e(ldh-1);
        
        MKL_INT total_steps = 0;
        MKL_INT step = 6;
        E0 = 1.0e10;
        std::vector<std::pair<double, MKL_INT>> eigenvals(ldh);
        eigenvals[0].first = 1.0e9;
        while (std::abs(E0 - eigenvals[0].first) > lanczos_precision && total_steps < ldh) {
            if (eigenvals[0].first < E0) {
                E0 = eigenvals[0].first;
            }
            if (matrix_free) {
                lanczos(0, step, dim_full[sec_mat], *this, rnorm, resid.data(), v.data(), hessenberg.data(), 2000, false);
            } else {
                lanczos(0, step, dim_full[sec_mat], HamMat_csr_full[sec_mat], rnorm, resid.data(), v.data(), hessenberg.data(), 2000, false);
            }
            total_steps += step;
            copy(total_steps, hessenberg.data() + ldh, 1, ritz.data(), 1);
            copy(total_steps-1, hessenberg.data() + 1, 1, e.data(), 1);
            int info = sterf(total_steps, ritz.data(), e.data());
            assert(info == 0);
            for (decltype(eigenvals.size()) j = 0; j < eigenvals.size(); j++) {
                eigenvals[j].first = ritz[j];
                eigenvals[j].second = static_cast<MKL_INT>(j);
            }
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return a.first < b.first; });
            std::cout << "Lanczos steps: " << total_steps << std::endl;
            std::cout << "Ritz values: "
                      << std::setw(25) << eigenvals[0].first << std::setw(25) << eigenvals[1].first
                      << std::setw(25) << eigenvals[2].first << std::setw(25) << eigenvals[3].first
                      << std::setw(25) << eigenvals[4].first << std::endl;
        }
        assert(total_steps < ldh);
        
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "E0   = " << E0 << std::endl;
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
        if (maxit <= 0) maxit = nev * 100; // arpack default
        sec_sym = 1;                       // work with dim_repr
        std::cout << "Calculating ground state (repr)..." << std::endl;
        
        if (dim_repr[sec_mat] < 1) {
            std::cout << "dim_repr = " << dim_repr[sec_mat] << "!!!" << std::endl;
            return;
        }
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
        if (maxit <= 0) maxit = nev * 100; // arpack default
        sec_sym = 1;                       // work with dim_repr
        std::cout << "Calculating highest energy state (repr)..." << std::endl;
        if (dim_repr[sec_mat] < 1) {
            std::cout << "dim_repr = " << dim_repr[sec_mat] << "!!!" << std::endl;
            return;
        }
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
    void model<T>::moprXvec_full(const mopr<T> &lhs, const T* vec_old, T* vec_new,
                                 const uint32_t &sec_source, const uint32_t &sec_target)
    {
        // note: vec_new has size dim_full[sec_target]
        assert(dim_full[sec_source] > 0 && dim_full[sec_target] > 0);
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        std::cout << "mopr * vec (s = " << sec_source << ", t = " << sec_target << ")... " << std::endl;
        for (MKL_INT j = 0; j < dim_full[sec_target]; j++) vec_new[j] = 0.0;
        
        #pragma omp parallel for schedule(dynamic,16)
        for (MKL_INT j = 0; j < dim_full[sec_source]; j++) {
            if (std::abs(vec_old[j]) < lanczos_precision) continue;
            
            std::vector<std::pair<MKL_INT, T>> values;
            for (uint32_t cnt_opr = 0; cnt_opr < lhs.size(); cnt_opr++) {
                auto &A = lhs[cnt_opr];
                auto sj = vec_old[j];
                if (A.q_diagonal() && (sec_source == sec_target)) {
                    values.push_back(std::pair<MKL_INT, T>(j,sj * basis_full[sec_source][j].diagonal_operator(props,A)));
                } else {
                    auto intermediate_state = oprXphi(A, basis_full[sec_source][j], props);
                    for (uint32_t cnt = 0; cnt < intermediate_state.size(); cnt++) {
                        auto &ele = intermediate_state[cnt];
                        
                        MKL_INT i;
                        if (Lin_Ja_full[sec_target].size() > 0 && Lin_Jb_full[sec_target].size() > 0) {
                            uint64_t i_a, i_b;
                            ele.first.label_sub(props, i_a, i_b);
                            i = Lin_Ja_full[sec_target][i_a] + Lin_Jb_full[sec_target][i_b];
                        } else {
                            i = binary_search<mbasis_elem,MKL_INT>(basis_full[sec_target], ele.first, 0, dim_full[sec_target]);
                        }
                        assert(i >= 0 && i < dim_full[sec_target]);
                        assert(basis_full[sec_target][i] == ele.first);
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
    void model<T>::moprXeigenvec_full(const mopr<T> &lhs, T* vec_new, const MKL_INT &which_col,
                                      const uint32_t &sec_source, const uint32_t &sec_target)
    {
        assert(which_col >= 0 && which_col < nconv);
        T* vec_old = eigenvecs_full.data() + dim_full[sec_source] * which_col;
        moprXvec_full(lhs, vec_old, vec_new, sec_source, sec_target);
    }
    
    
    template <typename T>
    T model<T>::measure_full(const mopr<T> &lhs, const MKL_INT &which_col, const uint32_t &sec_full)
    {
        assert(which_col >= 0 && which_col < nconv);
        MKL_INT base = dim_full[sec_full] * which_col;
        std::vector<T> vec_new(dim_full[sec_full]);
        moprXeigenvec_full(lhs, vec_new.data(), which_col, sec_full, sec_full);
        return dotc(dim_full[sec_full], eigenvecs_full.data() + base, 1, vec_new.data(), 1);
    }
    
    
    template <typename T>
    T model<T>::measure_full(const std::vector<mopr<T>> &lhs, const std::vector<uint32_t> &sec_source_list, const MKL_INT &which_col)
    {
        assert(which_col >= 0 && which_col < nconv);
        assert(sec_source_list.size() > 0 && sec_source_list.size() == lhs.size());
        MKL_INT base = dim_full[sec_source_list[0]] * which_col;
        
        std::vector<uint32_t> sec_target_list(sec_source_list.size());
        for (decltype(sec_source_list.size()) j = 1; j < sec_source_list.size(); j++)sec_target_list[j-1] = sec_source_list[j];
        sec_target_list.back() = sec_source_list.front();
        
        std::vector<T> vec_new(dim_full[sec_target_list[0]]);
        moprXeigenvec_full(lhs[0], vec_new.data(), sec_source_list[0], sec_target_list[0]);
        std::vector<T> vec_temp;
        for (decltype(lhs.size()) j = 1; j < lhs.size(); j++) {
            vec_temp.resize(dim_full[sec_target_list[j]]);
            moprXvec_full(lhs[j], vec_new.data(), vec_temp.data(), sec_source_list[j], sec_target_list[j]);
            vec_new = vec_temp;
        }
        return dotc(dim_full[sec_source_list[0]], eigenvecs_full.data() + base, 1, vec_new.data(), 1);
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
            assert(dim_repr[sec_repr] == basis_repr_deprec[sec_repr].size());
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
        
        std::cout << "Generating LIL Hamiltonian Matrix (repr) (deprecated)..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        lil_mat<std::complex<double>> matrix_lil(dim_repr[sec_mat], upper_triangle);
        
        double faked = fake_pos;
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_repr[sec_mat]; i++) {
            auto repr_i = basis_repr_deprec[sec_mat][i];
            if (std::abs(basis_coeff_deprec[sec_mat][repr_i]) < lanczos_precision) {
                matrix_lil.add(i, i, static_cast<T>(faked));
                #pragma omp atomic
                faked += fake_incr;
                continue;
            }
            for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)                                                      // diagonal part:
                matrix_lil.add(i, i, basis_full[sec_mat][repr_i].diagonal_operator(props,Ham_diag[cnt]));
            qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_full[sec_mat][repr_i], props);  // non-diagonal part:
            for (uint32_t cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
                MKL_INT state_j;
                if (Lin_Ja_full[sec_mat].size() > 0 && Lin_Jb_full[sec_mat].size() > 0) {
                    uint64_t i_a, i_b;
                    ele_new.first.label_sub(props, i_a, i_b);
                    state_j = Lin_Ja_full[sec_mat][i_a] + Lin_Jb_full[sec_mat][i_b];
                } else {
                    state_j = binary_search<mbasis_elem,MKL_INT>(basis_full[sec_mat], ele_new.first, 0, dim_full[sec_mat]);
                }
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
        HamMat_csr_repr[sec_mat] = csr_mat<std::complex<double>>(matrix_lil);
        std::cout << "Hamiltonian generated." << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    
    // Explicit instantiation
    //template class model<double>;
    template class model<std::complex<double>>;

}
