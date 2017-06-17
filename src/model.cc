#include <iostream>
//#include <fstream>
//#include <iomanip>
#include <climits>
#include "qbasis.h"

namespace qbasis {
    template <typename T>
    void model<T>::check_translation(const lattice &latt)
    {
        std::cout << "Checking translational symmetry." << std::endl;
        std::cout << "In the future replace this check with serious stuff!" << std::endl;
        sym_translation.clear();
        auto bc = latt.boundary();
        for (uint32_t j = 0; j < latt.dimension(); j++) {
            if (bc[j] == "pbc" || bc[j] == "PBC") {
                sym_translation.push_back(true);
            } else {
                sym_translation.push_back(false);
            }
        }
    }
    
    
    // need further optimization! (for example, special treatment of dilute limit; special treatment of quantum numbers; quick sort of sign)
    template <typename T>
    void model<T>::enumerate_basis_full(const lattice &latt,
                                        std::initializer_list<mopr<std::complex<double>>> conserve_lst,
                                        std::initializer_list<double> val_lst,
                                        const bool &use_translation)
    {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) {
                std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
                std::cout << "Number of procs   = " << omp_get_num_procs() << std::endl << std::endl;
            }
        }
        
        uint32_t n_sites = latt.total_sites();
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
        
        // checking if reaching code capability
        MKL_INT mkl_int_max = LLONG_MAX;
        if (mkl_int_max != LLONG_MAX) {
            mkl_int_max = INT_MAX;
            //std::cout << "int_max = " << INT_MAX << std::endl;
            assert(mkl_int_max == INT_MAX);
            std::cout << "Using 32-bit integers." << std::endl;
        } else {
            std::cout << "Using 64-bit integers." << std::endl;
        }
        assert(mkl_int_max > 0);
        uint32_t site_max = log(mkl_int_max) / log(dim_local);
        std::cout << "Capability of current code for current model: maximal " << site_max << " sites. (in the ideal case of infinite memory available)" << std::endl << std::endl;
        assert(n_sites < site_max);
        
        std::cout << "Enumerating the full basis with " << val_lst.size() << " conserved quantum numbers..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        
        // Hilbert space size if without any symmetry
        MKL_INT dim_total = int_pow<MKL_INT,MKL_INT>(static_cast<MKL_INT>(dim_local), static_cast<MKL_INT>(n_sites));
        assert(dim_total > 0); // prevent overflow
        std::cout << "Hilbert space size **if** without any symmetry: " << dim_total << std::endl;
        
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
        
        dim_full = 0;
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
        std::cout << "Hilbert space size with symmetry: " << dim_full << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl << std::endl;
        start = end;
        
        // pick the fruits
        basis_full.clear();
        std::cout << "memory performance not optimal in the following line, think about improvements." << std::endl;
        basis_full.reserve(dim_full);
        std::cout << "Moving temporary basis (" << basis_temp.size() << " pieces) to basis_full... ";
        for (auto it = basis_temp.begin(); it != basis_temp.end(); it++) {
            basis_full.insert(basis_full.end(), std::make_move_iterator(it->begin()), std::make_move_iterator(it->end()));
            it->erase(it->begin(), it->end()); // should I?
            it->shrink_to_fit();
        }
        assert(dim_full == static_cast<MKL_INT>(basis_full.size()));
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << elapsed_seconds.count() << "s." << std::endl << std::endl;
        
        sort_basis_full();
    }
    
    template <typename T>
    void model<T>::sort_basis_full()
    {
        bool sorted = true;
        for (MKL_INT j = 0; j < dim_full - 1; j++) {
            assert(basis_full[j] != basis_full[j+1]);
            if (basis_full[j+1] < basis_full[j]) {
                sorted = false;
                break;
            }
        }
        if (! sorted) {
            std::chrono::time_point<std::chrono::system_clock> start, end;
            start = std::chrono::system_clock::now();
            std::cout << "sorting basis(full)... ";
            std::sort(basis_full.begin(), basis_full.end());
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::cout << elapsed_seconds.count() << "s." << std::endl << std::endl;
        }
    }
    
    template <typename T>
    void model<T>::basis_init_repr(const std::vector<int> &momentum, const lattice &latt)
    {
        assert(latt.dimension() == static_cast<uint32_t>(momentum.size()));
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::cout << "Classifying states according to momentum: (";
        for (uint32_t j = 0; j < momentum.size(); j++) {
            if (latt.boundary()[j] == "pbc" || latt.boundary()[j] == "PBC") {
                std::cout << momentum[j] << "\t";
            } else {
                std::cout << "NA\t";
            }
        }
        std::cout << ")..." << std::endl;
        
        std::cout << "-------- if all dims obc, let's stop here (to be implemented) -------" << std::endl;
        
        auto num_sub = latt.num_sublattice();
        auto L = latt.Linear_size();
        basis_belong.resize(dim_full);
        std::fill(basis_belong.begin(), basis_belong.end(), -1);
        basis_coeff.resize(dim_full);
        std::fill(basis_coeff.begin(), basis_coeff.end(), std::complex<double>(0.0,0.0));
        for (MKL_INT i = 0; i < dim_full; i++) {
            if (basis_belong[i] != -1) continue;
            basis_belong[i] = i;
            basis_coeff[i] = std::complex<double>(1.0, 0.0);
            #pragma omp parallel for schedule(dynamic,1)
            for (uint32_t site = num_sub; site < latt.total_sites(); site += num_sub) {
                std::vector<int> disp;
                int sub, sgn;
                latt.site2coor(disp, sub, site);
                bool flag = false;
                for (uint32_t d = 0; d < latt.dimension(); d++) {
                    if (latt.boundary()[d] != "pbc" && latt.boundary()[d] != "PBC" && disp[d] != 0) {
                        flag = true;
                        break;
                    }
                }
                if (flag) continue;            // such translation forbidden by boundary condition
                auto basis_temp = basis_full[i];
                basis_temp.translate(props, latt, disp, sgn);
                auto j = binary_search<mbasis_elem,MKL_INT>(basis_full, basis_temp, 0, dim_full);
                double exp_coef = 0.0;
                for (uint32_t d = 0; d < latt.dimension(); d++) {
                    if (latt.boundary()[d] == "pbc" || latt.boundary()[d] == "PBC") {
                        exp_coef += momentum[d] * disp[d] / static_cast<double>(L[d]);
                    }
                }
                auto coef = std::exp(std::complex<double>(0.0, 2.0 * pi * exp_coef));
                if (sgn % 2 == 1) coef *= std::complex<double>(-1.0, 0.0);
                #pragma omp critical
                {
                    basis_belong[j] = i;
                    basis_coeff[j] += coef;
                }
            }
        }
        
        std::list<MKL_INT> temp_repr;
        temp_repr.push_back(0);
        dim_repr = 1;
        for (MKL_INT j = 1; j < dim_full; j++) {
            if (basis_belong[j] > temp_repr.back()) {
                dim_repr++;
                temp_repr.push_back(basis_belong[j]);
            }
        }
        MKL_INT redundant = 0;
        auto it = temp_repr.begin();
        while (it != temp_repr.end()) {
            if (std::abs(basis_coeff[*it]) < opr_precision) {
                it = temp_repr.erase(it);
                redundant++;
                dim_repr--;
            } else {
                //std::cout << std::abs(std::imag(basis_coeff[*it])) << std::endl;
                assert(std::abs(std::imag(basis_coeff[*it])) < opr_precision);
                it++;
            }
        }
        assert(dim_repr == static_cast<MKL_INT>(temp_repr.size()));
        if (redundant > 0) std::cout << redundant << " redundant reprs removed." << std::endl;
        basis_repr.resize(dim_repr);
        std::copy(temp_repr.begin(), temp_repr.end(), basis_repr.begin());
        assert(is_sorted_norepeat(basis_repr));
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    
    template <typename T>
    void model<T>::generate_Ham_sparse_full(const bool &upper_triangle)
    {
        std::cout << "Generating Hamiltonian Matrix..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        lil_mat<T> matrix_lil(dim_full, upper_triangle);
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_full; i++) {
            for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)                                       // diagonal part:
                matrix_lil.add(i, i, basis_full[i].diagonal_operator(props, Ham_diag[cnt]));
            qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_full[i], props);  // non-diagonal part:
            for (decltype(intermediate_state.size()) cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
                MKL_INT j = binary_search<mbasis_elem,MKL_INT>(basis_full, ele_new.first, 0, dim_full);       // < j | H | i > obtained
                assert(j != -1);
                if (upper_triangle) {
                    if (i <= j) matrix_lil.add(i, j, conjugate(ele_new.second));
                } else {
                    matrix_lil.add(i, j, conjugate(ele_new.second));
                }
            }
        }
        HamMat_csr_full = csr_mat<T>(matrix_lil);
        std::cout << "Hamiltonian generated." << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    
    template <typename T>
    void model<T>::generate_Ham_sparse_repr(const bool &upper_triangle)
    {
        if (dim_repr < 1) {
            std::cout << "dim_repr = " << dim_repr << "!!!" << std::endl;
            return;
        }
        std::cout << "Generating Hamiltonian Matrix..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        lil_mat<std::complex<double>> matrix_lil(dim_repr, upper_triangle);
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_repr; i++) {
            auto repr_i = basis_repr[i];
            for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)                                            // diagonal part:
                matrix_lil.add(i, i, basis_full[repr_i].diagonal_operator(props,Ham_diag[cnt]));
            qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_full[repr_i], props);  // non-diagonal part:
            for (uint32_t cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
                MKL_INT state_j = binary_search<mbasis_elem,MKL_INT>(basis_full, ele_new.first, 0, dim_full);
                assert(state_j != -1);
                auto repr_j = basis_belong[state_j];
                auto j = binary_search<MKL_INT,MKL_INT>(basis_repr, repr_j, 0, dim_repr);                 // < j |P'_k H | i > obtained
                if (j < 0) continue;
                auto coeff = basis_coeff[state_j]/std::sqrt(std::real(basis_coeff[repr_i] * basis_coeff[repr_j]));
                if (upper_triangle) {
                    if (i <= j) matrix_lil.add(i, j, conjugate(ele_new.second) * coeff);
                } else {
                    matrix_lil.add(i, j, conjugate(ele_new.second) * coeff);
                }
            }
        }
        HamMat_csr_repr = csr_mat<std::complex<double>>(matrix_lil);
        std::cout << "Hamiltonian generated." << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    template <typename T>
    std::vector<std::complex<double>> model<T>::to_dense()
    {
        assert(false);
//        if (! latt_syms.empty()) {
//            generate_Ham_sparse_repr();
//            return HamMat_csr_repr.to_dense();
//        } else {
//            generate_Ham_sparse_full();
//            auto temp = HamMat_csr_full.to_dense();
//            std::vector<std::complex<double>> res(temp.size());
//            for (decltype(temp.size()) j = 0; j < temp.size(); j++) res[j] = std::complex<double>(temp[j]);
//            return res;
//        }
    }
    
    template <typename T>
    void model<T>::MultMv(const T *x, T *y) const
    {
        for (MKL_INT j = 0; j < dim_full; j++) y[j] = static_cast<T>(0.0);
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_full; i++) {
            std::vector<std::pair<MKL_INT, T>> mat_free{std::pair<MKL_INT, T>(i,static_cast<T>(0.0))};
            for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)
                mat_free[0].second += basis_full[i].diagonal_operator(props, Ham_diag[cnt]);
            qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_full[i], props);
            intermediate_state.simplify();
            for (decltype(intermediate_state.size()) cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
                MKL_INT j = binary_search<mbasis_elem,MKL_INT>(basis_full, ele_new.first, 0, dim_full);       // < j | H | i > obtained
                assert(j != -1);
                if (std::abs(ele_new.second) > opr_precision)
                    mat_free.emplace_back(j, conjugate(ele_new.second));
            }
            for (auto it = mat_free.begin(); it < mat_free.end(); it++)
                y[i] += (x[it->first] * it->second);
        }
    }
    
    template <typename T>
    void model<T>::MultMv(T *x, T *y)
    {
        for (MKL_INT j = 0; j < dim_full; j++) y[j] = static_cast<T>(0.0);
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_full; i++) {
            std::vector<std::pair<MKL_INT, T>> mat_free{std::pair<MKL_INT, T>(i,static_cast<T>(0.0))};
            for (uint32_t cnt = 0; cnt < Ham_diag.size(); cnt++)
                mat_free[0].second += basis_full[i].diagonal_operator(props, Ham_diag[cnt]);
            qbasis::wavefunction<T> intermediate_state = oprXphi(Ham_off_diag, basis_full[i], props);
            intermediate_state.simplify();
            for (decltype(intermediate_state.size()) cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
                MKL_INT j = binary_search<mbasis_elem,MKL_INT>(basis_full, ele_new.first, 0, dim_full);       // < j | H | i > obtained
                assert(j != -1);
                if (std::abs(ele_new.second) > opr_precision)
                    mat_free.emplace_back(j, conjugate(ele_new.second));
            }
            for (auto it = mat_free.begin(); it < mat_free.end(); it++)
                y[i] += (x[it->first] * it->second);
        }
    }
    
    
    template <typename T>
    void model<T>::locate_E0_full(const MKL_INT &nev, const MKL_INT &ncv, const bool &matrix_free)
    {
        assert(ncv > nev + 1);
        std::cout << "Calculating ground state..." << std::endl;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) {
                std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
                std::cout << "Number of procs   = " << omp_get_num_procs() << std::endl << std::endl;
            }
        }
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<T> v0(dim_full, 1.0);
        eigenvals_full.resize(nev);
        eigenvecs_full.resize(dim_full * nev);
        if (matrix_free) {
            iram(dim_full, *this, v0.data(), nev, ncv, nconv, "sr", eigenvals_full.data(), eigenvecs_full.data());
        } else {
            iram(dim_full, HamMat_csr_full, v0.data(), nev, ncv, nconv, "sr", eigenvals_full.data(), eigenvecs_full.data());
        }
        assert(nconv > 1);
        E0 = eigenvals_full[0];
        gap = eigenvals_full[1] - eigenvals_full[0];
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "E0   = " << E0 << std::endl;
        std::cout << "Gap  = " << gap << std::endl;
    }
    
    template <typename T>
    void model<T>::locate_E0_repr(const MKL_INT &nev, const MKL_INT &ncv)
    {
        assert(ncv > nev + 1);
        std::cout << "Calculating ground state in the subspace..." << std::endl;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) {
                std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
                std::cout << "Number of procs   = " << omp_get_num_procs() << std::endl << std::endl;
            }
        }
        if (dim_repr < 1) {
            std::cout << "dim_repr = " << dim_repr << "!!!" << std::endl;
            return;
        }
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<std::complex<double>> v0(HamMat_csr_repr.dimension(), 1.0);
        eigenvals_repr.resize(nev);
        eigenvecs_repr.resize(HamMat_csr_repr.dimension() * nev);
        qbasis::iram(dim_repr, HamMat_csr_repr, v0.data(), nev, ncv, nconv, "sr", eigenvals_repr.data(), eigenvecs_repr.data());
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
    void model<T>::locate_Emax_full(const MKL_INT &nev, const MKL_INT &ncv)
    {
        assert(ncv > nev + 1);
        std::cout << "Calculating highest energy state..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<T> v0(HamMat_csr_full.dimension(), 1.0);
        eigenvals_full.resize(nev);
        eigenvecs_full.resize(HamMat_csr_full.dimension() * nev);
        qbasis::iram(dim_full, HamMat_csr_full, v0.data(), nev, ncv, nconv, "lr", eigenvals_full.data(), eigenvecs_full.data());
        assert(nconv > 0);
        Emax = eigenvals_full[0];
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "Emax = " << Emax << std::endl;
    }
    
    template <typename T>
    void model<T>::locate_Emax_repr(const MKL_INT &nev, const MKL_INT &ncv)
    {
        assert(ncv > nev + 1);
        std::cout << "Calculating highest energy state in the subspace..." << std::endl;
        if (dim_repr < 1) {
            std::cout << "dim_repr = " << dim_repr << "!!!" << std::endl;
            return;
        }
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<std::complex<double>> v0(HamMat_csr_repr.dimension(), 1.0);
        eigenvals_repr.resize(nev);
        eigenvecs_repr.resize(HamMat_csr_repr.dimension() * nev);
        qbasis::iram(dim_repr, HamMat_csr_repr, v0.data(), nev, ncv, nconv, "lr", eigenvals_repr.data(), eigenvecs_repr.data());
        assert(nconv > 0);
        Emax = eigenvals_repr[0];
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "Emax = " << Emax << std::endl;
    }
    
    template <typename T>
    void model<T>::moprXeigenvec(const mopr<T> &lhs, T* vec_new, const MKL_INT &which_col)
    {
        assert(which_col >= 0 && which_col < nconv);
        for (MKL_INT j = 0; j < HamMat_csr_full.dimension(); j++) vec_new[j] = 0.0;
//        // leave these lines for a while, for openmp debugging purpose
//        static MKL_INT debug_flag = 0;
//        std::vector<std::list<MKL_INT>> jobid_list(100);
//        std::vector<std::list<double>> job_start(100), job_start_wait(100), job_finish_wait(100);
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        if (HamMat_csr_full.dimension() == dim_full) {
            MKL_INT base = dim_full * which_col;
            
            #pragma omp parallel for schedule(dynamic,16)
            for (MKL_INT j = 0; j < dim_full; j++) {
                if (std::abs(eigenvecs_full[base + j]) < lanczos_precision) continue;
//                std::chrono::time_point<std::chrono::system_clock> enter_time, start_wait, finish_wait;
//                enter_time = std::chrono::system_clock::now();
                std::vector<std::pair<MKL_INT, T>> values;
                for (uint32_t cnt_opr = 0; cnt_opr < lhs.size(); cnt_opr++) {
                    auto &A = lhs[cnt_opr];
                    auto temp = eigenvecs_full[base + j];
                    if (A.q_diagonal()) {
                        values.push_back(std::pair<MKL_INT, T>(j,temp * basis_full[j].diagonal_operator(props,A)));
                    } else {
                        auto intermediate_state = oprXphi(A, basis_full[j], props);
                        for (uint32_t cnt = 0; cnt < intermediate_state.size(); cnt++) {
                            auto &ele = intermediate_state[cnt];
                            values.push_back(std::pair<MKL_INT, T>(binary_search<mbasis_elem,MKL_INT>(basis_full, ele.first, 0, dim_full), temp * ele.second));
                        }
                    }
                }
//                // previously hope the sort helps the speed of the critical region,
//                // however the overhead is too much
//                std::sort(values.begin(), values.end(),
//                          [](const std::pair<MKL_INT, T> &a, const std::pair<MKL_INT, T> &b){ return a.first < b.first; });
//                start_wait = std::chrono::system_clock::now();
                #pragma omp critical
                {
                    for (decltype(values.size()) cnt = 0; cnt < values.size(); cnt++)
                        vec_new[values[cnt].first] += values[cnt].second;
//                    finish_wait = std::chrono::system_clock::now();
//                    auto id = omp_get_thread_num();
//                    std::chrono::duration<double> elapsed_seconds;
//                    jobid_list[id].push_back(j);
//                    elapsed_seconds = enter_time - start;
//                    job_start[id].push_back(elapsed_seconds.count());
//                    elapsed_seconds = start_wait - start;
//                    job_start_wait[id].push_back(elapsed_seconds.count());
//                    elapsed_seconds = finish_wait - start;
//                    job_finish_wait[id].push_back(elapsed_seconds.count());
                }
            }
            
        } else {
            std::cout << "not implemented yet" << std::endl;
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        
//        #pragma omp parallel
//        {
//            int tid = omp_get_thread_num();
//            if (tid == 0) {
//                if (! debug_flag) {
//                    for (MKL_INT j = 0; j < omp_get_num_threads(); j++) {
//                        std::string output_name = "thread_"+std::to_string(j)+".dat";
//                        std::ofstream fout(output_name, std::ios::out | std::ios::app);
//                        auto it0 = jobid_list[j].begin();
//                        auto it1 = job_start[j].begin();
//                        auto it2 = job_start_wait[j].begin();
//                        auto it3 = job_finish_wait[j].begin();
//                        while(it0 != jobid_list[j].end()){
//                            fout << std::setw(30) << *it0 << std::setw(30) << *it1
//                            << std::setw(30) << *it2 << std::setw(30) << *it3 << std::endl;
//                            it0++; it1++; it2++; it3++;
//                        }
//                        fout.close();
//                    }
//                    debug_flag = 1;
//                }
//            }
//        }
        
    }
    
    template <typename T>
    T model<T>::measure(const mopr<T> &lhs, const MKL_INT &which_col)
    {
        assert(which_col >= 0 && which_col < nconv);
        if (HamMat_csr_full.dimension() == dim_full) {
            MKL_INT base = dim_full * which_col;
            std::vector<T> vec_new(dim_full);
            moprXeigenvec(lhs, vec_new.data(), which_col);
            return dotc(dim_full, eigenvecs_full.data() + base, 1, vec_new.data(), 1);
        } else {
            std::cout << "not implemented yet" << std::endl;
            return static_cast<T>(0.0);
        }
    }
    
    template <typename T>
    T model<T>::measure(const mopr<T> &lhs1, const mopr<T> &lhs2, const MKL_INT &which_col)
    {
        assert(which_col >= 0 && which_col < nconv);
        if (HamMat_csr_full.dimension() == dim_full) {
            std::vector<T> vec_new1(dim_full);
            std::vector<T> vec_new2(dim_full);
            moprXeigenvec(lhs1, vec_new1.data(), which_col);
            moprXeigenvec(lhs2, vec_new2.data(), which_col);
            return dotc(dim_full, vec_new1.data(), 1, vec_new2.data(), 1);
        } else {
            std::cout << "not implemented yet" << std::endl;
            return static_cast<T>(0.0);
        }
    }
    
    
    // Explicit instantiation
    //template class model<double>;
    template class model<std::complex<double>>;

}
