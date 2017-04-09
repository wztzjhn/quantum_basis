#include <iostream>
//#include <fstream>
//#include <iomanip>
#include "qbasis.h"

namespace qbasis {

    template <typename T>
    void model<T>::enumerate_basis_full()
    {
        assert(dim_full > 0);
        basis_full[0].reset();
        for (MKL_INT j = 1; j < dim_full; j++) {
            basis_full[j] = basis_full[j-1];
            basis_full[j].increment();
        }
        
        sort_basis_full();
    }
    
    // need further optimization!
    template <typename T>
    void model<T>::enumerate_basis_full_conserve(const MKL_INT &n_sites, std::initializer_list<std::string> lst,
                                                 std::initializer_list<mopr<std::complex<double>>> conserve_lst,
                                                 std::initializer_list<double> val_lst)
    {
        assert(conserve_lst.size() == val_lst.size());
        std::list<qbasis::mbasis_elem> basis_temp;
        auto GS = mbasis_elem(n_sites,lst);
        GS.reset();
        auto state_new = GS;
        
        MKL_INT cnt = 0;
        while (! state_new.q_maximized()) {
            bool flag = true;
            auto it_opr = conserve_lst.begin();
            auto it_val = val_lst.begin();
            while (it_opr != conserve_lst.end()) {
                auto temp = state_new.diagonal_operator(*it_opr);
                if (std::abs(temp - *it_val) >= 1e-5) {
                    flag = false;
                    break;
                }
                it_opr++;
                it_val++;
            }
            if (flag) basis_temp.push_back(state_new);
            state_new.increment();
            cnt++;
        }
        
        bool flag = true;
        auto it_opr = conserve_lst.begin();
        auto it_val = val_lst.begin();
        while (it_opr != conserve_lst.end()) {
            auto temp = state_new.diagonal_operator(*it_opr);
            if (std::abs(temp - *it_val) >= 1e-5) {
                flag = false;
                break;
            }
            it_opr++;
            it_val++;
        }
        if (flag) basis_temp.push_back(state_new);
        
        dim_full = basis_temp.size();
        basis_full.clear();
        for (auto & ele : basis_temp) basis_full.push_back(ele);
        
        sort_basis_full();
    }
    
    template <typename T>
    void model<T>::sort_basis_full()
    {
        std::cout << "checking if basis sorted..." << std::endl;
        bool sorted = true;
        for (MKL_INT j = 0; j < dim_full - 1; j++) {
            assert(basis_full[j] != basis_full[j+1]);
            if (basis_full[j+1] < basis_full[j]) {
                sorted = false;
                break;
            }
        }
        if (! sorted) {
            std::cout << "sorting basis..." << std::endl;
            std::sort(basis_full.begin(), basis_full.end());
            std::cout << "sorting finished." << std::endl;
        } else {
            std::cout << "yes it is already sorted." << std::endl;
        }
    }
    
    template <typename T>
    void model<T>::basis_repr_init(const std::vector<MKL_INT> &momentum, const lattice &latt)
    {
        assert(latt.dimension() == static_cast<MKL_INT>(momentum.size()));
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::cout << "Classifying states according to momentum: (";
        for (MKL_INT j = 0; j < static_cast<MKL_INT>(momentum.size()); j++) {
            if (latt.boundary()[j] == "pbc" || latt.boundary()[j] == "PBC") {
                std::cout << momentum[j] << "\t";
            } else {
                std::cout << "NA\t";
            }
        }
        std::cout << ")..." << std::endl;
        
        MKL_INT num_sub = latt.num_sublattice();
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
            for (MKL_INT site = num_sub; site < latt.total_sites(); site += num_sub) {
                std::vector<MKL_INT> disp;
                MKL_INT sub, sgn;
                latt.site2coor(disp, sub, site);
                bool flag = false;
                for (MKL_INT d = 0; d < latt.dimension(); d++) {
                    if (latt.boundary()[d] != "pbc" && latt.boundary()[d] != "PBC" && disp[d] != 0) {
                        flag = true;
                        break;
                    }
                }
                if (flag) continue;            // such translation forbidden by boundary condition
                auto basis_temp = basis_full[i];
                basis_temp.translate(latt, disp, sgn);
                auto j = binary_search(basis_full, basis_temp, 0, dim_full);
                double exp_coef = 0.0;
                for (MKL_INT d = 0; d < latt.dimension(); d++) {
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
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) {
                std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
                std::cout << "Number of procs   = " << omp_get_num_procs() << std::endl;
            }
        }
        std::cout << "Generating Hamiltonian Matrix..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        lil_mat<T> matrix_lil(dim_full, upper_triangle);
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_full; i++) {
            for (MKL_INT cnt = 0; cnt < Ham_diag.size(); cnt++)                        // diagonal part:
                matrix_lil.add(i, i, basis_full[i].diagonal_operator(Ham_diag[cnt]));
            qbasis::wavefunction<T> intermediate_state = Ham_off_diag * basis_full[i];  // non-diagonal part:
            for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
                MKL_INT j = binary_search(basis_full, ele_new.first, 0, dim_full);       // < j | H | i > obtained
                assert(j != -1);
                if (upper_triangle) {
                    if (i <= j) matrix_lil.add(i, j, conjugate(ele_new.second));
                } else {
                    matrix_lil.add(i, j, conjugate(ele_new.second));
                }
                
            }
        }
        HamMat_csr_full = csr_mat<T>(matrix_lil);
        matrix_lil.destroy();
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
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) {
                std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
                std::cout << "Number of procs   = " << omp_get_num_procs() << std::endl;
            }
        }
        std::cout << "Generating Hamiltonian Matrix..." << std::endl;
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        lil_mat<std::complex<double>> matrix_lil(dim_repr, upper_triangle);
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_repr; i++) {
            auto repr_i = basis_repr[i];
            for (MKL_INT cnt = 0; cnt < Ham_diag.size(); cnt++)                             // diagonal part:
                matrix_lil.add(i, i, basis_full[repr_i].diagonal_operator(Ham_diag[cnt]));
            
            qbasis::wavefunction<T> intermediate_state = Ham_off_diag * basis_full[repr_i];  // non-diagonal part:
            for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
                MKL_INT state_j = binary_search(basis_full, ele_new.first, 0, dim_full);
                assert(state_j != -1);
                auto repr_j = basis_belong[state_j];
                auto j = binary_search(basis_repr, repr_j, 0, dim_repr);                    // < j |P'_k H | i > obtained
                if (j < 0) continue;
//                assert(basis_repr[j] == repr_j);
//                assert(j >= 0);
                auto coeff = basis_coeff[state_j]/std::sqrt(std::real(basis_coeff[repr_i] * basis_coeff[repr_j]));
                if (upper_triangle) {
                    if (i <= j) matrix_lil.add(i, j, conjugate(ele_new.second) * coeff);
                } else {
                    matrix_lil.add(i, j, conjugate(ele_new.second) * coeff);
                }
            }
        }
        HamMat_csr_repr = csr_mat<std::complex<double>>(matrix_lil);
        matrix_lil.destroy();
        std::cout << "Hamiltonian generated." << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    
    template <typename T>
    void model<T>::locate_E0_full(const MKL_INT &nev, const MKL_INT &ncv)
    {
        assert(ncv > nev + 1);
        std::cout << "Calculating ground state..." << std::endl;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) {
                std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
                std::cout << "Number of procs   = " << omp_get_num_procs() << std::endl;
            }
        }
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<T> v0(HamMat_csr_full.dimension(), 1.0);
        eigenvals_full.resize(nev);
        eigenvecs_full.resize(HamMat_csr_full.dimension() * nev);
        qbasis::iram(HamMat_csr_full, v0.data(), nev, ncv, nconv, "sr", eigenvals_full.data(), eigenvecs_full.data());
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
        if (dim_repr < 1) {
            std::cout << "dim_repr = " << dim_repr << "!!!" << std::endl;
            return;
        }
        assert(ncv > nev + 1);
        std::cout << "Calculating ground state..." << std::endl;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) {
                std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
                std::cout << "Number of procs   = " << omp_get_num_procs() << std::endl;
            }
        }
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<std::complex<double>> v0(HamMat_csr_repr.dimension(), 1.0);
        eigenvals_repr.resize(nev);
        eigenvecs_repr.resize(HamMat_csr_repr.dimension() * nev);
        qbasis::iram(HamMat_csr_repr, v0.data(), nev, ncv, nconv, "sr", eigenvals_repr.data(), eigenvecs_repr.data());
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
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) {
                std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
                std::cout << "Number of procs   = " << omp_get_num_procs() << std::endl;
            }
        }
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<T> v0(HamMat_csr_full.dimension(), 1.0);
        eigenvals_full.resize(nev);
        eigenvecs_full.resize(HamMat_csr_full.dimension() * nev);
        qbasis::iram(HamMat_csr_full, v0.data(), nev, ncv, nconv, "lr", eigenvals_full.data(), eigenvecs_full.data());
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
        if (dim_repr < 1) {
            std::cout << "dim_repr = " << dim_repr << "!!!" << std::endl;
            return;
        }
        assert(ncv > nev + 1);
        std::cout << "Calculating highest energy state..." << std::endl;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) {
                std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
                std::cout << "Number of procs   = " << omp_get_num_procs() << std::endl;
            }
        }
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        std::vector<std::complex<double>> v0(HamMat_csr_repr.dimension(), 1.0);
        eigenvals_repr.resize(nev);
        eigenvecs_repr.resize(HamMat_csr_repr.dimension() * nev);
        qbasis::iram(HamMat_csr_repr, v0.data(), nev, ncv, nconv, "lr", eigenvals_repr.data(), eigenvecs_repr.data());
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
                for (MKL_INT cnt_opr = 0; cnt_opr < lhs.size(); cnt_opr++) {
                    auto &A = lhs[cnt_opr];
                    auto temp = eigenvecs_full[base + j];
                    if (A.q_diagonal()) {
                        values.push_back(std::pair<MKL_INT, T>(j,temp * basis_full[j].diagonal_operator(A)));
                    } else {
                        auto intermediate_state = A * basis_full[j];
                        for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
                            auto &ele = intermediate_state[cnt];
                            values.push_back(std::pair<MKL_INT, T>(binary_search(basis_full, ele.first, 0, dim_full), temp * ele.second));
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
    template class model<double>;
    template class model<std::complex<double>>;

}
