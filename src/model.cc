#include <iostream>
//#include <fstream>
//#include <iomanip>
#include "qbasis.h"

namespace qbasis {

    template <typename T>
    void model<T>::enumerate_basis_all()
    {
        assert(dim_all > 0);
        basis_all[0].reset();
        for (MKL_INT j = 1; j < dim_all; j++) {
            basis_all[j] = basis_all[j-1];
            basis_all[j].increment();
        }
    }
    
    // need further optimization!
    template <typename T>
    void model<T>::enumerate_basis_conserve(const MKL_INT &n_sites, std::initializer_list<std::string> lst,
                                            std::initializer_list<mopr<std::complex<double>>> conserve_lst,
                                            std::initializer_list<double> val_lst)
//                                            const mopr<std::complex<double>> &conserve, const double &val)
    {
        MKL_INT num_conserve = static_cast<MKL_INT>(conserve_lst.size());
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
        
        dim_all = basis_temp.size();
        basis_all.clear();
        for (auto & ele : basis_temp) basis_all.push_back(ele);
    }
    
    template <typename T>
    void model<T>::sort_basis_all()
    {
        std::cout << "checking if basis sorted..." << std::endl;
        bool sorted = true;
        for (MKL_INT j = 0; j < dim_all - 1; j++) {
            assert(basis_all[j] != basis_all[j+1]);
            if (basis_all[j+1] < basis_all[j]) {
                sorted = false;
                break;
            }
        }
        if (! sorted) {
            std::cout << "sorting basis..." << std::endl;
            std::sort(basis_all.begin(), basis_all.end());
            std::cout << "sorting finished." << std::endl;
        } else {
            std::cout << "yes it is already sorted." << std::endl;
        }
    }
    
    
    template <typename T>
    void model<T>::generate_Ham_all_sparse(const bool &upper_triangle)
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
        lil_mat<T> matrix_lil(dim_all, upper_triangle);
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_all; i++) {
            for (MKL_INT cnt = 0; cnt < Ham_diag.size(); cnt++)                        // diagonal part:
                matrix_lil.add(i, i, basis_all[i].diagonal_operator(Ham_diag[cnt]));
            qbasis::wavefunction<T> intermediate_state = Ham_off_diag * basis_all[i];  // non-diagonal part:
            for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
                MKL_INT j = binary_search(basis_all, ele_new.first, 0, dim_all);       // < j | H | i > obtained
                assert(j != -1);
                if (upper_triangle) {
                    if (i <= j) matrix_lil.add(i, j, conjugate(ele_new.second));
                } else {
                    matrix_lil.add(i, j, conjugate(ele_new.second));
                }
                
            }
        }
        HamMat_csr = csr_mat<T>(matrix_lil);
        matrix_lil.destroy();
        std::cout << "Hamiltonian generated." << std::endl;
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    }
    
    template <typename T>
    void model<T>::locate_E0(const MKL_INT &nev, const MKL_INT &ncv)
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
        std::vector<T> v0(HamMat_csr.dimension(), 1.0);
        eigenvals.resize(nev);
        eigenvecs.resize(HamMat_csr.dimension() * nev);
        qbasis::iram(HamMat_csr, v0.data(), nev, ncv, nconv, "sr", eigenvals.data(), eigenvecs.data());
        assert(nconv > 1);
        E0 = eigenvals[0];
        gap = eigenvals[1] - eigenvals[0];
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "E0   = " << E0 << std::endl;
        std::cout << "Gap  = " << gap << std::endl;
    }
    
    template <typename T>
    void model<T>::locate_Emax(const MKL_INT &nev, const MKL_INT &ncv)
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
        std::vector<T> v0(HamMat_csr.dimension(), 1.0);
        eigenvals.resize(nev);
        eigenvecs.resize(HamMat_csr.dimension() * nev);
        qbasis::iram(HamMat_csr, v0.data(), nev, ncv, nconv, "lr", eigenvals.data(), eigenvecs.data());
        assert(nconv > 0);
        Emax = eigenvals[0];
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
        std::cout << "Emax = " << Emax << std::endl;
    }
    
    template <typename T>
    void model<T>::moprXeigenvec(const mopr<T> &lhs, T* vec_new, const MKL_INT &which_col)
    {
        assert(which_col >= 0 && which_col < nconv);
        for (MKL_INT j = 0; j < HamMat_csr.dimension(); j++) vec_new[j] = 0.0;
//        // leave these lines for a while, for openmp debugging purpose
//        static MKL_INT debug_flag = 0;
//        std::vector<std::list<MKL_INT>> jobid_list(100);
//        std::vector<std::list<double>> job_start(100), job_start_wait(100), job_finish_wait(100);
        
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        if (HamMat_csr.dimension() == dim_all) {
            MKL_INT base = dim_all * which_col;
            
            #pragma omp parallel for schedule(dynamic,16)
            for (MKL_INT j = 0; j < dim_all; j++) {
                if (std::abs(eigenvecs[base + j]) < lanczos_precision) continue;
//                std::chrono::time_point<std::chrono::system_clock> enter_time, start_wait, finish_wait;
//                enter_time = std::chrono::system_clock::now();
                std::vector<std::pair<MKL_INT, T>> values;
                for (MKL_INT cnt_opr = 0; cnt_opr < lhs.size(); cnt_opr++) {
                    auto &A = lhs[cnt_opr];
                    auto temp = eigenvecs[base + j];
                    if (A.q_diagonal()) {
                        values.push_back(std::pair<MKL_INT, T>(j,temp * basis_all[j].diagonal_operator(A)));
                    } else {
                        auto intermediate_state = A * basis_all[j];
                        for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
                            auto &ele = intermediate_state[cnt];
                            values.push_back(std::pair<MKL_INT, T>(binary_search(basis_all, ele.first, 0, dim_all), temp * ele.second));
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
        if (HamMat_csr.dimension() == dim_all) {
            MKL_INT base = dim_all * which_col;
            std::vector<T> vec_new(dim_all);
            moprXeigenvec(lhs, vec_new.data(), which_col);
            return dotc(dim_all, eigenvecs.data() + base, 1, vec_new.data(), 1);
        } else {
            std::cout << "not implemented yet" << std::endl;
            return static_cast<T>(0.0);
        }
    }
    
    template <typename T>
    T model<T>::measure(const mopr<T> &lhs1, const mopr<T> &lhs2, const MKL_INT &which_col)
    {
        assert(which_col >= 0 && which_col < nconv);
        if (HamMat_csr.dimension() == dim_all) {
            std::vector<T> vec_new1(dim_all);
            std::vector<T> vec_new2(dim_all);
            moprXeigenvec(lhs1, vec_new1.data(), which_col);
            moprXeigenvec(lhs2, vec_new2.data(), which_col);
            return dotc(dim_all, vec_new1.data(), 1, vec_new2.data(), 1);
        } else {
            std::cout << "not implemented yet" << std::endl;
            return static_cast<T>(0.0);
        }
    }
    
    
    // Explicit instantiation
    template class model<double>;
    template class model<std::complex<double>>;

}
