#include <iostream>
#include "qbasis.h"

namespace qbasis {

    template <typename T>
    MKL_INT binary_search(const std::vector<T> &some_basis, const T &val)
    {
        MKL_INT low = 0;
        MKL_INT high = some_basis.size() - 1;
        MKL_INT mid;
        while(low <= high) {
            mid = (low + high) / 2;
            if (val == some_basis[mid]) return mid;
            else if (val < some_basis[mid]) high = mid - 1;
            else low = mid + 1;
        }
        assert(false);
        return -1;
    }
    
    
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
            std::cout << "sorting finished." << std::endl << std::endl;
        } else {
            std::cout << "yes it is already sorted." << std::endl << std::endl;
        }
    }
    
    
    inline double conjugate(const double &rhs) { return rhs; }
    inline std::complex<double> conjugate(const std::complex<double> &rhs) { return std::conj(rhs); }
    
//    template <typename T>
//    MKL_INT generate_Ham_all_AtRow(model<T> &model0, threads_pool &jobs)
//    {
//        MKL_INT row = jobs.get_job();
//        while (row >= 0) {
//            // diagonal part:
//            for (MKL_INT cnt = 0; cnt < model0.Ham_diag.size(); cnt++)
//                model0.HamMat_lil.add(row, row, model0.basis_all[row].diagonal_operator(model0.Ham_diag[cnt]));
//            // non-diagonal part:
//            qbasis::wavefunction<T> intermediate_state = model0.Ham_off_diag * model0.basis_all[row];
//            for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
//                auto &ele_new = intermediate_state[cnt];
//                MKL_INT j = binary_search(model0.basis_all, ele_new.first);       // < j | H | i > obtained
//                if (model0.HamMat_lil.q_sym()) {
//                    if (row <= j) model0.HamMat_lil.add(row, j, conjugate(ele_new.second));
//                } else {
//                    model0.HamMat_lil.add(row, j, conjugate(ele_new.second));
//                }
//            }
//            row = jobs.get_job();
//        }
//        return 0;
//    }
//    template MKL_INT generate_Ham_all_AtRow(model<double> &model0, threads_pool &jobs);
//    template MKL_INT generate_Ham_all_AtRow(model<std::complex<double>> &model0, threads_pool &jobs);
    

    
    
    
    template <typename T>
    void model<T>::generate_Ham_all_sparse(const bool &upper_triangle)
    {
        std::cout << "Generating Hamiltonian Matrix..." << std::endl;
        lil_mat<T> matrix_lil(dim_all, upper_triangle);
        #pragma omp parallel for schedule(dynamic,1)
        for (MKL_INT i = 0; i < dim_all; i++) {
            for (MKL_INT cnt = 0; cnt < Ham_diag.size(); cnt++)                        // diagonal part:
                matrix_lil.add(i, i, basis_all[i].diagonal_operator(Ham_diag[cnt]));
            qbasis::wavefunction<T> intermediate_state = Ham_off_diag * basis_all[i];  // non-diagonal part:
            for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele_new = intermediate_state[cnt];
                MKL_INT j = binary_search(basis_all, ele_new.first);                   // < j | H | i > obtained
                if (upper_triangle) {
                    if (i <= j) matrix_lil.add(i, j, conjugate(ele_new.second));
                } else {
                    matrix_lil.add(i, j, conjugate(ele_new.second));
                }
                
            }
        }
        HamMat_csr = csr_mat<T>(matrix_lil);
        matrix_lil.destroy();
        std::cout << "Hamiltonian generated." << std::endl << std::endl;
    }
    
    template <typename T>
    void model<T>::locate_E0(const MKL_INT &nev, const MKL_INT &ncv)
    {
        assert(ncv > nev + 1);
        MKL_INT nconv;
        std::vector<T> v0(HamMat_csr.dimension(), 1.0);
        eigenvals.resize(nev);
        eigenvecs.resize(HamMat_csr.dimension() * nev);
        qbasis::iram(HamMat_csr, v0.data(), nev, ncv, nconv, "sr", eigenvals.data(), eigenvecs.data());
        assert(nconv > 1);
        E0 = eigenvals[0];
        gap = eigenvals[1] - eigenvals[0];
        std::cout << "E0   = " << eigenvals[0] << std::endl;
        std::cout << "Gap  = " << eigenvals[1] - eigenvals[0] << std::endl;
    }
    
    template <typename T>
    void model<T>::locate_Emax(const MKL_INT &nev, const MKL_INT &ncv)
    {
        assert(ncv > nev + 1);
        MKL_INT nconv;
        std::vector<T> v0(HamMat_csr.dimension(), 1.0);
        eigenvals.resize(nev);
        eigenvecs.resize(HamMat_csr.dimension() * nev);
        qbasis::iram(HamMat_csr, v0.data(), nev, ncv, nconv, "lr", eigenvals.data(), eigenvecs.data());
        assert(nconv > 0);
        Emax = eigenvals[0];
        std::cout << "Emax = " << eigenvals[0] << std::endl;
    }
    
    // Explicit instantiation
    template MKL_INT binary_search(const std::vector<qbasis::mbasis_elem> &basis_all, const qbasis::mbasis_elem &val);
    
    template class model<double>;
    template class model<std::complex<double>>;

}
