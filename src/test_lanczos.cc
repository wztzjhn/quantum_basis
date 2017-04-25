#include <iostream>
#include <iomanip>
#include <fstream>
#include "qbasis.h"

void test_lanczos_memoAll() {
    std::cout << "--------- test lanczos ---------" << std::endl;
    MKL_INT dim=8;
    MKL_INT m = 6;
    MKL_INT ldh = 15;
    
    qbasis::lil_mat<std::complex<double>> sp_lil(8,true);
    sp_lil.add(3,3,11.0);
    sp_lil.add(2,3,8.0);
    sp_lil.add(0,3,2.0);
    sp_lil.add(1,1,4.0);
    sp_lil.add(0,0,1.0);
    sp_lil.add(4,4,12.0);
    sp_lil.add(2,4,std::complex<double>(9.0, 2.0));
    sp_lil.add(2,2,7.0);
    sp_lil.add(1,3,5.0);
    sp_lil.add(1,6,5.0);
    sp_lil.add(1,7,4.0);
    sp_lil.add(3,6,2.0);
    qbasis::csr_mat<std::complex<double>> sp_csr_uppper(sp_lil);
    
    sp_lil.use_full_matrix();
    sp_lil.add(3,2,8.0);
    sp_lil.add(3,0,2.0);
    sp_lil.add(4,2,std::complex<double>(9.0, -2.0));
    sp_lil.add(3,1,5.0);
    sp_lil.add(6,1,5.0);
    sp_lil.add(7,1,4.0);
    sp_lil.add(6,3,2.0);
    qbasis::csr_mat<std::complex<double>> sp_csr_full(sp_lil);
    sp_lil.destroy();
    
    sp_csr_full.prt();
    auto dense_mat = sp_csr_full.to_dense();
    for (MKL_INT row = 0; row < dim; row++) {
        for (MKL_INT col = 0; col < dim; col++) {
            std::cout << dense_mat[row + col * dim] << "\t";
        }
        std::cout << std::endl;
    }

    std::vector<std::complex<double>> x = {1.0, std::complex<double>(2.3, 3.4), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    qbasis::scal(dim, 1.0/qbasis::nrm2(dim, x.data(), 1), x.data(), 1);
    
    std::complex<double> *v_up = new std::complex<double>[dim*dim];
    std::complex<double> *resid_up = new std::complex<double>[dim];
    double hessenberg_up[2*ldh];
    hessenberg_up[0] = 0.0;
    double betak_up = 0.0;
    for (MKL_INT i=0; i<dim; i++) resid_up[i] = x[i];
    
    lanczos(0, m, sp_csr_uppper, betak_up, resid_up, v_up, hessenberg_up, ldh);
    
    std::complex<double> *hess_up= new std::complex<double>[ldh*ldh];
    qbasis::hess2matform(hessenberg_up, hess_up, m, ldh);
    std::complex<double> *res_up = new std::complex<double>[ldh*dim];
    for (MKL_INT j=0; j < m; j++) sp_csr_uppper.MultMv(v_up + j*dim, res_up + j*dim);
    qbasis::gemm('n', 'n', dim, m, m, std::complex<double>(-1.0,0), v_up, dim, hess_up, ldh, std::complex<double>(1.0,0), res_up, dim);
    for (MKL_INT i=0; i < dim; i++) {
        for (MKL_INT j = 0; j < m; j++) {
            if (j < m - 1) {
                if (std::abs(res_up[i + j * dim]) >= qbasis::lanczos_precision) {
                    std::cout << "(i,j) = (" << i << "," << j << ")" << std::endl;
                    std::cout << "res   = " << res_up[i + j * dim] << std::endl;
                }
                assert(std::abs(res_up[i + j * dim]) < qbasis::lanczos_precision);
            } else {
                if (std::abs(res_up[i + j * dim] - betak_up * resid_up[i]) >= qbasis::lanczos_precision) {
                    std::cout << "(i,j)   = (" << i << "," << j << ")" << std::endl;
                    std::cout << "res     = " << res_up[i + j * dim] << std::endl;
                    std::cout << "b*resid = " << betak_up * resid_up[i] << std::endl;
                }
                assert(std::abs(res_up[i + j * dim] - betak_up * resid_up[i]) < qbasis::lanczos_precision);
            }
        }
    }
    
    // check with full matrix
    std::complex<double> *v_full = new std::complex<double>[dim*dim];
    std::complex<double> *resid_full = new std::complex<double>[dim];
    double hessenberg_full[2*ldh];
    hessenberg_full[0] = 0.0;
    double betak_full = 0.0;
    for (MKL_INT i=0; i<dim; i++) resid_full[i] = x[i];
    
    lanczos(0, m, sp_csr_full, betak_full, resid_full, v_full, hessenberg_full, ldh);
    
    assert(std::abs(betak_up - betak_full) < qbasis::lanczos_precision);
    for (MKL_INT j = 0; j < dim*m; j++) {
        assert(std::abs(v_up[j] - v_full[j]) < qbasis::lanczos_precision);
    }
    for (MKL_INT j = 0; j < dim; j++) {
        assert(std::abs(resid_up[j] - resid_full[j]) < qbasis::lanczos_precision);
    }
    for (MKL_INT j = 1; j < m; j++) {
        assert(std::abs(hessenberg_up[j] - hessenberg_full[j]) < qbasis::lanczos_precision);
        assert(std::abs(hessenberg_up[j+ldh] - hessenberg_full[j+ldh]) < qbasis::lanczos_precision);
    }
    assert(std::abs(hessenberg_up[ldh] - hessenberg_full[ldh]) < qbasis::lanczos_precision);
    
    delete [] v_up;
    delete [] resid_up;
    delete [] hess_up;
    delete [] res_up;
    delete [] v_full;
    delete [] resid_full;
    std::cout << std::endl << std::endl;
}

void test_iram()
{
    std::cout << "--------- test iram ---------" << std::endl;
    MKL_INT dim=8;
    qbasis::lil_mat<std::complex<double>> sp_lil(8,true);
    sp_lil.add(3,3,11.0);
    sp_lil.add(2,3,8.0);
    sp_lil.add(0,3,2.0);
    sp_lil.add(1,1,4.0);
    sp_lil.add(0,0,1.0);
    sp_lil.add(4,4,12.0);
    sp_lil.add(2,4,std::complex<double>(9.0, 2.0));
    sp_lil.add(2,2,7.0);
    sp_lil.add(1,3,5.0);
    sp_lil.add(1,6,5.0);
    sp_lil.add(1,7,4.0);
    sp_lil.add(3,6,2.0);
    qbasis::csr_mat<std::complex<double>> sp_csr_uppper(sp_lil);
    
    std::vector<std::complex<double>> x = {1.0, std::complex<double>(2.3, 3.4), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    
    MKL_INT nev = 2, ncv = 5, nconv;
    std::vector<double> eigenvals(nev), tol(nev);
    std::vector<std::complex<double>> eigenvecs(nev * dim);
    iram(sp_csr_uppper, x.data(), nev, ncv, nconv, "sr", eigenvals.data(), eigenvecs.data(), true);
    assert(nconv == 2);
    assert(std::abs(eigenvals[0] + 5.2955319) < 0.000001);
    assert(std::abs(eigenvals[1] + 3.3838164) < 0.000001);
    for (MKL_INT j = 0; j < nconv; j++) {
        std::cout << "sigma[" << j << "] = " << std::setprecision(9) << eigenvals[j] << std::endl;
    }
    assert(std::abs(std::abs(eigenvecs[0]) - 0.0949047) < 0.000001);
    assert(std::abs(std::abs(eigenvecs[1]) - 0.5984956) < 0.000001);
    assert(std::abs(std::abs(eigenvecs[2]) - 0.3237924) < 0.000001);
    assert(std::abs(std::abs(eigenvecs[7]) - 0.4520759) < 0.000001);
    std::cout << std::endl << std::endl;
}
