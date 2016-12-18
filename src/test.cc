#include <iostream>
#include <iomanip>
#include "qbasis.h"



void test_basis();
void test_operator();

void test_lanczos_memoAll();
void test_iram();
void test_cfraction();


int main(){
    test_lanczos_memoAll();
    test_iram();
    test_basis();
    //test_operator();
    
    test_cfraction();
}

void test_cfraction() {
    std::vector<double> a(1000,2.0), b(1000,1.0);
    a[0] = 1.0;
    std::cout << "len =   5, sqrt(2) = " << qbasis::continued_fraction(a.data(), b.data(), 5) << std::endl;
    std::cout << "len =  10, sqrt(2) = " << qbasis::continued_fraction(a.data(), b.data(), 10) << std::endl;
    std::cout << "len =  50, sqrt(2) = " << qbasis::continued_fraction(a.data(), b.data(), 50) << std::endl;
    
    a[0] = 3.0;
    for (MKL_INT j = 1; j < a.size(); j++) {
        a[j] = 6.0;
        b[j] = (2.0 * j - 1.0) * (2.0 * j - 1.0);
    }
    std::cout << "len =   5, pi = " << qbasis::continued_fraction(a.data(), b.data(), 5) << std::endl;
    std::cout << "len =  10, pi = " << qbasis::continued_fraction(a.data(), b.data(), 10) << std::endl;
    std::cout << "len =  50, pi = " << qbasis::continued_fraction(a.data(), b.data(), 50) << std::endl;
}

void test_basis() {
    std::cout << "--------- test basis ---------" << std::endl;
    qbasis::basis_elem ele1(9, "spin-1");
    qbasis::basis_elem ele2(ele1);
    ele1.siteWrite(7, 1);
    ele2.siteWrite(0, 2);
    ele1.prt();
    ele2.prt();
    std::cout << "ele1 < ele2  ? " << (ele1 < ele2) << std::endl;
    std::cout << "ele1 == ele2 ? " << (ele1 == ele2) << std::endl;
    std::cout << std::endl;
    
    qbasis::mbasis_elem mele1(9, {"spin-1/2", "spin-1"});
    qbasis::mbasis_elem mele2(9, {"spin-1/2", "spin-1"});
    mele1.prt();
    std::cout << "mele1 == mele2 ? " << (mele1 == mele2) << std::endl;
    
}

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




void test_operator(){
    using namespace qbasis;
    std::vector<std::vector<std::complex<double>>> vec_sigmax = {{0, 1}, {1, 0}};
    std::vector<std::vector<std::complex<double>>> vec_sigmay = {{0, std::complex<double>(0,-1)}, {std::complex<double>(0,1), 0}};
    std::vector<std::vector<std::complex<double>>> vec_sigmaz = {{1, 0}, {0, -1}};
    std::vector<opr<std::complex<double>>> sigmax_list, sigmay_list, sigmaz_list;
    for (decltype(sigmax_list.size()) i = 0; i < 5; i++) {
        sigmax_list.push_back(opr<std::complex<double>>(i, 0, true, vec_sigmax));
        sigmay_list.push_back(opr<std::complex<double>>(i, 1, 0, vec_sigmay));
        sigmaz_list.push_back(opr<std::complex<double>>(i, 3, true, vec_sigmaz));
    }

    mopr<std::complex<double>> ham1(sigmaz_list[3]);
    mopr<std::complex<double>> ham2(sigmay_list[2]);


    ham1 = ham1 + ham2;
    ham1.prt(); std::cout << std::endl;

    ham1 = ham1 * ham2;
    ham1.prt(); std::cout << std::endl;

    std::cout << ham1[0].q_prop_identity() << std::endl;



    //ham *= ham;
    //ham.prt(); std::cout << std::endl;

    auto temp1 = std::vector<std::complex<double>>(3);
    temp1[0] =std::complex<double>(2.0,1.0);
    temp1[1] = 0.8;
    temp1[2] =std::complex<double>(0.5,0.3);
    opr<std::complex<double>> chi(3,3,true,temp1);
    chi.prt();
    std::cout << std::endl;

    auto temp2 = std::vector<std::vector<std::complex<double>>>(3,std::vector<std::complex<double>>(3));
    temp2[0][0] = std::complex<double>(0.3,0.5);
    temp2[0][1] = 2.0;
    temp2[0][2] = std::complex<double>(0.0, 1.3);
    temp2[1][0] = 2.3;
    temp2[1][1] = std::complex<double>(2.0,2.6);
    temp2[1][2] = std::complex<double>(0.9, 1.1);
    temp2[2][0] = 0.0;
    temp2[2][1] = std::complex<double>(3.3,3.3);
    temp2[2][2] = std::complex<double>(0.5,0.3);
    opr<std::complex<double>> psi(3, 3, true, temp2);
    psi.prt();
    std::cout << std::endl;

//    std::cout << "diagonal + diagonal" << std::endl;
//    auto alpha = chi;
//    alpha += alpha;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "nondiagonal + nondiagonal" << std::endl;
//    alpha = psi;
//    alpha += alpha;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "diagonal + nondiagonal" << std::endl;
//    alpha = chi;
//    alpha += psi;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "nondiagonal + diagonal" << std::endl;
//    alpha = psi;
//    alpha += chi;
//    alpha.prt();
//    std::cout << std::endl;

//    std::cout << "test memory leak" << std::endl;
//    for (size_t i = 0; i < 1000000; i++) {
//        alpha = chi;
//        alpha += psi;
//        alpha = chi;
//        alpha += chi;
//        alpha = psi;
//        alpha += chi;
//        alpha = psi;
//        alpha += psi;
//    }
//    std::cout << std::endl;

//    std::cout << "diagonal - diagonal" << std::endl;
//    alpha = chi;
//    alpha -= alpha;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "nondiagonal - nondiagonal" << std::endl;
//    alpha = psi;
//    alpha -= alpha;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "diagonal - nondiagonal" << std::endl;
//    alpha = chi;
//    alpha -= psi;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "nondiagonal - diagonal" << std::endl;
//    alpha = psi;
//    alpha -= chi;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "diagonal * diagonal" << std::endl;
//    alpha = chi;
//    alpha *= alpha;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "nondiagonal * nondiagonal" << std::endl;
//    alpha = psi;
//    alpha *= alpha;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "diagonal * nondiagonal" << std::endl;
//    alpha = chi;
//    alpha *= psi;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "nondiagonal * diagonal" << std::endl;
//    alpha = psi;
//    alpha *= chi;
//    alpha.prt();
//    std::cout << std::endl;
//
//    std::cout << "nondiagonal (*) diagonal" << std::endl;
//    auto beta = psi * chi;
//    beta.prt();
//    std::cout << std::endl;
//
//    std::cout << "nondiagonal (*) nondiagonal" << std::endl;
//    auto gamma = psi * psi;
//    gamma.prt();
//    std::cout << std::endl;

//    std::vector<opr<std::complex<double>>> kkk(5, psi);
//    kkk[0].prt();
//    kkk[1].prt();
//    kkk[2].prt();
//
//    kkk[2].negative();
//    kkk[2].prt();
//    kkk[2] *= 0.0;
//    kkk[2].prt();
//    auto ttst = kkk[0] * kkk[1];
//    ttst.prt();
//
//    std::cout << std::endl;
//    chi.prt();
//    psi.prt();
//    kkk[0] = chi  - chi;
//    kkk[0].prt();
//    kkk[0].simplify();
//    kkk[0].prt();
//    std::cout << std::endl;
//
//    std::complex<double> prefactor;
//    kkk[0] = normalize(chi, prefactor);
//    kkk[0].prt();
//    std::cout << "prefactor = " << prefactor << std::endl;
//    std::cout << "norm^2 now = " << kkk[0].norm() * kkk[0].norm() << std::endl;
//    std::cout << std::endl;
//
//    std::vector<std::vector<std::complex<double>>> vec_pauli = {{0, std::complex<double>(0,-1)}, {std::complex<double>(0,1), 0}};
//    opr<std::complex<double>> pauli(5, 2, false, vec_pauli);
//    pauli *= 2.0;
//    pauli.prt();
//    std::cout << "pauli.norm = " << pauli.norm() << std::endl;
//    auto pauli_new = normalize(pauli, prefactor);
//    std::cout << "prefactor = " << prefactor << std::endl;
//    std::cout << "norm now = " << pauli_new.norm() << std::endl;
//    pauli_new.prt();






}
