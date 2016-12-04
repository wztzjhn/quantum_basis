#include <iostream>
#include <iomanip>
#include <vector>
#include "basis.h"
#include "operators.h"
#include "sparse.h"
#include "lanczos.h"

void test_operator();
void test_csrmm();
void test_lanczos();

int main(){
    //test_operator();
    test_csrmm();
    test_lanczos();

}

void test_csrmm() {
    lil_mat<std::complex<double>> sp_lil_test(2,true);
    sp_lil_test.add(0,0,1.0);
    sp_lil_test.add(0,1,std::complex<double>(2.0,3.0));
    sp_lil_test.add(1,1,4.5);
    sp_lil_test.prt();
    csr_mat<std::complex<double>> sp_csr_test(sp_lil_test);
    sp_csr_test.prt();
    std::complex<double> testv1[6], testv2[6];
    for (MKL_INT i = 0; i < 6; i++) {
        testv1[i] = std::complex<double>(i,i+1);
    }
//    sp_csr_test.MultMm(testv1, testv2,3);
//    for (MKL_INT i=0; i < 2; i++) {
//        for (MKL_INT j = 0; j < 3; j++) {
//            std::cout << std::setw(18) << testv2[i + j * 2];
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
}

void test_lanczos() {
    lil_mat<std::complex<double>> sp_lil(8,true);
    sp_lil.prt();
    sp_lil.add(3,3,11.0);
    sp_lil.add(2,3,8.0);
    //sp_lil.add(3,2,10.0);
    sp_lil.add(0,3,2.0);
    //sp_lil.add(1,0,3.0);
    sp_lil.add(1,1,4.0);
    sp_lil.add(0,0,1.0);
    sp_lil.add(4,4,12.0);
    sp_lil.add(2,4,std::complex<double>(9.0, 2.0));
    sp_lil.add(2,2,7.0);
    //sp_lil.add(2,0,6.0);
    sp_lil.add(1,3,5.0);
    sp_lil.add(1,6,5.0);
    sp_lil.add(1,7,4.0);
    sp_lil.add(3,6,2.0);
    sp_lil.prt();
    
    
    csr_mat<std::complex<double>> sp_csr(sp_lil);
    sp_lil.destroy();
    sp_csr.prt();
    
    MKL_INT k = 2;
    MKL_INT dim=8;
    MKL_INT ldh = 15;
    
    
    
    std::complex<double> *v = new std::complex<double>[dim*dim];
    std::complex<double> *resid = new std::complex<double>[dim];
    double hessenberg[2*ldh];
    std::vector<std::complex<double>> x = {1.0, std::complex<double>(2.3, 3.4), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double temp =nrm2(dim, x.data(), 1);
    std::cout << "norm of x = " << temp << std::endl;
    scal(dim, 1.0/nrm2(dim, x.data(), 1), x.data(), 1);
    for (MKL_INT i=0; i<dim; i++) resid[i] = x[i];
    
    double betak = 0.0;
    lanczos(0, k, sp_csr, betak, resid, v, hessenberg, ldh);

    
    std::complex<double> *hess= new std::complex<double>[ldh*ldh];
    hess2matform(hessenberg, hess, k, ldh);
    std::complex<double> *res = new std::complex<double>[ldh*dim];
    for (MKL_INT j=0; j < k; j++) {
        sp_csr.MultMv(v + j*dim, res + j*dim);
    }
    gemm('n', 'n', dim, k, k, std::complex<double>(-1.0,0), v, dim, hess, ldh, std::complex<double>(1.0,0), res, dim);
    for (MKL_INT i=0; i < dim; i++) {
        for (MKL_INT j = 0; j < k; j++) {
            std::complex<double> out = std::abs(res[i + j * dim])>1e-12?res[i + j * dim]:0.0;
            std::cout << std::setw(25) << out;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "betak = " << betak << std::endl;
    std::cout << "Vextra:" << std::endl;
    for (MKL_INT i = 0; i < dim; i++) {
        std::cout << std::setw(12) << betak * resid[i] << std::endl;
    }
    std::cout << "hessenberg: " << std::endl;
    for (MKL_INT i=0; i < k; i++) {
        std::cout << std::setw(12) << hessenberg[i] << std::setw(12) << hessenberg[i+ldh] << std::endl;
    }
    std::cout << "all v: " << std::endl;
    for (MKL_INT i=0; i<dim; i++) {
        for (MKL_INT j=0; j < k; j++) {
            std::cout << std::setw(25) << v[i+j*dim];
        }
        std::cout << std::setw(25) << resid[i] << std::endl;
    }
    
    std::cout << "-------------" << std::endl;
    MKL_INT kinc = 2;
    k=k+kinc;
    lanczos(k-kinc, kinc, sp_csr, betak, resid, v, hessenberg, ldh);
    hess2matform(hessenberg, hess, k, ldh);
    for (MKL_INT j=0; j < k; j++) {
        sp_csr.MultMv(v + j*dim, res + j*dim);
    }
    gemm('n', 'n', dim, k, k, std::complex<double>(-1.0,0), v, dim, hess, ldh, std::complex<double>(1.0,0), res, dim);
    for (MKL_INT i=0; i < dim; i++) {
        for (MKL_INT j = 0; j < k; j++) {
            std::complex<double> out = std::abs(res[i + j * dim])>1e-12?res[i + j * dim]:0.0;
            std::cout << std::setw(25) << out;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    
    std::cout << "betak = " << betak << std::endl;
    std::cout << "Vextra:" << std::endl;
    for (MKL_INT i = 0; i < dim; i++) {
        std::cout << std::setw(12) << betak * resid[i] << std::endl;
    }
    std::cout << "hessenberg: " << std::endl;
    for (MKL_INT i=0; i < k; i++) {
        std::cout << std::setw(12) << hessenberg[i] << std::setw(12) << hessenberg[i+ldh] << std::endl;
    }
    std::cout << "all v: " << std::endl;
    for (MKL_INT i=0; i<dim; i++) {
        for (MKL_INT j=0; j < k; j++) {
            std::cout << std::setw(25) << v[i+j*dim];
        }
        std::cout << std::setw(25) << resid[i] << std::endl;
    }
    
    std::cout << "eigenval: " << std::endl;
    std::vector<double> ritz(k);
    std::vector<double> vecs(k*ldh);
    select_shifts(hessenberg, ldh, k, "sm", ritz.data());
    for (MKL_INT j = 0; j < k; j++) {
        std::cout << std::setw(12) << ritz[j];
    }
    std::cout << std::endl;
//    std::cout << "eigenvec: " << std::endl;
//    for (MKL_INT i=0; i<k; i++) {
//        for (MKL_INT j=0; j < k; j++) {
//            std::cout << std::setw(25) << vecs[i+j*ldh];
//        }
//        std::cout << std::endl;
//    }
    
}


void test_operator(){
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
