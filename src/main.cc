#include <iostream>
#include <vector>
#include "basis.h"
#include "operators.h"
#include "sparse.h"

void test_operator();

int main(){
    //test_operator();
    lil_mat<double> sp_lil(100000000);
    //sp_lil.prt();
    sp_lil.add(3, 2, 4.0);
    sp_lil.add(3, 4, 0.1);
    sp_lil.prt();
    
    csr_mat<double> sp_csr(sp_lil);
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
