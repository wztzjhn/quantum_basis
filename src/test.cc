#include <iostream>
#include <iomanip>
#include "qbasis.h"

void test_trimer();

void test_basis();
void test_operator();

void test_lanczos_memoAll();
void test_iram();
void test_cfraction();
void test_dotc();
void test_lattice();
void test_dimer();
void test_Hubbard();
void test_tJ();
void test_bubble();

int main(){
    //test_lanczos_memoAll();
    //test_iram();
    //test_basis();
    //test_operator();
    
    //test_cfraction();
    //test_dotc();
    //test_lattice();
    
    //test_dimer();
    
    //test_Hubbard();
    
    test_tJ();
    
    //test_bubble();
    
    //std::cout << boost::math::binomial_coefficient<double>(10, 2) << std::endl;
    
    //std::cout << qbasis::dynamic_base(std::vector<MKL_INT>{1,2,3}, std::vector<MKL_INT>{2,3,5}) << std::endl;
    
    //test_trimer();
}

void test_dimer() {
    qbasis::mbasis_elem basis_uu(2, {"spin-1/2"});
    qbasis::mbasis_elem basis_dd(2, {"spin-1/2"});
    basis_dd.siteWrite(0, 0, 1);
    basis_dd.siteWrite(1, 0, 1);
    qbasis::mbasis_elem basis_ud(2, {"spin-1/2"});
    basis_ud.siteWrite(1, 0, 1);
    qbasis::mbasis_elem basis_du(2, {"spin-1/2"});
    basis_du.siteWrite(0, 0, 1);
    // define the basic operators: S+, S-, Sz
    std::vector<std::vector<std::complex<double>>> Sx(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sy(2,std::vector<std::complex<double>>(2));
    std::vector<std::complex<double>> Sz(2);
    Sx[0][0] = 0.0;
    Sx[0][1] = 0.5;
    Sx[1][0] = 0.5;
    Sx[1][1] = 0.0;
    Sy[0][0] = 0.0;
    Sy[0][1] = std::complex<double>(0.0, -0.5);
    Sy[1][0] = std::complex<double>(0.0,  0.5);
    Sy[1][1] = 0.0;
    Sz[0] = 0.5;
    Sz[1] = -0.5;
    qbasis::opr<std::complex<double>> s0x(0,0,false,Sx);
    qbasis::opr<std::complex<double>> s0y(0,0,false,Sy);
    qbasis::opr<std::complex<double>> s0z(0,0,false,Sz);
    qbasis::opr<std::complex<double>> s1x(1,0,false,Sx);
    qbasis::opr<std::complex<double>> s1y(1,0,false,Sy);
    qbasis::opr<std::complex<double>> s1z(1,0,false,Sz);
    auto Ham = s0x * s1x + s0y * s1y + s0z * s1z;
    
    qbasis::wavefunction<std::complex<double>> singlet(  std::complex<double>( 1.0/sqrt(2.0)) * basis_ud
                                                       + std::complex<double>(-1.0/sqrt(2.0)) * basis_du);
    qbasis::wavefunction<std::complex<double>> triplet_z(  std::complex<double>(1.0/sqrt(2.0)) * basis_ud
                                                         + std::complex<double>(1.0/sqrt(2.0)) * basis_du);
    qbasis::wavefunction<std::complex<double>> triplet_u(basis_uu);
    qbasis::wavefunction<std::complex<double>> triplet_d(basis_dd);
    qbasis::wavefunction<std::complex<double>> triplet_x(  std::complex<double>(-1.0/sqrt(2.0)) * basis_uu
                                                         + std::complex<double>( 1.0/sqrt(2.0)) * basis_dd);
    qbasis::wavefunction<std::complex<double>> triplet_y(  std::complex<double>(0.0, 1.0/sqrt(2.0)) * basis_uu
                                                         + std::complex<double>(0.0, 1.0/sqrt(2.0)) * basis_dd);
    auto res0 = s0x * singlet;
    auto res1 = s0x * triplet_x;
    auto res2 = s0x * triplet_y;
    auto res3 = s0x * triplet_z;
    res0.prt();
    std::cout << std::endl << std::endl;
    
}

void test_lattice() {
    qbasis::lattice square("square",std::vector<MKL_INT>{3,3},std::vector<std::string>{"pbc", "pbc"});
    std::vector<MKL_INT> coor = {1,2};
    MKL_INT sub = 0;
    for (MKL_INT site = 0; site < square.total_sites(); site++) {
        square.site2coor(coor, sub, site);
        std::cout << "(" << coor[0] << "," << coor[1] << "," << sub << ") : " << site << std::endl;
        MKL_INT site2;
        square.coor2site(coor, sub, site2);
        assert(site == site2);
    }
    
    auto plan = square.translation_plan(std::vector<MKL_INT>{2, 1});
    for (MKL_INT j = 0; j < square.total_sites(); j++) {
        std::cout << j << " -> " << plan[j] << std::endl;
    }
    std::cout << std::endl;
}

void test_dotc() {
    std::vector<std::complex<double>> x(2), y(2);
    x[0] = std::complex<double>(1.0,2.0);
    x[1] = std::complex<double>(2.0,3.0);
    y[0] = std::complex<double>(2.0,-2.0);
    y[1] = std::complex<double>(-5.0,7.0);
    std::complex<double> z = qbasis::dotc(2, x.data(), 1, y.data(), 1);
    std::cout << "z = " << z << std::endl;
    assert(std::abs(z - std::complex<double>(9.0,23.0)) < qbasis::lanczos_precision);
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
    ele1.siteWrite(5, 1);
    ele2.siteWrite(0, 2);
    ele1.prt(); std::cout << std::endl;
    ele2.prt(); std::cout << std::endl;
    std::cout << "ele1 < ele2  ? " << (ele1 < ele2) << std::endl;
    std::cout << "ele1 == ele2 ? " << (ele1 == ele2) << std::endl;
    
    auto stats = ele1.statistics();
    for (MKL_INT j = 0; j < stats.size(); j++) {
        std::cout << "stat " << j << ", count = " << stats[j] << std::endl;
    }
    
    qbasis::mbasis_elem mele1(9, {"spin-1/2", "spin-1"});
    qbasis::mbasis_elem mele2(9, {"spin-1/2", "spin-1"});
    mele1.prt();
    std::cout << std::endl;
    std::cout << "mele1 == mele2 ? " << (mele1 == mele2) << std::endl;
    mele1.siteWrite(3, 1, 2);
    mele1.siteWrite(2, 1, 2);
    mele1.siteWrite(1, 0, 1);
    mele2 = mele1;
    auto stats2 = mele1.statistics();
    for (MKL_INT j = 0; j < stats2.size(); j++) {
        std::cout << "stat " << j << ", count = " << stats2[j] << std::endl;
    }
    
    qbasis::lattice square("square",std::vector<MKL_INT>{3,3},std::vector<std::string>{"pbc", "pbc"});
    MKL_INT sgn;
    mele1.translate(square, std::vector<MKL_INT>{1,2}, sgn);
    std::cout << "translational equiv?: " << qbasis::trans_equiv(mele1, mele2, square) << std::endl;
    
    std::cout << std::endl;
    
    
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

// test Hubbard chain
void test_Hubbard() {
    std::cout << "testing Hubbard chain." << std::endl;
    MKL_INT Nsites = 6;
    double U = 1.1;
    MKL_INT total_fermion = 4;
    
    auto c_up = std::vector<std::vector<std::complex<double>>>(4,std::vector<std::complex<double>>(4, 0.0));
    auto c_dn = std::vector<std::vector<std::complex<double>>>(4,std::vector<std::complex<double>>(4, 0.0));
    c_up[0][1] = std::complex<double>(1.0,0.0);
    c_up[2][3] = std::complex<double>(1.0,0.0);
    c_dn[0][2] = std::complex<double>(1.0,0.0);
    c_dn[1][3] = std::complex<double>(-1.0,0.0);
    
    qbasis::mopr<std::complex<double>> Nfermion;
    qbasis::model<std::complex<double>> Hubbard;
    
    qbasis::lattice lattice("chain",std::vector<MKL_INT>{Nsites},std::vector<std::string>{"pbc"});
    
    for (MKL_INT i = 0; i < Nsites; i++) {
        auto c_up_i    = qbasis::opr<std::complex<double>>(i,0,true,c_up);
        auto c_dn_i    = qbasis::opr<std::complex<double>>(i,0,true,c_dn);
        auto c_up_dg_i = c_up_i; c_up_dg_i.dagger();
        auto c_dn_dg_i = c_dn_i; c_dn_dg_i.dagger();
        auto n_up_i    = c_up_dg_i * c_up_i;
        auto n_dn_i    = c_dn_dg_i * c_dn_i;
        
        MKL_INT j = (i < Nsites - 1) ? (i+1) : 0;
        auto c_up_j    = qbasis::opr<std::complex<double>>(j,0,true,c_up);
        auto c_dn_j    = qbasis::opr<std::complex<double>>(j,0,true,c_dn);
        auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
        auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
        
        Hubbard.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_up_dg_i * c_up_j ));
        Hubbard.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_up_dg_j * c_up_i ));
        Hubbard.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_dn_dg_i * c_dn_j ));
        Hubbard.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_dn_dg_j * c_dn_i ));
        Hubbard.add_diagonal_Ham(std::complex<double>(U,0.0) * (n_up_i * n_dn_i));
        
        Nfermion += (n_up_i + n_dn_i);
    }
    
    Hubbard.enumerate_basis_full_conserve(Nsites, {"electron"}, {Nfermion}, {static_cast<double>(total_fermion)});
    std::cout << "dim_all = " << Hubbard.dim_full << std::endl;
    
    /*
    for (MKL_INT j = 0; j < Hubbard.dim_all; j++) {
        std::cout << "j = " << j << std::endl;
        Hubbard.basis_all[j].prt_nonzero();
        std::cout << std::endl;
    }
    */
    
    // generating Hamiltonian matrix
    Hubbard.generate_Ham_sparse_full(false);
    std::cout << std::endl;
    
    
    Hubbard.locate_E0_full(28,39);
    std::cout << std::endl;
    
//    std::cout << "full matrix" << std::endl;
//    auto xxx = Hubbard.HamMat_csr_full.to_dense();
//    for (MKL_INT i = 0; i < Hubbard.dim_full; i++) {
//        for (MKL_INT j = 0; j < Hubbard.dim_full; j++) {
//            std::cout << xxx[i + j * Hubbard.dim_full] << "\t";
//        }
//        std::cout << std::endl;
//    }
    
    
//    for (MKL_INT j = 0; j < Hubbard.dim_full; j++) {
//        std::cout << "basis [ " << j << "]:" << std::endl;
//        Hubbard.basis_full[j].prt_nonzero();
//        std::cout << std::endl;
//    }
    
    
    Hubbard.basis_init_repr(std::vector<MKL_INT>{0}, lattice);
//        for (MKL_INT j = 0; j < Hubbard.dim_repr; j++) {
//            std::cout << "repr [" << j << "]:" << Hubbard.basis_repr[j] << std::endl;
//        }
//        for (MKL_INT j = 0; j < Hubbard.dim_full; j++) {
//            std::cout << "basis_long[" << j << "]:" << Hubbard.basis_belong[j] << std::endl;
//        }
//    
//        for (MKL_INT j = 0; j < Hubbard.dim_full; j++) {
//            std::cout << "basis_coeff[" << j << "]:" << Hubbard.basis_coeff[j] << std::endl;
//        }
    
    
    Hubbard.generate_Ham_sparse_repr();
    std::cout << std::endl;
    

    
    Hubbard.locate_E0_repr();
    std::cout << std::endl;
    
    Hubbard.basis_init_repr(std::vector<MKL_INT>{1}, lattice);
    Hubbard.generate_Ham_sparse_repr();
    std::cout << std::endl;
    
    
//    for (MKL_INT j = 0; j < Hubbard.dim_repr; j++) {
//        std::cout << "repr [" << j << "]:" << Hubbard.basis_repr[j] << std::endl;
//    }
//    for (MKL_INT j = 0; j < Hubbard.dim_full; j++) {
//        std::cout << "basis_long[" << j << "]:" << Hubbard.basis_belong[j] << std::endl;
//    }
//    
//    for (MKL_INT j = 0; j < Hubbard.dim_full; j++) {
//        std::cout << "basis_coeff[" << j << "]:" << Hubbard.basis_coeff[j] << std::endl;
//    }
    
    Hubbard.locate_E0_repr();
    std::cout << std::endl;
    
    Hubbard.basis_init_repr(std::vector<MKL_INT>{2}, lattice);
    Hubbard.generate_Ham_sparse_repr();
    std::cout << std::endl;
    
    
//    for (MKL_INT j = 0; j < Hubbard.dim_repr; j++) {
//        std::cout << "repr [" << j << "]:" << Hubbard.basis_repr[j] << std::endl;
//    }
//    for (MKL_INT j = 0; j < Hubbard.dim_full; j++) {
//        std::cout << "basis_long[" << j << "]:" << Hubbard.basis_belong[j] << std::endl;
//    }
//    
//    for (MKL_INT j = 0; j < Hubbard.dim_full; j++) {
//        std::cout << "basis_coeff[" << j << "]:" << Hubbard.basis_coeff[j] << std::endl;
//    }
    
    Hubbard.locate_E0_repr();
    std::cout << std::endl;
    
//    for (MKL_INT j = 0; j < Hubbard.dim_full; j++) {
//        std::cout << "eigenvec[" << j << "]=" << Hubbard.eigenvecs_full[j] << std::endl;
//    }
    
    
    Hubbard.basis_init_repr(std::vector<MKL_INT>{3}, lattice);
    Hubbard.generate_Ham_sparse_repr();
    std::cout << std::endl;
    Hubbard.locate_E0_repr();
    std::cout << std::endl;
}


// test t-J model on square lattice
void test_tJ()
{
    
    std::cout << "testing tJ model." << std::endl;
    MKL_INT lx = 3, ly = 2;
    qbasis::lattice lattice("square",std::vector<MKL_INT>{lx,ly},std::vector<std::string>{"pbc", "pbc"});
    MKL_INT total_up = lx * ly / 2;
    MKL_INT total_dn = lx * ly / 2;
    
    double J = 1.9;
    
    auto c_up  = std::vector<std::vector<std::complex<double>>>(3,std::vector<std::complex<double>>(3, 0.0));
    auto c_dn  = std::vector<std::vector<std::complex<double>>>(3,std::vector<std::complex<double>>(3, 0.0));
    c_up[0][1] = std::complex<double>(1.0,0.0);
    c_dn[0][2] = std::complex<double>(1.0,0.0);
    
    qbasis::mopr<std::complex<double>> N_up, N_dn;
    qbasis::model<std::complex<double>> tJ;

    for (MKL_INT m = 0; m < lattice.Lx(); m++) {
        for (MKL_INT n = 0; n < lattice.Ly(); n++) {
            MKL_INT site_i, site_j;
            lattice.coor2site(std::vector<MKL_INT>{m,n}, 0, site_i);
            auto c_up_i = qbasis::opr<std::complex<double>>(site_i,0,true,c_up);
            auto c_dn_i = qbasis::opr<std::complex<double>>(site_i,0,true,c_dn);
            auto c_up_dg_i = c_up_i; c_up_dg_i.dagger();
            auto c_dn_dg_i = c_dn_i; c_dn_dg_i.dagger();
            auto Sp_i = c_up_dg_i * c_dn_i;
            auto Sm_i = c_dn_dg_i * c_up_i;
            auto Sz_i = std::complex<double>(0.5,0.0) * (c_up_dg_i * c_up_i - c_dn_dg_i * c_dn_i);
            auto n_i  = c_up_dg_i * c_up_i + c_dn_dg_i * c_dn_i;
            
            N_up += (c_up_dg_i * c_up_i);
            N_dn += (c_dn_dg_i * c_dn_i);
            
            // right neighbor
            lattice.coor2site(std::vector<MKL_INT>{m+1,n}, 0, site_j);
            auto c_up_j = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
            auto c_dn_j = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
            auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
            auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
            auto Sp_j = c_up_dg_j * c_dn_j;
            auto Sm_j = c_dn_dg_j * c_up_j;
            auto Sz_j = std::complex<double>(0.5,0.0) * (c_up_dg_j * c_up_j - c_dn_dg_j * c_dn_j);
            auto n_j  = c_up_dg_j * c_up_j + c_dn_dg_j * c_dn_j;
            
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_up_dg_i * c_up_j ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_up_dg_j * c_up_i ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_dn_dg_i * c_dn_j ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_dn_dg_j * c_dn_i ));
            tJ.add_offdiagonal_Ham(std::complex<double>(0.5 * J,0.0) * (Sp_i * Sm_j));
            tJ.add_offdiagonal_Ham(std::complex<double>(0.5 * J,0.0) * (Sm_i * Sp_j));
            tJ.add_diagonal_Ham(std::complex<double>(J,0.0) * Sz_i * Sz_j);
            tJ.add_diagonal_Ham(std::complex<double>(-0.25 * J,0.0) * n_i * n_j);
            
            // top neighbor
            lattice.coor2site(std::vector<MKL_INT>{m,n+1}, 0, site_j);
            c_up_j = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
            c_dn_j = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
            c_up_dg_j = c_up_j; c_up_dg_j.dagger();
            c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
            Sp_j = c_up_dg_j * c_dn_j;
            Sm_j = c_dn_dg_j * c_up_j;
            Sz_j = std::complex<double>(0.5,0.0) * (c_up_dg_j * c_up_j - c_dn_dg_j * c_dn_j);
            n_j  = c_up_dg_j * c_up_j + c_dn_dg_j * c_dn_j;
            
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_up_dg_i * c_up_j ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_up_dg_j * c_up_i ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_dn_dg_i * c_dn_j ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_dn_dg_j * c_dn_i ));
            tJ.add_offdiagonal_Ham(std::complex<double>(0.5 * J,0.0) * (Sp_i * Sm_j));
            tJ.add_offdiagonal_Ham(std::complex<double>(0.5 * J,0.0) * (Sm_i * Sp_j));
            tJ.add_diagonal_Ham(std::complex<double>(J,0.0) * Sz_i * Sz_j);
            tJ.add_diagonal_Ham(std::complex<double>(-0.25 * J,0.0) * n_i * n_j);
        }
    }
    
    tJ.enumerate_basis_full_conserve(lattice.total_sites(), {"tJ"}, {N_up, N_dn}, {static_cast<double>(total_up),static_cast<double>(total_dn)});
    std::cout << "dim_all = " << tJ.dim_full << std::endl;
    
    // generating Hamiltonian matrix
    //tJ.generate_Ham_sparse_full(false);
    //std::cout << std::endl;
    
    //tJ.locate_E0_full();
    //std::cout << std::endl;
    
    
    MKL_INT i = 0, j = 0;
//    for (MKL_INT i = 0; i < lattice.Lx(); i++) {
//        for (MKL_INT j = 0; j < lattice.Ly(); j++) {
            tJ.basis_init_repr(std::vector<MKL_INT>{i,j}, lattice);
            tJ.generate_Ham_sparse_repr();
            std::cout << std::endl;
            
            tJ.locate_E0_repr();
            std::cout << std::endl;
//        }
//    }
    
    
    
}

// test Kondo lattice chain
void test_Kondo()
{
    
}

void test_trimer()
{
    qbasis::lattice lattice("chain",std::vector<MKL_INT>{3},std::vector<std::string>{"obc"});
    
    std::vector<std::vector<std::complex<double>>> Splus(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sminus(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sx(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sy(2,std::vector<std::complex<double>>(2));
    std::vector<std::complex<double>> Sz(2);
    Splus[0][0] = 0.0;
    Splus[0][1] = 1.0;
    Splus[1][0] = 0.0;
    Splus[1][1] = 0.0;
    Sminus[0][0] = 0.0;
    Sminus[0][1] = 0.0;
    Sminus[1][0] = 1.0;
    Sminus[1][1] = 0.0;
    Sx[0][0] = 0.0;
    Sx[0][1] = 0.5;
    Sx[1][0] = 0.5;
    Sx[1][1] = 0.0;
    Sy[0][0] = 0.0;
    Sy[0][1] = std::complex<double>(0.0,-0.5);
    Sy[1][0] = std::complex<double>(0.0,0.5);
    Sy[1][1] = 0.0;
    Sz[0] = 0.5;
    Sz[1] = -0.5;
    
    qbasis::model<std::complex<double>> trimer;
    qbasis::mopr<std::complex<double>> Sz_total;
    //trimer.dim_all = qbasis::int_pow(2, 3);
    //trimer.basis_all = std::vector<qbasis::mbasis_elem>(trimer.dim_all,qbasis::mbasis_elem(3,{"spin-1/2"}));
    
    qbasis::opr<std::complex<double>> s0x(0,0,false,Sx);
    qbasis::opr<std::complex<double>> s1x(1,0,false,Sx);
    qbasis::opr<std::complex<double>> s2x(2,0,false,Sx);
    qbasis::opr<std::complex<double>> s0y(0,0,false,Sy);
    qbasis::opr<std::complex<double>> s1y(1,0,false,Sy);
    qbasis::opr<std::complex<double>> s2y(2,0,false,Sy);
    qbasis::opr<std::complex<double>> s0z(0,0,false,Sz);
    qbasis::opr<std::complex<double>> s1z(1,0,false,Sz);
    qbasis::opr<std::complex<double>> s2z(2,0,false,Sz);
    
    trimer.add_offdiagonal_Ham(s0x * s1x + s0y * s1y);
    trimer.add_diagonal_Ham(s0z * s1z);
    trimer.add_offdiagonal_Ham(s1x * s2x + s1y * s2y);
    trimer.add_diagonal_Ham(s1z * s2z);
    
    Sz_total = (s0z + s1z + s2z);
    
    trimer.enumerate_basis_full_conserve(3, {"spin-1/2"}, {Sz_total}, {0.5});
    std::cout << "dim_all = " << trimer.dim_full << std::endl;
    
    for (MKL_INT j = 0; j < trimer.dim_full; j++) {
        std::cout << " j = " << j << std::endl;
        trimer.basis_full[j].prt_nonzero(); std::cout << std::endl;
    }
    
    trimer.generate_Ham_sparse_full(true);
    
    trimer.HamMat_csr_full.prt();
    
    trimer.locate_E0_full(2,3);
    for (MKL_INT j = 0; j < trimer.dim_full; j++) {
        std::cout << "j= "  << j << ", mod=" << std::abs(trimer.eigenvecs_full[j]) << ", allp= " << trimer.eigenvecs_full[j] << std::endl;
    }
    
    
}

void test_bubble() {
    std::vector<MKL_INT> val{3,1,10,2,5,12,9,-3};
    auto cnt = qbasis::bubble_sort(val, 0, 8);
    std::cout << "cnt = " << cnt << std::endl;
    std::cout << "vals: " << std::endl;
    for (MKL_INT j = 0; j < val.size(); j++) {
        std::cout << val[j] << "  ";
    }
    std::cout << std::endl;
}
