#include <iostream>
#include <iomanip>
#include "../../../quantum_basis/src/qbasis.h"

int main() {
    qbasis::initialize();
    std::cout << std::setprecision(10);

    // local matrix representation
    auto c_up = std::vector<std::vector<double>>(4,std::vector<double>(4, 0.0));
    auto c_dn = std::vector<std::vector<double>>(4,std::vector<double>(4, 0.0));
    c_up[0][1] = 1.0;
    c_up[2][3] = 1.0;
    c_dn[0][2] = 1.0;
    c_dn[1][3] = -1.0;

    // operators
    auto c_up_i0    = qbasis::opr<double>(0,0,true,c_up);
    auto c_dn_i0    = qbasis::opr<double>(0,0,true,c_dn);
    auto c_up_dg_i0 = c_up_i0; c_up_dg_i0.dagger();
    auto c_dn_dg_i0 = c_dn_i0; c_dn_dg_i0.dagger();
    auto n_up_i0    = c_up_dg_i0 * c_up_i0;
    auto n_dn_i0    = c_dn_dg_i0 * c_dn_i0;
    auto s_plus_i0  = c_up_dg_i0 * c_dn_i0;
    auto s_minus_i0 = c_dn_dg_i0 * c_up_i0;
    auto s_z_i0     = 0.5 * (c_up_dg_i0 * c_up_i0 - c_dn_dg_i0 * c_dn_i0);

    auto c_up_i1    = qbasis::opr<double>(0,1,true,c_up);
    auto c_dn_i1    = qbasis::opr<double>(0,1,true,c_dn);
    auto c_up_dg_i1 = c_up_i1; c_up_dg_i1.dagger();
    auto c_dn_dg_i1 = c_dn_i1; c_dn_dg_i1.dagger();
    auto n_up_i1    = c_up_dg_i1 * c_up_i1;
    auto n_dn_i1    = c_dn_dg_i1 * c_dn_i1;
    auto s_plus_i1  = c_up_dg_i1 * c_dn_i1;
    auto s_minus_i1 = c_dn_dg_i1 * c_up_i1;
    auto s_z_i1     = 0.5 * (c_up_dg_i1 * c_up_i1 - c_dn_dg_i1 * c_dn_i1);

    auto c_up_j0    = qbasis::opr<double>(1,0,true,c_up);
    auto c_dn_j0    = qbasis::opr<double>(1,0,true,c_dn);
    auto c_up_dg_j0 = c_up_j0; c_up_dg_j0.dagger();
    auto c_dn_dg_j0 = c_dn_j0; c_dn_dg_j0.dagger();
    auto n_up_j0    = c_up_dg_j0 * c_up_j0;
    auto n_dn_j0    = c_dn_dg_j0 * c_dn_j0;
    auto s_plus_j0  = c_up_dg_j0 * c_dn_j0;
    auto s_minus_j0 = c_dn_dg_j0 * c_up_j0;
    auto s_z_j0     = 0.5 * (c_up_dg_j0 * c_up_j0 - c_dn_dg_j0 * c_dn_j0);

    auto c_up_j1    = qbasis::opr<double>(1,1,true,c_up);
    auto c_dn_j1    = qbasis::opr<double>(1,1,true,c_dn);
    auto c_up_dg_j1 = c_up_j1; c_up_dg_j1.dagger();
    auto c_dn_dg_j1 = c_dn_j1; c_dn_dg_j1.dagger();
    auto n_up_j1    = c_up_dg_j1 * c_up_j1;
    auto n_dn_j1    = c_dn_dg_j1 * c_dn_j1;
    auto s_plus_j1  = c_up_dg_j1 * c_dn_j1;
    auto s_minus_j1 = c_dn_dg_j1 * c_up_j1;
    auto s_z_j1     = 0.5 * (c_up_dg_j1 * c_up_j1 - c_dn_dg_j1 * c_dn_j1);

    auto N_up_i = n_up_i0 + n_up_i1;
    auto N_dn_i = n_dn_i0 + n_dn_i1;
    auto N_up_j = n_up_j0 + n_up_j1;
    auto N_dn_j = n_dn_j0 + n_dn_j1;
    qbasis::mopr<double> N_total_i = N_up_i + N_dn_i;
    qbasis::mopr<double> N_total_j = N_up_j + N_dn_j;
    qbasis::mopr<double> N_total   = N_total_i + N_total_j;

    std::vector<qbasis::basis_prop> props;
    props.emplace_back(2, "electron");
    props.emplace_back(2, "electron");
    std::vector<qbasis::mbasis_elem> basis_22, basis_13, basis_31, basis_total;
    std::vector<qbasis::mopr<double>> conserve_list = {N_total_i, N_total_j};
    qbasis::enumerate_basis(props, basis_22, conserve_list, {2.0, 2.0});

    for (uint32_t i = 0; i < basis_22.size(); i++) {
        std::cout << "------------------- state " << i << " -------------------" << std::endl;
        basis_22[i].prt_states(props);
        std::cout << std::endl;
    }
    return 0;
    // |0>: c_{0 up}^dg c_{0 dn}^dg
    // |1>: c_{1 up}^dg c_{0 up}^dg
    // |2>: c_{1 up}^dg c_{0 dn}^dg
    // |3>: c_{1 dn}^dg c_{0 up}^dg
    // |4>: c_{1 dn}^dg c_{0 dn}^dg
    // |5>: c_{1 up}^dg c_{1 dn}^dg

//    qbasis::wavefunction<double> s(props), tp(props), tm(props), t0(props), d0(props), d1(props);
//    s  += std::pair<qbasis::mbasis_elem, double>(basis[3], -1.0 / std::sqrt(2.0));
//    s  += std::pair<qbasis::mbasis_elem, double>(basis[2],  1.0 / std::sqrt(2.0));
//    tp += std::pair<qbasis::mbasis_elem, double>(basis[1], -1.0);
//    tm += std::pair<qbasis::mbasis_elem, double>(basis[4], -1.0);
//    t0 += std::pair<qbasis::mbasis_elem, double>(basis[3], -1.0 / std::sqrt(2.0));
//    t0 += std::pair<qbasis::mbasis_elem, double>(basis[2], -1.0 / std::sqrt(2.0));
//    d0 += std::pair<qbasis::mbasis_elem, double>(basis[0], 1.0);
//    d1 += std::pair<qbasis::mbasis_elem, double>(basis[5], 1.0);
//
////    auto Ham = n_up_i1 + n_dn_i1; // Delta
////    auto Ham = n_up_i0 * n_dn_i0 + n_up_i1 * n_dn_i1; // U
////    auto Ham = (n_up_i0 + n_dn_i0) * (n_up_i1 + n_dn_i1); // (U'-J/2)
////    auto Ham = 0.5 * (s_plus_i0 * s_minus_i1 + s_minus_i0 * s_plus_i1) + s_z_i0 * s_z_i1; // -2J
////    auto Ham =  (c_up_dg_i0 * c_dn_dg_i0 * c_dn_i1 * c_up_i1)
////              + (c_up_dg_i1 * c_dn_dg_i1 * c_dn_i0 * c_up_i0); // J'
//
////    auto Ham = 0.5 * (  1.0 * (s_plus_i0 * s_minus_i1 + s_minus_i0 * s_plus_i1) + 2.0 * s_z_i0 * s_z_i1
////                      + 0.5 * (s_plus_i0 * s_minus_i0 + s_minus_i0 * s_plus_i0) + 1.0 * s_z_i0 * s_z_i0
////                      + 0.5 * (s_plus_i1 * s_minus_i1 + s_minus_i1 * s_plus_i1) + 1.0 * s_z_i1 * s_z_i1 );
////    auto proj_tp = (n_up_i0 - n_up_i0 * n_dn_i0) * (n_up_i1 - n_up_i1 * n_dn_i1);
////    auto proj_tm = (n_dn_i0 - n_dn_i0 * n_up_i0) * (n_dn_i1 - n_dn_i1 * n_up_i1);
////    Ham -= (proj_tp * Ham + proj_tm * Ham); // |t0><t0|
//    auto Ham = (c_up_dg_i0 * c_dn_dg_i0 * c_dn_i1 * c_up_i1);
//    qbasis::wavefunction<double> res(props);
//
//    std::cout << "matrix: " << std::endl;
//
//    qbasis::oprXphi(Ham, props, res, s);
////    std::cout << "H | s >:" << std::endl;
////    res.prt_states(props);
//    std::cout << qbasis::inner_product(res, s)  << "\t";
//    std::cout << qbasis::inner_product(res, tp) << "\t";
//    std::cout << qbasis::inner_product(res, tm) << "\t";
//    std::cout << qbasis::inner_product(res, t0) << "\t";
//    std::cout << qbasis::inner_product(res, d0) << "\t";
//    std::cout << qbasis::inner_product(res, d1) << std::endl;
//
//    qbasis::oprXphi(Ham, props, res, tp);
////    std::cout << "H | tp >:" << std::endl;
////    res.prt_states(props);
//    std::cout << qbasis::inner_product(res, s)  << "\t";
//    std::cout << qbasis::inner_product(res, tp) << "\t";
//    std::cout << qbasis::inner_product(res, tm) << "\t";
//    std::cout << qbasis::inner_product(res, t0) << "\t";
//    std::cout << qbasis::inner_product(res, d0) << "\t";
//    std::cout << qbasis::inner_product(res, d1) << std::endl;
//
//    qbasis::oprXphi(Ham, props, res, tm);
////    std::cout << "H | tm >:" << std::endl;
////    res.prt_states(props);
//    std::cout << qbasis::inner_product(res, s)  << "\t";
//    std::cout << qbasis::inner_product(res, tp) << "\t";
//    std::cout << qbasis::inner_product(res, tm) << "\t";
//    std::cout << qbasis::inner_product(res, t0) << "\t";
//    std::cout << qbasis::inner_product(res, d0) << "\t";
//    std::cout << qbasis::inner_product(res, d1) << std::endl;
//
//    qbasis::oprXphi(Ham, props, res, t0);
////    std::cout << "H | t0 >:" << std::endl;
////    res.prt_states(props);
//    std::cout << qbasis::inner_product(res, s)  << "\t";
//    std::cout << qbasis::inner_product(res, tp) << "\t";
//    std::cout << qbasis::inner_product(res, tm) << "\t";
//    std::cout << qbasis::inner_product(res, t0) << "\t";
//    std::cout << qbasis::inner_product(res, d0) << "\t";
//    std::cout << qbasis::inner_product(res, d1) << std::endl;
//
//    qbasis::oprXphi(Ham, props, res, d0);
////    std::cout << "H | d0 >:" << std::endl;
////    res.prt_states(props);
//    std::cout << qbasis::inner_product(res, s)  << "\t";
//    std::cout << qbasis::inner_product(res, tp) << "\t";
//    std::cout << qbasis::inner_product(res, tm) << "\t";
//    std::cout << qbasis::inner_product(res, t0) << "\t";
//    std::cout << qbasis::inner_product(res, d0) << "\t";
//    std::cout << qbasis::inner_product(res, d1) << std::endl;
//
//    qbasis::oprXphi(Ham, props, res, d1);
////    std::cout << "H | d1 >:" << std::endl;
////    res.prt_states(props);
//    std::cout << qbasis::inner_product(res, s)  << "\t";
//    std::cout << qbasis::inner_product(res, tp) << "\t";
//    std::cout << qbasis::inner_product(res, tm) << "\t";
//    std::cout << qbasis::inner_product(res, t0) << "\t";
//    std::cout << qbasis::inner_product(res, d0) << "\t";
//    std::cout << qbasis::inner_product(res, d1) << std::endl;


}