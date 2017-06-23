#include <iostream>
#include <iomanip>
#include <fstream>
#include "qbasis.h"
#include "graph.h"

void test_basis() {
    std::cout << "--------- test basis ---------" << std::endl;
    std::vector<qbasis::basis_prop> props1;
    props1.emplace_back(9,"spin-1/2");
    
    qbasis::mbasis_elem basis1(props1);
    basis1.prt_bits(props1);
    basis1.increment(props1);
    basis1.siteWrite(props1, 3, 0, 1);
    basis1.prt_bits(props1);
    basis1.prt_states(props1);
    assert(basis1.siteRead(props1, 0, 0) == 1);
    assert(basis1.siteRead(props1, 1, 0) == 0);
    assert(basis1.siteRead(props1, 2, 0) == 0);
    assert(basis1.siteRead(props1, 3, 0) == 1);
    basis1.siteWrite(props1, 7, 0, 1);
    basis1.prt_bits(props1);
    basis1.siteWrite(props1, 8, 0, 1);
    basis1.prt_bits(props1);
    assert(basis1.siteRead(props1, 4, 0) == 0);
    assert(basis1.siteRead(props1, 5, 0) == 0);
    assert(basis1.siteRead(props1, 6, 0) == 0);
    assert(basis1.siteRead(props1, 7, 0) == 1);
    assert(basis1.siteRead(props1, 8, 0) == 1);
    std::cout << std::endl;
    
    auto basis_list1 = qbasis::enumerate_basis_all(props1);
    std::cout << "basis_list1: " << std::endl;
    for (uint64_t j = 0; j < basis_list1.size(); j++) {
        std::cout << "j = " << j << "\t";
        basis_list1[j].prt_bits(props1);
        assert(j == basis_list1[j].label(props1));
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    props1.emplace_back(9,8);
    std::cout << "props1[1].bits_per_site = " << static_cast<unsigned>(props1[1].bits_per_site) << std::endl;
    qbasis::mbasis_elem basis2(props1);
    basis2.siteWrite(props1, 2, 0, 1);
    basis2.siteWrite(props1, 8, 0, 1);
    basis2.siteWrite(props1, 1, 1, 1);
    basis2.siteWrite(props1, 2, 1, 6);
    basis2.siteWrite(props1, 4, 1, 2);
    basis2.prt_bits(props1);
    basis2.prt_states(props1);
    assert(basis2.siteRead(props1, 0, 0) == 0);
    assert(basis2.siteRead(props1, 1, 0) == 0);
    assert(basis2.siteRead(props1, 2, 0) == 1);
    assert(basis2.siteRead(props1, 3, 0) == 0);
    assert(basis2.siteRead(props1, 4, 0) == 0);
    assert(basis2.siteRead(props1, 5, 0) == 0);
    assert(basis2.siteRead(props1, 6, 0) == 0);
    assert(basis2.siteRead(props1, 7, 0) == 0);
    assert(basis2.siteRead(props1, 8, 0) == 1);
    assert(basis2.siteRead(props1, 0, 1) == 0);
    assert(basis2.siteRead(props1, 1, 1) == 1);
    assert(basis2.siteRead(props1, 2, 1) == 6);
    assert(basis2.siteRead(props1, 3, 1) == 0);
    assert(basis2.siteRead(props1, 4, 1) == 2);
    assert(basis2.siteRead(props1, 5, 1) == 0);
    std::cout << std::endl;
    
    std::cout << "haha000" << std::endl;
    auto basis3 = basis2;
    std::cout << "haha001" << std::endl;
    basis3.increment(props1, 0);
    assert(basis2 < basis3);
    std::cout << "haha002" << std::endl;
    basis3 = basis2;
    std::cout << "haha003" << std::endl;
    basis3.increment(props1, 1);
    assert(basis2 < basis3);
    basis3 = basis2;
    basis3.increment(props1);
    assert(basis2 < basis3);
    std::cout << std::endl;
    
    std::cout << "haha004" << std::endl;
    auto basis4 = std::move(basis1);
    std::cout << "haha005" << std::endl;
    
    std::vector<qbasis::mbasis_elem> basis_vec{basis2, basis3};
    
    std::cout << std::endl << "haha006" << std::endl;
    basis_vec[0].prt_bits(props1);
    basis_vec[1].prt_bits(props1);
    
    std::vector<qbasis::mbasis_elem> basis_vec2(2);
    
    std::move(basis_vec.begin(),basis_vec.end(),basis_vec2.begin());
    
    
    std::cout << std::endl << "haha007" << std::endl;
    basis_vec2[0].prt_bits(props1);

//    mele2 = mele1;
//    auto stats2 = mele1.statistics();
//    for (MKL_INT j = 0; j < stats2.size(); j++) {
//        std::cout << "stat " << j << ", count = " << stats2[j] << std::endl;
//    }
//    
//    qbasis::lattice square("square",std::vector<MKL_INT>{3,3},std::vector<std::string>{"pbc", "pbc"});
//    MKL_INT sgn;
//    mele1.translate(square, std::vector<MKL_INT>{1,2}, sgn);
//    std::cout << "translational equiv?: " << qbasis::trans_equiv(mele1, mele2, square) << std::endl;
//    
//    std::cout << std::endl;
    
    
//    qbasis::wavefunction<std::complex<double>> phi(basis3);
//    phi.prt_states(props1);
//    
//    phi += basis2;
//    phi.prt_states(props1);
    
    qbasis::lattice lattice("chain",std::vector<uint32_t>{3},std::vector<std::string>{"pbc"});
    
    qbasis::model<std::complex<double>> test_model;
    test_model.add_orbital(lattice.total_sites(), "spin-1/2");
    test_model.add_orbital(lattice.total_sites(), "electron");
    test_model.enumerate_basis_full(lattice);
    for (MKL_INT j = 0; j < test_model.dim_full; j++) {
        std::cout << "j= " << j << ", basis_label = " << test_model.basis_full[j].label(test_model.props) << std::endl;
        //assert(j == test_model.basis_full[j].label(test_model.props));
    }
    

}

void test_basis2()
{
    std::cout << "test basis2 " << std::endl;
    std::vector<qbasis::basis_prop> props;
    props.emplace_back(4,"spin-1/2");
    
    qbasis::lattice latt("chain",std::vector<uint32_t>{4},std::vector<std::string>{"pbc"});
    
    auto basis_list = qbasis::enumerate_basis_all(props);
    
    std::vector<qbasis::mbasis_elem> reps;
    std::vector<uint64_t> belong2rep;
    std::vector<std::vector<int>> dist2rep;
    qbasis::classify_trans_full2rep(props, basis_list, latt, std::vector<bool>{true}, reps, belong2rep, dist2rep);
    
    std::cout << " ------- reps -------" << std::endl;
    for (uint64_t j = 0; j < reps.size(); j++) {
        std::cout << "i = " << j << ", ";
        reps[j].prt_bits(props);
    }
    
    std::cout << " ------- basis -------" << std::endl;
    for (uint64_t j = 0; j < basis_list.size(); j++) {
        std::cout << "j = " << j << ", ";
        basis_list[j].prt_bits(props);
        std::cout << "r = " << belong2rep[j] << ", dist = " << dist2rep[j][0] << std::endl << std::endl;
    }
    
    std::cout << "haha" << std::endl;
    
    assert(belong2rep[0] == 0);
    assert(belong2rep[1] == 1);
    assert(belong2rep[2] == 1);
    assert(belong2rep[3] == 2);
    assert(belong2rep[4] == 1);
    assert(belong2rep[5] == 3);
    assert(belong2rep[6] == 2);
    assert(belong2rep[7] == 4);
    assert(belong2rep[8] == 1);
    assert(belong2rep[9] == 2);
    assert(belong2rep[10] == 3);
    assert(belong2rep[11] == 4);
    assert(belong2rep[12] == 2);
    assert(belong2rep[13] == 4);
    assert(belong2rep[14] == 4);
    assert(belong2rep[15] == 5);
    
    
    std::vector<std::vector<uint32_t>> groups;
    std::vector<uint32_t> omega_g;
    std::vector<uint32_t> belong2group;
    qbasis::classify_trans_rep2group(props, reps, latt, std::vector<bool>{true}, groups, omega_g, belong2group);
    for (uint32_t j = 0; j < groups.size(); j++) {
        std::cout << "group: " << groups[j][0] << ", omega_g = " << omega_g[j] << std::endl;
    }
    for (uint64_t j = 0; j < reps.size(); j++) {
        std::cout << "j = " << j <<  std::endl;
        reps[j].prt_bits(props);
        std::cout << "belong to group: " << belong2group[j] << std::endl;
    }
    
}

void test_basis3()
{
    std::cout << "test basis3 " << std::endl;
    qbasis::lattice latt("triangular",std::vector<uint32_t>{2,3},std::vector<std::string>{"pbc","pbc"});
    auto latt_child = qbasis::divide_lattice(latt);
    
    std::vector<qbasis::basis_prop> props;
    props.emplace_back(latt_child.total_sites(),"spin-1/2");
    
    auto basis_list = qbasis::enumerate_basis_all(props);
    
    std::vector<qbasis::mbasis_elem> reps;
    std::vector<uint64_t> belong2rep;
    std::vector<std::vector<int>> dist2rep;
    qbasis::classify_trans_full2rep(props, basis_list, latt_child, std::vector<bool>{true,true}, reps,belong2rep, dist2rep);
    
    for (uint64_t j = 0; j < basis_list.size(); j++) {
        std::cout << "j = " << j << ", ";
        basis_list[j].prt_bits(props);
        std::cout << "r = " << belong2rep[j] << ", dist = " << dist2rep[j][0] << "," << dist2rep[j][1] << std::endl << std::endl;
    }
    
    std::cout << "haha" << std::endl;
}

void test_basis4()
{
    std::cout << "test basis 4" << std::endl;
    
    // local matrix representation
    // Spins:
    std::vector<std::vector<std::complex<double>>> Splus(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sminus(2,std::vector<std::complex<double>>(2));
    std::vector<std::complex<double>> Sz(2);
    Splus[0][0]  = 0.0;
    Splus[0][1]  = 1.0;
    Splus[1][0]  = 0.0;
    Splus[1][1]  = 0.0;
    Sminus[0][0] = 0.0;
    Sminus[0][1] = 0.0;
    Sminus[1][0] = 1.0;
    Sminus[1][1] = 0.0;
    Sz[0]        = 0.5;
    Sz[1]        = -0.5;
    
    qbasis::lattice latt("chain",std::vector<uint32_t>{6},std::vector<std::string>{"pbc"});
//    std::vector<qbasis::basis_prop> props;
//    props.emplace_back(latt.total_sites(),"spin-1/2");
    
    qbasis::model<std::complex<double>> model_test4;
    model_test4.add_orbital(latt.total_sites(), "spin-1/2");
    qbasis::mopr<std::complex<double>> Sz_total;
    
    
    for (int x = 0; x < latt.total_sites(); x++) {
        uint32_t site_i;
        latt.coor2site(std::vector<int>{x}, 0, site_i); // obtain site label of (x)
        auto Sz_i      = qbasis::opr<std::complex<double>>(site_i,0,false,Sz);
        Sz_total += Sz_i;
    }
    model_test4.enumerate_basis_full(latt, {Sz_total}, {0.0});
    
    std::cout << std::numeric_limits<double>::epsilon() << std::endl;
    
}
