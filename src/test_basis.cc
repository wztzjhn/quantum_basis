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
    
    std::vector<qbasis::mbasis_elem> basis_list1;
    qbasis::enumerate_basis<double>(props1, basis_list1);
    qbasis::sort_basis_normal_order(basis_list1);
    //std::cout << "basis_list1: " << std::endl;
    for (uint64_t j = 0; j < basis_list1.size(); j++) {
        //std::cout << "j = " << j << "\t";
        //basis_list1[j].prt_bits(props1);
        assert(j == basis_list1[j].label(props1));
        //std::cout << std::endl;
    }
    //std::cout << std::endl;
    
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
    test_model.enumerate_basis_full();
    /*
    for (MKL_INT j = 0; j < test_model.dim_full; j++) {
        std::cout << "j= " << j << ", basis_label = " << test_model.basis_full[j].label(test_model.props) << std::endl;
    }
    */

}

void test_basis2()
{
    std::cout << "--------- test basis2 ---------" << std::endl;
    std::vector<qbasis::basis_prop> props;
    props.emplace_back(4,"spin-1/2");
    
    qbasis::lattice latt("chain",std::vector<uint32_t>{4},std::vector<std::string>{"pbc"});
    
    std::vector<qbasis::mbasis_elem> basis_list;
    qbasis::enumerate_basis<double>(props, basis_list);
    qbasis::sort_basis_normal_order(basis_list);
    
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
    std::vector<qbasis::mbasis_elem> group_examples;
    qbasis::classify_trans_rep2group(props, reps, latt, std::vector<bool>{true}, groups, omega_g, belong2group);
    for (uint32_t j = 0; j < groups.size(); j++) {
        std::cout << "group: " << groups[j][0] << ", omega_g = " << omega_g[j] << std::endl;
    }
    for (uint64_t j = 0; j < reps.size(); j++) {
        std::cout << "j = " << j <<  std::endl;
        reps[j].prt_bits(props);
        std::cout << "belong to group: " << belong2group[j] << std::endl;
    }
    std::cout << std::endl;
    
}

void test_basis3()
{
    std::cout << "--------- test basis3 ---------" << std::endl;
    qbasis::lattice latt("triangular",std::vector<uint32_t>{2,3},std::vector<std::string>{"pbc","pbc"});
    auto latt_child = qbasis::divide_lattice(latt);
    
    std::vector<qbasis::basis_prop> props;
    props.emplace_back(latt_child.total_sites(),"spin-1/2");
    
    std::vector<qbasis::mbasis_elem> basis_list;
    qbasis::enumerate_basis<double>(props, basis_list);
    qbasis::sort_basis_normal_order(basis_list);
    
    std::vector<qbasis::mbasis_elem> reps;
    std::vector<uint64_t> belong2rep;
    std::vector<std::vector<int>> dist2rep;
    qbasis::classify_trans_full2rep(props, basis_list, latt_child, std::vector<bool>{true,true}, reps,belong2rep, dist2rep);
    
    for (uint64_t j = 0; j < basis_list.size(); j++) {
        std::cout << "j = " << j << ", ";
        basis_list[j].prt_bits(props);
        std::cout << "r = " << belong2rep[j] << ", dist = " << dist2rep[j][0] << "," << dist2rep[j][1] << std::endl << std::endl;
    }
    
    std::cout << "haha" << std::endl << std::endl;
}

void test_basis4()
{
    std::cout << "--------- test basis4 (check divide and conquer on 1D chain) ---------" << std::endl;
    
    uint32_t L = 8;
    
    // local matrix representation
    // Spins:
    std::vector<std::vector<std::complex<double>>> Splus(3,std::vector<std::complex<double>>(3));
    std::vector<std::vector<std::complex<double>>> Sminus(3,std::vector<std::complex<double>>(3));
    std::vector<std::complex<double>> Sz(3);
    Splus[0][0]  = 0.0;
    Splus[0][1]  = sqrt(2.0);
    Splus[0][2]  = 0.0;
    Splus[1][0]  = 0.0;
    Splus[1][1]  = 0.0;
    Splus[1][2]  = sqrt(2.0);
    Sminus[0][0] = 0.0;
    Sminus[0][1] = 0.0;
    Sminus[0][2] = 0.0;
    Sminus[1][0] = sqrt(2.0);
    Sminus[1][1] = 0.0;
    Sminus[1][2] = 0.0;
    Sminus[2][1] = 0.0;
    Sminus[2][1] = sqrt(2.0);
    Sminus[2][2] = 0.0;
    Sz[0]        = 1.0;
    Sz[1]        = 0.0;
    Sz[2]        = -1.0;
    
    std::vector<std::string> bc{"pbc"};
    qbasis::lattice lattice("chain",std::vector<uint32_t>{L},bc);
    
    qbasis::model<std::complex<double>> model_test4;
    model_test4.add_orbital(lattice.total_sites(), "spin-1");
    
    qbasis::mopr<std::complex<double>> Sz_total;
    
    for (int x = 0; x < L; x++) {
        uint32_t site_i, site_j;
        lattice.coor2site(std::vector<int>{x}, 0, site_i); // obtain site label of (x)
        // construct operators on each site
        // spin
        auto Splus_i   = qbasis::opr<std::complex<double>>(site_i,0,false,Splus);
        auto Sminus_i  = qbasis::opr<std::complex<double>>(site_i,0,false,Sminus);
        auto Sz_i      = qbasis::opr<std::complex<double>>(site_i,0,false,Sz);
        
        // with neighbor (x+1)
        if (bc[0] == "pbc" || (bc[0] == "obc" && x < L - 1)) {
            lattice.coor2site(std::vector<int>{x+1}, 0, site_j);
            // spin exchanges
            auto Splus_j   = qbasis::opr<std::complex<double>>(site_j,0,false,Splus);
            auto Sminus_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Sminus);
            auto Sz_j      = qbasis::opr<std::complex<double>>(site_j,0,false,Sz);
            model_test4.add_offdiagonal_Ham(std::complex<double>(0.5,0.0) * (Splus_i * Sminus_j + Sminus_i * Splus_j));
            model_test4.add_diagonal_Ham(std::complex<double>(1.0,0.0) * (Sz_i * Sz_j));
        }
        
        Sz_total += Sz_i;
    }
    
    model_test4.enumerate_basis_full({Sz_total}, {0.0});
    
    std::cout << std::numeric_limits<double>::epsilon() << std::endl;
    
    model_test4.fill_Weisse_table(lattice);
    std::pair<std::vector<uint32_t>,std::vector<uint32_t>> check;
    check.first  = std::vector<uint32_t>{0};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{0,0,0,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{0,1,0,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{0,2,0,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{1,0,0,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{1,2,0,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,0,0,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,1,0,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,0,0}) == check);
    check.first  = std::vector<uint32_t>{2};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{0,1,0,1}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{0,2,0,1}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{1,0,1,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{1,2,1,1}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,0,1,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,1,1,1}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,1,1}) == check);
    check.first  = std::vector<uint32_t>{4};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{0,2,0,2}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{1,2,0,2}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,0,2,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,1,2,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,2,2}) == check);
    check.first  = std::vector<uint32_t>{6};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{0,2,0,3}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{1,2,1,3}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,0,3,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,1,3,1}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,3,3}) == check);
    check.first  = std::vector<uint32_t>{0};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{1,2,0,1}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,1,0,1}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,0,1}) == check);
    check.first  = std::vector<uint32_t>{2};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{1,2,1,2}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,1,1,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,1,2}) == check);
    check.first  = std::vector<uint32_t>{4};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{1,2,0,3}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,1,2,1}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,2,3}) == check);
    check.first  = std::vector<uint32_t>{6};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{1,2,1,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,1,3,0}) == check);
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,3,0}) == check);
    check.first  = std::vector<uint32_t>{0};
    check.second = std::vector<uint32_t>{3};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,0,3}) == check);
    check.first  = std::vector<uint32_t>{2};
    check.second = std::vector<uint32_t>{3};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,1,0}) == check);
    check.first  = std::vector<uint32_t>{4};
    check.second = std::vector<uint32_t>{3};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,2,1}) == check);
    check.first  = std::vector<uint32_t>{6};
    check.second = std::vector<uint32_t>{3};
    assert(model_test4.Weisse_e_lt.index(std::vector<uint64_t>{2,2,3,2}) == check);
    
    check.first  = std::vector<uint32_t>{1};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{0,0,0,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{0,1,0,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{0,2,0,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{1,0,1,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{1,2,1,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,0,1,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,1,1,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,1,0}) == check);
    check.first  = std::vector<uint32_t>{3};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{0,1,0,1}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{0,2,0,1}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{1,0,0,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{1,2,0,1}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,0,2,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,1,2,1}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,2,1}) == check);
    check.first  = std::vector<uint32_t>{5};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{0,2,0,2}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{1,2,1,2}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,0,3,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,1,3,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,3,2}) == check);
    check.first  = std::vector<uint32_t>{7};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{0,2,0,3}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{1,2,0,3}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,0,0,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,1,0,1}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,0,3}) == check);
    check.first  = std::vector<uint32_t>{1};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{1,2,0,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,1,2,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,2,0}) == check);
    check.first  = std::vector<uint32_t>{3};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{1,2,1,1}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,1,3,1}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,3,1}) == check);
    check.first  = std::vector<uint32_t>{5};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{1,2,0,2}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,1,0,0}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,0,2}) == check);
    check.first  = std::vector<uint32_t>{7};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{1,2,1,3}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,1,1,1}) == check);
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,1,3}) == check);
    check.first  = std::vector<uint32_t>{1};
    check.second = std::vector<uint32_t>{2};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,3,0}) == check);
    check.first  = std::vector<uint32_t>{3};
    check.second = std::vector<uint32_t>{2};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,0,1}) == check);
    check.first  = std::vector<uint32_t>{5};
    check.second = std::vector<uint32_t>{2};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,1,2}) == check);
    check.first  = std::vector<uint32_t>{7};
    check.second = std::vector<uint32_t>{2};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,2,3}) == check);
    check.first  = std::vector<uint32_t>{1};
    check.second = std::vector<uint32_t>{3};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,0,0}) == check);
    check.first  = std::vector<uint32_t>{3};
    check.second = std::vector<uint32_t>{3};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,1,1}) == check);
    check.first  = std::vector<uint32_t>{5};
    check.second = std::vector<uint32_t>{3};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,2,2}) == check);
    check.first  = std::vector<uint32_t>{7};
    check.second = std::vector<uint32_t>{3};
    assert(model_test4.Weisse_e_gt.index(std::vector<uint64_t>{2,2,3,3}) == check);
    
    check.first  = std::vector<uint32_t>{0};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{0,0,0,0}) == check);
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{1,1,0,0}) == check);
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,0,0}) == check);
    check.first  = std::vector<uint32_t>{1};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{1,1,1,0}) == check);
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,1,0}) == check);
    check.first  = std::vector<uint32_t>{2};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{1,1,1,1}) == check);
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,1,1}) == check);
    check.first  = std::vector<uint32_t>{3};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{1,1,0,1}) == check);
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,2,1}) == check);
    check.first  = std::vector<uint32_t>{4};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,2,2}) == check);
    check.first  = std::vector<uint32_t>{5};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,3,2}) == check);
    check.first  = std::vector<uint32_t>{6};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,3,3}) == check);
    check.first  = std::vector<uint32_t>{7};
    check.second = std::vector<uint32_t>{0};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,0,3}) == check);
    check.first  = std::vector<uint32_t>{1};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,2,0}) == check);
    check.first  = std::vector<uint32_t>{2};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,1,2}) == check);
    check.first  = std::vector<uint32_t>{3};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,3,1}) == check);
    check.first  = std::vector<uint32_t>{4};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,2,3}) == check);
    check.first  = std::vector<uint32_t>{5};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,0,2}) == check);
    check.first  = std::vector<uint32_t>{6};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,3,0}) == check);
    check.first  = std::vector<uint32_t>{7};
    check.second = std::vector<uint32_t>{1};
    assert(model_test4.Weisse_e_eq.index(std::vector<uint64_t>{2,2,1,3}) == check);
    
    
    
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{0,0,0}) == std::vector<uint32_t>{2});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{0,1,0}) == std::vector<uint32_t>{4});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{0,2,0}) == std::vector<uint32_t>{8});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{1,0,0}) == std::vector<uint32_t>{4});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{1,2,0}) == std::vector<uint32_t>{8});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{1,2,1}) == std::vector<uint32_t>{8});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{2,0,0}) == std::vector<uint32_t>{8});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{2,1,0}) == std::vector<uint32_t>{8});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{2,1,1}) == std::vector<uint32_t>{8});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{2,2,0}) == std::vector<uint32_t>{8});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{2,2,1}) == std::vector<uint32_t>{8});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{2,2,2}) == std::vector<uint32_t>{8});
    assert(model_test4.Weisse_w_lt.index(std::vector<uint64_t>{2,2,3}) == std::vector<uint32_t>{8});
    
    assert(model_test4.Weisse_w_eq.index(std::vector<uint64_t>{0,0,0}) == std::vector<uint32_t>{1});
    assert(model_test4.Weisse_w_eq.index(std::vector<uint64_t>{1,1,0}) == std::vector<uint32_t>{4});
    assert(model_test4.Weisse_w_eq.index(std::vector<uint64_t>{2,2,0}) == std::vector<uint32_t>{8});
    assert(model_test4.Weisse_w_eq.index(std::vector<uint64_t>{2,2,1}) == std::vector<uint32_t>{8});
    
    
    
    model_test4.generate_Ham_sparse_full();
    
    std::cout << "E0 from full basis: " << std::endl;
    model_test4.locate_E0_full();
    
    
    int momentum = 2;
    
    model_test4.enumerate_basis_repr(lattice, std::vector<int>{momentum}, {Sz_total}, {0.0});
    
//    std::cout << "label for basis_repr[0]: " << model_test4.basis_repr[model_test4.sec_repr][0].label(model_test4.props) << std::endl;
//    auto xxxxxx =  qbasis::norm_trans_repr(model_test4.props, model_test4.basis_repr[model_test4.sec_repr][0],
//                                           lattice, std::vector<uint32_t>{8}, std::vector<int>{0});
//    std::cout << "xxxxx = " << xxxxxx << std::endl;
//    std::cout << "nu[0] = " << model_test4.norm_repr[model_test4.sec_repr][0] << std::endl;
    
    model_test4.generate_Ham_sparse_repr(lattice, std::vector<int>{momentum});
    //auto xx = model_test4.HamMat_csr_repr[model_test4.sec_repr].to_dense();
    
    
    
    std::cout << "E0 from repr basis: " << std::endl;
    model_test4.locate_E0_repr();
    
    
    model_test4.basis_init_repr_deprecated(std::vector<int>{momentum}, lattice);
    std::cout << "dim_repr = " << model_test4.dimension_repr() << std::endl;
    
    model_test4.generate_Ham_sparse_repr_deprecated();
    //auto yy = model_test4.HamMat_csr_repr[model_test4.sec_repr].to_dense();
    
    std::cout << "E0 from repr basis(deprec): " << std::endl;
    model_test4.locate_E0_repr();
    
    assert(std::abs(model_test4.eigenvals_full[0] + 11.337) < 0.0001);
    assert(std::abs(model_test4.eigenvals_full[1] + 10.7434) < 0.0001);
    std::cout << std::endl;
    
//    for (MKL_INT i = 0; i < model_test4.dimension_repr(); i++) {
//        for (MKL_INT j = i; j < model_test4.dimension_repr(); j++) {
//            auto pos = i + j * model_test4.dimension_repr();
//            std::cout << "(i,j) = (" << i << "," << j << ")" << std::endl;
//            std::cout << "xx, yy = " << xx[pos] << ", " << yy[pos] << std::endl;
//            assert(std::abs(xx[pos] - yy[pos]) < 0.000001);
//        }
//    }
    
}



void test_basis5()
{
    std::cout << "--------- test basis5 (check divide and conquer on square lattice) ---------" << std::endl;
    
    uint32_t Lx = 2, Ly = 4;
    
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
    
    std::vector<std::string> bc{"pbc", "pbc"};
    qbasis::lattice lattice("square",std::vector<uint32_t>{Lx, Ly},bc);
    
    qbasis::model<std::complex<double>> model_test5;
    model_test5.add_orbital(lattice.total_sites(), "spin-1/2");
    
    qbasis::mopr<std::complex<double>> Sz_total;
    
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            uint32_t site_i, site_j;
            lattice.coor2site(std::vector<int>{x,y}, 0, site_i); // obtain site label of (x,y)
            // construct operators on each site
            // spin
            auto Splus_i   = qbasis::opr<std::complex<double>>(site_i,0,false,Splus);
            auto Sminus_i  = qbasis::opr<std::complex<double>>(site_i,0,false,Sminus);
            auto Sz_i      = qbasis::opr<std::complex<double>>(site_i,0,false,Sz);
        
            // with neighbor (x+1, y)
            if (bc[0] == "pbc" || (bc[0] == "obc" && x < Lx - 1)) {
                lattice.coor2site(std::vector<int>{x+1,y}, 0, site_j);
                // spin exchanges
                auto Splus_j   = qbasis::opr<std::complex<double>>(site_j,0,false,Splus);
                auto Sminus_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Sminus);
                auto Sz_j      = qbasis::opr<std::complex<double>>(site_j,0,false,Sz);
                model_test5.add_offdiagonal_Ham(std::complex<double>(0.5,0.0) * (Splus_i * Sminus_j + Sminus_i * Splus_j));
                model_test5.add_diagonal_Ham(std::complex<double>(1.0,0.0) * (Sz_i * Sz_j));
            }
            
            // with neighbor (x, y+1)
            if (bc[1] == "pbc" || (bc[1] == "obc" && y < Ly - 1)) {
                lattice.coor2site(std::vector<int>{x,y+1}, 0, site_j);
                auto Splus_j   = qbasis::opr<std::complex<double>>(site_j,0,false,Splus);
                auto Sminus_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Sminus);
                auto Sz_j      = qbasis::opr<std::complex<double>>(site_j,0,false,Sz);
                model_test5.add_offdiagonal_Ham(std::complex<double>(0.5,0.0) * (Splus_i * Sminus_j + Sminus_i * Splus_j));
                model_test5.add_diagonal_Ham(std::complex<double>(1.0,0.0) * (Sz_i * Sz_j));
            }
            
            Sz_total += Sz_i;
        }
    }
    
    model_test5.enumerate_basis_full({Sz_total}, {0.0});
    
    std::cout << std::numeric_limits<double>::epsilon() << std::endl;
    
    model_test5.fill_Weisse_table(lattice);
//    std::pair<std::vector<uint32_t>,std::vector<uint32_t>> check;
//    check.first  = std::vector<uint32_t>{0};
//    check.second = std::vector<uint32_t>{0};
//    assert(model_test5.table_lt.index(std::vector<uint64_t>{0,0,0,0}) == check);

    model_test5.generate_Ham_sparse_full();
    
    model_test5.locate_E0_full();
    
//    assert(std::abs(model_test5.eigenvals_full[0] + 3.65109) < 0.00001);
//    assert(std::abs(model_test5.eigenvals_full[1] + 3.12842) < 0.00001);
    std::cout << std::endl;
    
}



