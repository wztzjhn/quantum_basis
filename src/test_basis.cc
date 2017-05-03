#include <iostream>
#include <iomanip>
#include <fstream>
#include "qbasis.h"

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
    
    props1.emplace_back(5,8);
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
    
    qbasis::lattice lattice("chain",std::vector<uint32_t>{5},std::vector<std::string>{"pbc"});
    
    qbasis::model<std::complex<double>> test_model;
    test_model.add_orbital(lattice.total_sites(), "spin-1/2");
    test_model.add_orbital(lattice.total_sites(), "electron");
    test_model.enumerate_basis_full(lattice);
    for (MKL_INT j = 0; j < test_model.dim_full; j++) {
        std::cout << "j= " << j << ", basis_label = " << test_model.basis_full[j].label(test_model.props) << std::endl;
        assert(j == test_model.basis_full[j].label(test_model.props));
    }
    
    
    
}
