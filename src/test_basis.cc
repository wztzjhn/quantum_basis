#include <iostream>
#include <iomanip>
#include <fstream>
#include "qbasis.h"

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
