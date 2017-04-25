#include <iostream>
#include <iomanip>
#include <fstream>
#include "qbasis.h"

void test_lattice() {
    qbasis::lattice square("square",std::vector<uint32_t>{3,3},std::vector<std::string>{"pbc", "pbc"});
    std::vector<int> coor = {1,2};
    int sub = 0;
    for (uint32_t site = 0; site < square.total_sites(); site++) {
        square.site2coor(coor, sub, site);
        std::cout << "(" << coor[0] << "," << coor[1] << "," << sub << ") : " << site << std::endl;
        uint32_t site2;
        square.coor2site(coor, sub, site2);
        assert(site == site2);
    }
    
    auto plan = square.translation_plan(std::vector<int>{2, 1});
    for (uint32_t j = 0; j < square.total_sites(); j++) {
        std::cout << j << " -> " << plan[j] << std::endl;
    }
    std::cout << std::endl;
}
