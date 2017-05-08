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
    
    std::cout << "kagome" << std::endl;
    qbasis::lattice kagome("kagome",std::vector<uint32_t>{2,3},std::vector<std::string>{"pbc", "pbc"});
    for (uint32_t site = 0; site < kagome.total_sites(); site++) {
        kagome.site2coor(coor, sub, site);
        std::cout << "(" << coor[0] << "," << coor[1] << "," << sub << ") : " << site << std::endl;
        uint32_t site2;
        kagome.coor2site(coor, sub, site2);
        assert(site == site2);
    }
    std::cout << std::endl;
    std::cout << "kagome child" << std::endl;
    auto kagome_child = qbasis::divide_lattice(kagome);
    for (uint32_t site = 0; site < kagome_child.total_sites(); site++) {
        kagome_child.site2coor(coor, sub, site);
        std::cout << "(" << coor[0] << "," << coor[1] << "," << sub << ") : " << site << std::endl;
        uint32_t site2;
        kagome_child.coor2site(coor, sub, site2);
        assert(site == site2);
    }
    std::cout << std::endl;
    
    
    std::cout << "honeycomb" << std::endl;
    qbasis::lattice honeycomb("honeycomb",std::vector<uint32_t>{3,2},std::vector<std::string>{"pbc", "pbc"});
    for (uint32_t site = 0; site < honeycomb.total_sites(); site++) {
        honeycomb.site2coor(coor, sub, site);
        std::cout << "(" << coor[0] << "," << coor[1] << "," << sub << ") : " << site << std::endl;
        uint32_t site2;
        honeycomb.coor2site(coor, sub, site2);
        assert(site == site2);
    }
    std::cout << std::endl;
    std::cout << "honeycomb child" << std::endl;
    auto honeycomb_child = qbasis::divide_lattice(honeycomb);
    for (uint32_t site = 0; site < honeycomb_child.total_sites(); site++) {
        honeycomb_child.site2coor(coor, sub, site);
        std::cout << "(" << coor[0] << "," << coor[1] << "," << sub << ") : " << site << std::endl;
        uint32_t site2;
        honeycomb_child.coor2site(coor, sub, site2);
        assert(site == site2);
    }
    std::cout << std::endl;
    
}

void test_lattice2() {
    std::cout << "chain: " << std::endl;
    qbasis::lattice chain("chain", std::vector<uint32_t>{10},std::vector<std::string>{"pbc"});
    auto div = chain.divisor(std::vector<bool>{true});
    for (decltype(div.size()) j = 0; j < div.size(); j++) {
        std::cout << "div: " << div[j][0] << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "square: " << std::endl;
    qbasis::lattice square("square", std::vector<uint32_t>{10, 10},std::vector<std::string>{"pbc", "pbc"});
    auto div2 = square.divisor(std::vector<bool>{true,false});
    for (decltype(div2.size()) j = 0; j < div2.size(); j++) {
        std::cout << "div: " << div2[j][0] << ", " << div2[j][1] << std::endl;
    }
    
}

