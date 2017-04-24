

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
