#include <iostream>
#include <vector>
#include "basis.h"
#include "operators.h"

int main(){
    basis_elem alpha(10,true,284);
    alpha.prt();
    std::cout << std::endl;
    
    basis_elem beta(9,"spin-1/2");
    beta.prt();
    std::cout << std::endl;
    
//    for (int i=0; i < 10000000; i++) {
//        DBitSet *p = new DBitSet;
//        delete p;
//    }
//    std::cout << std::endl;
//    DBitSet y(90);
//    std::vector<DBitSet> x(10000000,y);
////    for (decltype(x.size()) i=0; i<x.size(); i++) {
////        x[i].~dynamic_bitset();
////    }
//    std::cout << std::endl;
//    x.resize(0);
//    x.shrink_to_fit();
//    std::cout << std::endl;
    
//    double y = 10.0;
//    std::vector<double> x(100000000);
//    for (decltype(x.size()) i=0; i < x.size(); i++) {
//        x[i] = y;
//    }
//    std::cout << std::endl;
//    x.resize(1);
//    x.shrink_to_fit();
//    std::cout << std::endl;
//
//    double y1 = 10.0;
//    std::vector<double> x1(10000000);
//    for (decltype(x1.size()) i=0; i < x1.size(); i++) {
//        x1[i] = y1;
//    }
//    std::cout << std::endl;
//    x1.resize(1);
//    x1.shrink_to_fit();
//    std::cout << std::endl;
    
    
    std::vector<basis_elem> test(10000000);
    //test memory and default constructor
    test[0].prt();
    std::cout << std::endl;
    //test copy assignment constructor
    for (decltype(test.size()) i=0; i< test.size(); i++) {
        test[i] = alpha;
    }
    std::cout << std::endl;
    test.resize(0);
    test.shrink_to_fit();
    std::cout << test.size() << std::endl;
//
//    std::vector<basis_elem> test2(10000000);
//    for (decltype(test2.size()) i=0; i< test2.size(); i++) {
//        test2[i] = alpha;
//    }
//    test2.resize(1);
//    
//    std::vector<basis_elem> test3(10000000);
//    for (decltype(test3.size()) i=0; i< test3.size(); i++) {
//        test3[i] = alpha;
//    }
//    test3.resize(1);
//    
//    std::vector<basis_elem> test4(10000000);
//    for (decltype(test4.size()) i=0; i< test4.size(); i++) {
//        test4[i] = alpha;
//    }
//    test4.resize(1);
//    
//    std::vector<basis_elem> test5(10000000);
//    for (decltype(test5.size()) i=0; i< test5.size(); i++) {
//        test5[i] = alpha;
//    }
//    test5.resize(1);
//    
//    std::vector<basis_elem> test6(10000000);
//    for (decltype(test6.size()) i=0; i< test6.size(); i++) {
//        test6[i] = alpha;
//    }
//    test6.resize(1);
    
    
    
    basis_elem gg(23,"spin-1");
    gg.prt();
    std::cout << std::endl;
    
    mbasis_elem gamma(6, {"spin-1/2","spin-1"});
    gamma.test();
    
    opr<double> phi(3,2,false,std::vector<double>(3,0.2));
    phi.test();
    
    opr<double> phi2(6,1,false,std::vector<std::vector<double>>(3,std::vector<double>(3,0.4)));
    phi2.test();
    
}

