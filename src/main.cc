#include <iostream>
#include <vector>
#include "basis.h"
#include "operators.h"

int main(){
    basis_elem alpha(10,true,284);
    alpha.prt();
    std::cout << std::endl;
    
    
    mbasis_elem gamma(6, {"spin-1/2","spin-1", "spin-1"});
    gamma.test();
    mbasis_elem delta(gamma);
    delta.test();
    
    mbasis_elem kappa;
    kappa = std::move(gamma);
    kappa.test();
    
    
    opr<double> phi(3,2,false,std::vector<double>(3,0.2));
    phi.prt();
    auto temp = std::vector<std::vector<double>>(2,std::vector<double>(3,0.4));
    temp[0][1] = 0.5;
    temp[0][2] = 0.6;
    temp[1][0] = 0.7;
    temp[1][1] = 0.8;
    temp[1][2] = 0.9;
    
    opr<double> phi2(6,1,false,temp);
    phi2.prt();
    
    std::cout << std::endl;
    //phi2.prt();
    
    std::cout << "swapping" << std::endl;
    swap(phi, phi2);
    phi.prt(); std::cout << std::endl;
    phi2.prt(); std::cout << std::endl;
    
    std::cout << "testing copy constructor" << std::endl;
    auto phi3(phi);
    phi3.prt();
    
    std::cout << std::endl;
    std::cout << "testing move constructor" << std::endl;
    auto phi4(std::move(phi));
    phi4.prt();
    std::cout << "in phi:" << std::endl;
    phi.prt();
    
    std::cout << std::endl;
    std::cout << "testing copy assignment" << std::endl;
    phi = phi3;
    phi.prt();
    
    
    std::cout << std::endl;
    std::cout << "testing move assignment" << std::endl;
    phi = std::move(phi3);
    phi.prt();
    phi3.prt();
}

