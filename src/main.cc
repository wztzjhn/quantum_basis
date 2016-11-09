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
    phi.test();
    
    opr<double> phi2(6,1,false,std::vector<std::vector<double>>(3,std::vector<double>(3,0.4)));
    phi2.test();
    
}

