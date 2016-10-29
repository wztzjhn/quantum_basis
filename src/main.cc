#include <iostream>
#include "basis.h"
#include "operators.h"

int main(){
    basis alpha(10,true,284);
    alpha.test();
    
    std::cout << std::endl;
    basis beta(9,"spin-1/2");
    beta.test();
    
    std::cout << std::endl;
    basis gg(23,"spin-1");
    gg.test();
    
    mbasis gamma(6, {"spin-1/2","spin-1"});
    gamma.test();
    
    opr<double> phi(3,2,false,std::vector<double>(3,0.2));
    phi.test();
    
    opr<double> phi2(6,1,false,std::vector<std::vector<double>>(3,std::vector<double>(3,0.4)));
    phi2.test();
    
}

