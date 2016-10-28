#include <iostream>
#include "basis.h"

int main(){
    basis alpha(10,284);
    alpha.test();
    
    std::cout << std::endl;
    basis beta(9,"spin-1/2");
    beta.test();
    
    std::cout << std::endl;
    basis gg(23,"spin-1");
    gg.test();
    
    mbasis gamma(6, {2,3,5});
    gamma.test();
    
}

