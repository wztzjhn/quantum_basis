#include <iostream>
#include "basis.h"

int main(){
    basis alpha(10,2);
    alpha.test();
    std::cout << alpha.total_sites() << std::endl;
    std::cout << alpha.local_dimension() << std::endl;
}

