#include <vector>
#include <iostream>
#include "basis.h"

void basis::test() const
{
    std::cout << "test: " << bits.size() << std::endl;
    std::cout << bits.empty() << std::endl;
}
