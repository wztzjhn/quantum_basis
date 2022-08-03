#include <iostream>
#include <iomanip>
#include "qbasis.h"

int test_chain_Heisenberg_spin_half();

int main() {
    test_chain_Heisenberg_spin_half();

    return 0;
}

int test_chain_Heisenberg_spin_half() {
    qbasis::initialize();
    std::cout << std::setprecision(10);
    // parameters
    double J = 1.0;
    int L = 16;

    std::cout << "L =       " << L << std::endl;
    std::cout << "J =       " << J << std::endl << std::endl;

}