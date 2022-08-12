#include <cassert>
#include <iostream>
#include <iomanip>
#include "qbasis.h"

// Heisenberg model on a chain
int main() {
    qbasis::initialize();
    std::cout << std::setprecision(10);
    // parameters
    double J = 1.0;
    int L = 12;
    double Sz_total_val = 0.0;

    std::cout << "L =        " << L << std::endl;
    std::cout << "J =        " << J << std::endl << std::endl;
    std::cout << "Sz_total = " << Sz_total_val << std::endl;

    // lattice object
    std::vector<std::string> bc{"pbc"};
    qbasis::lattice lattice("chain",{static_cast<uint32_t>(L)},bc);

    // local matrix representation
    // Spins:
    std::vector<std::vector<std::complex<double>>> Splus(3,std::vector<std::complex<double>>(3));
    std::vector<std::vector<std::complex<double>>> Sminus(3,std::vector<std::complex<double>>(3));
    std::vector<std::complex<double>> Sz(3);

    Splus[0][0] = 0.0;
    Splus[0][1] = std::sqrt(2.0);
    Splus[0][2] = 0.0;
    Splus[1][0] = 0.0;
    Splus[1][1] = 0.0;
    Splus[1][2] = std::sqrt(2.0);
    Splus[2][0] = 0.0;
    Splus[2][1] = 0.0;
    Splus[2][2] = 0.0;

    Sminus[0][0] = 0.0;
    Sminus[0][1] = 0.0;
    Sminus[0][2] = 0.0;
    Sminus[1][0] = std::sqrt(2.0);
    Sminus[1][1] = 0.0;
    Sminus[1][2] = 0.0;
    Sminus[2][0] = 0.0;
    Sminus[2][1] = std::sqrt(2.0);
    Sminus[2][2] = 0.0;

    Sz[0]     = 1.0;
    Sz[1]     = 0.0;
    Sz[2]     = -1.0;

    // constructing the Hamiltonian in operator representation
    qbasis::model<std::complex<double>> Heisenberg(lattice);
    Heisenberg.add_orbital(lattice.total_sites(), "spin-1");
    qbasis::mopr<std::complex<double>> Sz_total;
    for (int x = 0; x < L; x++) {
        uint32_t site_i, site_j;
        std::vector<int> work(lattice.dimension());
        lattice.coor2site({x}, 0, site_i, work); // obtain site label of (x)
        // construct operators on each site
        // spin
        auto Splus_i  = qbasis::opr<std::complex<double>>(site_i,0,false,Splus);
        auto Sminus_i = qbasis::opr<std::complex<double>>(site_i,0,false,Sminus);
        auto Sz_i     = qbasis::opr<std::complex<double>>(site_i,0,false,Sz);

        // with neighbor (x+1)
        if (bc[0] == "pbc" || (bc[0] == "obc" && x < L - 1)) {
            lattice.coor2site({x+1}, 0, site_j, work);
            // spin exchanges
            auto Splus_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Splus);
            auto Sminus_j = qbasis::opr<std::complex<double>>(site_j,0,false,Sminus);
            auto Sz_j     = qbasis::opr<std::complex<double>>(site_j,0,false,Sz);
            Heisenberg.add_Ham(std::complex<double>(0.5 * J,0.0) * (Splus_i * Sminus_j + Sminus_i * Splus_j));
            Heisenberg.add_Ham(std::complex<double>(J,0.0) * (Sz_i * Sz_j));
        }
        Sz_total += Sz_i;
    }

    // to use translational symmetry, we first fill the Weisse tables
    Heisenberg.fill_Weisse_table();

    std::vector<double> E0_list;
    for (int momentum = 0; momentum < L; momentum++) {
        // constructing the Hilbert space basis
        Heisenberg.enumerate_basis_repr({momentum}, {Sz_total}, {Sz_total_val});

        // generating matrix of the Hamiltonian in the subspace
        Heisenberg.generate_Ham_sparse_repr();
        std::cout << std::endl;

        // obtaining the eigenvals of the matrix
        Heisenberg.locate_E0_lanczos(qbasis::which_sym::repr, 2, 1);
        std::cout << std::endl;

        E0_list.push_back(Heisenberg.eigenvals_repr[0]);
    }
    assert(std::abs(E0_list[0]  + 16.86955614) < 1e-8);
    assert(std::abs(E0_list[1]  + 15.2458356)  < 1e-8);
    assert(std::abs(E0_list[2]  + 14.40827083) < 1e-8);
    assert(std::abs(E0_list[3]  + 14.13433756) < 1e-8);
    assert(std::abs(E0_list[4]  + 14.54973865) < 1e-8);

    for (int momentum = 0; momentum < L; momentum++) {
        std::cout << "E0(k=" << momentum << ")=\t" << E0_list[momentum] << std::endl;
    }

}
