#include <iostream>
#include <iomanip>
#include "qbasis.h"

// Heisenberg model on a chain
int main() {
    std::cout << std::setprecision(10);
    // parameters
    double J = 1.0;
    int L = 16;

    std::cout << "L =       " << L << std::endl;
    std::cout << "J =       " << J << std::endl << std::endl;

    // lattice object
    std::vector<std::string> bc{"pbc"};
    qbasis::lattice lattice("chain",std::vector<uint32_t>{static_cast<uint32_t>(L)},bc);

    // local matrix representation
    // Spins:
    std::vector<std::vector<std::complex<double>>> Splus(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sminus(2,std::vector<std::complex<double>>(2));
    std::vector<std::complex<double>> Sz(2);
    Splus[0][0]  = 0.0;
    Splus[0][1]  = 1.0;
    Splus[1][0]  = 0.0;
    Splus[1][1]  = 0.0;
    Sminus[0][0] = 0.0;
    Sminus[0][1] = 0.0;
    Sminus[1][0] = 1.0;
    Sminus[1][1] = 0.0;
    Sz[0]        = 0.5;
    Sz[1]        = -0.5;

    // constructing the Hamiltonian in operator representation
    qbasis::model<std::complex<double>> Heisenberg;
    Heisenberg.add_orbital(lattice.total_sites(), "spin-1/2");
    for (int x = 0; x < L; x++) {
        uint32_t site_i, site_j;
        lattice.coor2site(std::vector<int>{x}, 0, site_i); // obtain site label of (x)
        // construct operators on each site
        // spin
        auto Splus_i   = qbasis::opr<std::complex<double>>(site_i,0,false,Splus);
        auto Sminus_i  = qbasis::opr<std::complex<double>>(site_i,0,false,Sminus);
        auto Sz_i      = qbasis::opr<std::complex<double>>(site_i,0,false,Sz);

        // with neighbor (x+1)
        if (bc[0] == "pbc" || (bc[0] == "obc" && x < L - 1)) {
            lattice.coor2site(std::vector<int>{x+1}, 0, site_j);
            // spin exchanges
            auto Splus_j   = qbasis::opr<std::complex<double>>(site_j,0,false,Splus);
            auto Sminus_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Sminus);
            auto Sz_j      = qbasis::opr<std::complex<double>>(site_j,0,false,Sz);
            Heisenberg.add_offdiagonal_Ham(std::complex<double>(0.5 * J,0.0) * (Splus_i * Sminus_j + Sminus_i * Splus_j));
            Heisenberg.add_diagonal_Ham(std::complex<double>(J,0.0) * (Sz_i * Sz_j));
        }
    }


    // constructing the Hilbert space basis
    Heisenberg.enumerate_basis_full(Heisenberg.dim_target_full, Heisenberg.basis_target_full,
                                    {}, {});

    // optional, will use more memory and give higher speed
    // generating matrix of the Hamiltonian in the full Hilbert space
    Heisenberg.generate_Ham_sparse_full();
    std::cout << std::endl;


    // obtaining the eigenvals of the matrix
    Heisenberg.locate_E0_full(10,20);
    std::cout << std::endl;


    // for the parameters considered, we should obtain:
    assert(std::abs(Heisenberg.eigenvals_full[0] + 7.142296361) < 1e-8);
}
