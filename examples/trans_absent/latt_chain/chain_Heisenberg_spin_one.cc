#include <iostream>
#include <iomanip>
#include "qbasis.h"

// Heisenberg model on a chain
int main() {
    qbasis::enable_ckpt = true;
    std::cout << std::setprecision(10);
    // parameters
    double J = 1.0;
    int L = 10;
    double Sz_total_val = 0.0;

    std::cout << "L =        " << L << std::endl;
    std::cout << "J =        " << J << std::endl << std::endl;
    std::cout << "Sz_total = " << Sz_total_val << std::endl;

    // lattice object
    std::vector<std::string> bc{"pbc"};
    qbasis::lattice lattice("chain",std::vector<uint32_t>{static_cast<uint32_t>(L)},bc);

    // local matrix representation
    // Spins:
    std::vector<std::vector<std::complex<double>>> Sx(3,std::vector<std::complex<double>>(3));
    std::vector<std::vector<std::complex<double>>> Sy(3,std::vector<std::complex<double>>(3));
    std::vector<std::complex<double>> Sz(3);

    Sx[0][0]  = 0.0;
    Sx[0][1]  = 1.0 / sqrt(2.0);
    Sx[0][2]  = 0.0;
    Sx[1][0]  = 1.0 / sqrt(2.0);
    Sx[1][1]  = 0.0;
    Sx[1][2]  = 1.0 / sqrt(2.0);
    Sx[2][0]  = 0.0;
    Sx[2][1]  = 1.0 / sqrt(2.0);
    Sx[2][2]  = 0.0;

    Sy[0][0]  = 0.0;
    Sy[0][1]  = std::complex<double>(0.0, -1.0 / sqrt(2.0));
    Sy[0][2]  = 0.0;
    Sy[1][0]  = std::complex<double>(0.0, 1.0 / sqrt(2.0));
    Sy[1][1]  = 0.0;
    Sy[1][2]  = std::complex<double>(0.0, -1.0 / sqrt(2.0));
    Sy[2][0]  = 0.0;
    Sy[2][1]  = std::complex<double>(0.0, 1.0 / sqrt(2.0));
    Sy[2][2]  = 0.0;

    Sz[0]     = 1.0;
    Sz[1]     = 0.0;
    Sz[2]     = -1.0;

    // constructing the Hamiltonian in operator representation
    qbasis::model<std::complex<double>> Heisenberg;
    Heisenberg.add_orbital(lattice.total_sites(), "spin-1");
    qbasis::mopr<std::complex<double>> Sz_total;
    for (int x = 0; x < L; x++) {
        uint32_t site_i, site_j;
        std::vector<int> work(lattice.dimension());
        lattice.coor2site(std::vector<int>{x}, 0, site_i, work); // obtain site label of (x)
        // construct operators on each site
        // spin
        auto Sx_i  = qbasis::opr<std::complex<double>>(site_i,0,false,Sx);
        auto Sy_i  = qbasis::opr<std::complex<double>>(site_i,0,false,Sy);
        auto Sz_i  = qbasis::opr<std::complex<double>>(site_i,0,false,Sz);

        // with neighbor (x+1)
        if (bc[0] == "pbc" || (bc[0] == "obc" && x < L - 1)) {
            lattice.coor2site(std::vector<int>{x+1}, 0, site_j, work);
            // spin exchanges
            auto Sx_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Sx);
            auto Sy_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Sy);
            auto Sz_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Sz);
            Heisenberg.add_offdiagonal_Ham(std::complex<double>(J,0.0) * (Sx_i * Sx_j + Sy_i * Sy_j));
            Heisenberg.add_diagonal_Ham(std::complex<double>(J,0.0) * (Sz_i * Sz_j));
        }
        Sz_total += Sz_i;
    }


    // constructing the Hilbert space basis
    Heisenberg.enumerate_basis_full({Sz_total}, {Sz_total_val});


    // optional, will use more memory and give higher speed
    // generating matrix of the Hamiltonian in the full Hilbert space
    //Heisenberg.generate_Ham_sparse_full();
    std::cout << std::endl;

    // obtaining the eigenvals of the matrix
    Heisenberg.locate_E0_lanczos(0,2,2);
    std::cout << std::endl;


    // for the parameters considered, we should obtain:
    assert(std::abs(Heisenberg.eigenvals_full[0] + 14.09412995) < 1e-8);
    assert(std::abs(Heisenberg.eigenvals_full[1] + 13.569322) < 1e-8);
}
