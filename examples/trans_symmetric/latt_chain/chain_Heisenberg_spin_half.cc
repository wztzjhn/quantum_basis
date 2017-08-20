#include <iostream>
#include <iomanip>
#include "qbasis.h"

// Heisenberg model on a chain
int main() {
    std::cout << std::setprecision(10);
    // parameters
    double J = 1.0;
    int L = 16;
    double Sz_total_val = 0.0;

    std::cout << "L =       " << L << std::endl;
    std::cout << "J =       " << J << std::endl << std::endl;

    // lattice object
    std::vector<std::string> bc{"pbc"};
    qbasis::lattice lattice("chain",{static_cast<uint32_t>(L)},bc);

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
    qbasis::mopr<std::complex<double>> Sz_total;
    Heisenberg.add_orbital(lattice.total_sites(), "spin-1/2");
    for (int x = 0; x < L; x++) {
        uint32_t site_i, site_j;
        lattice.coor2site({x}, 0, site_i); // obtain site label of (x)
        // construct operators on each site
        // spin
        auto Splus_i   = qbasis::opr<std::complex<double>>(site_i,0,false,Splus);
        auto Sminus_i  = qbasis::opr<std::complex<double>>(site_i,0,false,Sminus);
        auto Sz_i      = qbasis::opr<std::complex<double>>(site_i,0,false,Sz);

        // with neighbor (x+1)
        if (bc[0] == "pbc" || (bc[0] == "obc" && x < L - 1)) {
            lattice.coor2site({x+1}, 0, site_j);
            // spin exchanges
            auto Splus_j   = qbasis::opr<std::complex<double>>(site_j,0,false,Splus);
            auto Sminus_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Sminus);
            auto Sz_j      = qbasis::opr<std::complex<double>>(site_j,0,false,Sz);
            Heisenberg.add_offdiagonal_Ham(std::complex<double>(0.5 * J,0.0) * (Splus_i * Sminus_j + Sminus_i * Splus_j));
            Heisenberg.add_diagonal_Ham(std::complex<double>(J,0.0) * (Sz_i * Sz_j));
        }

        Sz_total += Sz_i;
    }


    // to use translational symmetry, we first fill the Weisse tables
    Heisenberg.fill_Weisse_table(lattice);

    std::vector<double> E0_list;
    for (int momentum = 0; momentum < L; momentum++) {
        // generate the translational symmetric basis
        Heisenberg.enumerate_basis_repr({momentum}, {Sz_total}, {Sz_total_val});

        // optional in future, will use more memory and give higher speed
        // generating matrix of the Hamiltonian in the full Hilbert space
        //Heisenberg.generate_Ham_sparse_repr();
        //std::cout << std::endl;

        // obtaining the eigenvals of the matrix
        Heisenberg.locate_E0_repr(10,20);
        std::cout << std::endl;

        E0_list.push_back(Heisenberg.eigenvals_repr[0]);
    }
    assert(std::abs(E0_list[0]  + 7.142296361) < 1e-8);
    assert(std::abs(E0_list[1]  + 6.523407057) < 1e-8);
    assert(std::abs(E0_list[2]  + 5.990986863) < 1e-8);
    assert(std::abs(E0_list[3]  + 5.615175598) < 1e-8);
    assert(std::abs(E0_list[4]  + 5.451965668) < 1e-8);
    assert(std::abs(E0_list[5]  + 5.525353087) < 1e-8);
    assert(std::abs(E0_list[6]  + 5.823231143) < 1e-8);
    assert(std::abs(E0_list[7]  + 6.298652725) < 1e-8);
    assert(std::abs(E0_list[8]  + 6.872106678) < 1e-8);
    assert(std::abs(E0_list[9]  + 6.298652725) < 1e-8);
    assert(std::abs(E0_list[10] + 5.823231143) < 1e-8);
    assert(std::abs(E0_list[11] + 5.525353087) < 1e-8);
    assert(std::abs(E0_list[12] + 5.451965668) < 1e-8);
    assert(std::abs(E0_list[13] + 5.615175598) < 1e-8);
    assert(std::abs(E0_list[14] + 5.990986863) < 1e-8);
    assert(std::abs(E0_list[15] + 6.523407057) < 1e-8);

    // -------------------------------------------------------------------------
    // the following is only for bench marking with older version of the code
    std::vector<double> E0_check_list;
    Heisenberg.enumerate_basis_full({Sz_total}, {Sz_total_val});
    for (int momentum = 0; momentum < L; momentum++) {
        // generate the translational symmetric basis
        Heisenberg.basis_init_repr_deprecated(lattice, {momentum});

        // optional in future, will use more memory and give higher speed
        // generating matrix of the Hamiltonian in the full Hilbert space
        Heisenberg.generate_Ham_sparse_repr_deprecated();
        std::cout << std::endl;

        // obtaining the eigenvals of the matrix
        Heisenberg.locate_E0_repr(10,20);
        std::cout << std::endl;

        E0_check_list.push_back(Heisenberg.eigenvals_repr[0]);
    }


    for (int momentum = 0; momentum < L; momentum++) {
        std::cout << "E0(k=" << momentum << ")=\t" << E0_list[momentum]
        << "\tvs\t" << E0_check_list[momentum] << std::endl;
    }

}
