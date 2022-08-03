#include <iostream>
#include <iomanip>
#include "qbasis.h"

// Heisenberg model on kagome lattice
int main() {
    qbasis::initialize();
    std::cout << std::setprecision(10);
    // parameters
    double J = 1.0;
    int Lx = 2;
    int Ly = 2;
    double Sz_total_val = 0;

    std::cout << "Lx =      " << Lx << std::endl;
    std::cout << "Ly =      " << Ly << std::endl;
    std::cout << "J =       " << J << std::endl;
    std::cout << "Sz =      " << Sz_total_val << std::endl << std::endl;


    // lattice object
    std::vector<std::string> bc{"pbc", "pbc"};
    assert(bc[0] == "pbc" && bc[1] == "pbc"); // "obc" requires more careful job in the following
    qbasis::lattice lattice("kagome",{static_cast<uint32_t>(Lx), static_cast<uint32_t>(Ly)},bc);


    // local matrix representation
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
    qbasis::model<std::complex<double>> Heisenberg(lattice);
    Heisenberg.add_orbital(lattice.total_sites(), "spin-1/2");
    qbasis::mopr<std::complex<double>> Sz_total;   // operators representating total Sz

    for (int m = 0; m < Lx; m++) {
        for (int n = 0; n < Ly; n++) {
            uint32_t site_i0, site_i1, site_i2;
            std::vector<int> work(lattice.dimension());
            lattice.coor2site({m,n}, 0, site_i0, work); // obtain site label of (x,y)
            lattice.coor2site({m,n}, 1, site_i1, work);
            lattice.coor2site({m,n}, 2, site_i2, work);
            // construct operators on each site
            auto Splus_i0   = qbasis::opr<std::complex<double>>(site_i0,0,false,Splus);
            auto Sminus_i0  = qbasis::opr<std::complex<double>>(site_i0,0,false,Sminus);
            auto Sz_i0      = qbasis::opr<std::complex<double>>(site_i0,0,false,Sz);

            auto Splus_i1   = qbasis::opr<std::complex<double>>(site_i1,0,false,Splus);
            auto Sminus_i1  = qbasis::opr<std::complex<double>>(site_i1,0,false,Sminus);
            auto Sz_i1      = qbasis::opr<std::complex<double>>(site_i1,0,false,Sz);

            auto Splus_i2   = qbasis::opr<std::complex<double>>(site_i2,0,false,Splus);
            auto Sminus_i2  = qbasis::opr<std::complex<double>>(site_i2,0,false,Sminus);
            auto Sz_i2      = qbasis::opr<std::complex<double>>(site_i2,0,false,Sz);

            //    1 -- 0 -> 1
            {
                Heisenberg.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i0 * Sminus_i1 + Sminus_i0 * Splus_i1));
                Heisenberg.add_Ham(std::complex<double>(J,0.0) * ( Sz_i0 * Sz_i1 ));
            }


            //    1 <- 0 -- 1
            {
                uint32_t site_j1;
                lattice.coor2site({m-1,n}, 1, site_j1, work);
                auto Splus_j1   = qbasis::opr<std::complex<double>>(site_j1,0,false,Splus);
                auto Sminus_j1  = qbasis::opr<std::complex<double>>(site_j1,0,false,Sminus);
                auto Sz_j1      = qbasis::opr<std::complex<double>>(site_j1,0,false,Sz);
                Heisenberg.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i0 * Sminus_j1 + Sminus_i0 * Splus_j1));
                Heisenberg.add_Ham(std::complex<double>(J,0.0) * ( Sz_i0 * Sz_j1 ));

            }


            /*   2
             *    ^
             *     \
             *      1
             *       \
             *        \
             *         2
             */
            {
                Heisenberg.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i1 * Sminus_i2 + Sminus_i1 * Splus_i2));
                Heisenberg.add_Ham(std::complex<double>(J,0.0) * ( Sz_i1 * Sz_i2 ));
            }


            /*   2
             *    \
             *     \
             *      1
             *       \
             *        v
             *         2
             */
            {
                uint32_t site_j2;
                lattice.coor2site({m+1,n-1}, 2, site_j2, work);
                auto Splus_j2   = qbasis::opr<std::complex<double>>(site_j2,0,false,Splus);
                auto Sminus_j2  = qbasis::opr<std::complex<double>>(site_j2,0,false,Sminus);
                auto Sz_j2      = qbasis::opr<std::complex<double>>(site_j2,0,false,Sz);
                Heisenberg.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i1 * Sminus_j2 + Sminus_i1 * Splus_j2));
                Heisenberg.add_Ham(std::complex<double>(J,0.0) * ( Sz_i1 * Sz_j2 ));
            }


            /*         0
             *        /
             *       /
             *      2
             *     /
             *    v
             *   0
             */
            {
                Heisenberg.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i2 * Sminus_i0 + Sminus_i2 * Splus_i0));
                Heisenberg.add_Ham(std::complex<double>(J,0.0) * ( Sz_i2 * Sz_i0 ));
            }


            /*         0
             *        ^
             *       /
             *      2
             *     /
             *    /
             *   0
             */
            {
                uint32_t site_j0;
                lattice.coor2site({m,n+1}, 0, site_j0, work);
                auto Splus_j0   = qbasis::opr<std::complex<double>>(site_j0,0,false,Splus);
                auto Sminus_j0  = qbasis::opr<std::complex<double>>(site_j0,0,false,Sminus);
                auto Sz_j0      = qbasis::opr<std::complex<double>>(site_j0,0,false,Sz);
                Heisenberg.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i2 * Sminus_j0 + Sminus_i2 * Splus_j0));
                Heisenberg.add_Ham(std::complex<double>(J,0.0) * ( Sz_i2 * Sz_j0 ));
            }

            // total Sz operator
            Sz_total += (Sz_i0 + Sz_i1 + Sz_i2);
        }
    }

    // constructing the Hilbert space basis
    Heisenberg.enumerate_basis_full({Sz_total}, {Sz_total_val});


    // optional, will use more memory and give higher speed
    // generating matrix of the Hamiltonian in the full Hilbert space
    Heisenberg.generate_Ham_sparse_full();
    std::cout << std::endl;


    // obtaining the eigenvals of the matrix
    Heisenberg.locate_E0_lanczos(0);
    std::cout << std::endl;


    // for the parameters considered, we should obtain:
    assert(std::abs(Heisenberg.energy_min() + 5.444875217) < 1e-8);

    return 0;
}
