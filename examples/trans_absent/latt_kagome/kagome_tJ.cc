#include <iostream>
#include <iomanip>
#include "qbasis.h"

// tJ model on kagome lattice
int main() {
    qbasis::initialize(true);
    std::cout << std::setprecision(10);
    // parameters
    double t = 1.0;
    double J = 1.0;
    int Lx = 2;
    int Ly = 2;
    double N_total_val = Lx * Ly * 2;
    double Sz_total_val = 0;

    std::cout << "Lx =      " << Lx << std::endl;
    std::cout << "Ly =      " << Ly << std::endl;
    std::cout << "t =       " << t << std::endl;
    std::cout << "J =       " << J << std::endl;
    std::cout << "N =       " << N_total_val << std::endl;
    std::cout << "Sz =      " << Sz_total_val << std::endl << std::endl;


    // lattice object
    std::vector<std::string> bc{"pbc", "pbc"};
    assert(bc[0] == "pbc" && bc[1] == "pbc"); // "obc" requires more careful job in the following
    qbasis::lattice lattice("kagome",{static_cast<uint32_t>(Lx), static_cast<uint32_t>(Ly)},bc);


    // local matrix representation
    auto c_up = std::vector<std::vector<std::complex<double>>>(3,std::vector<std::complex<double>>(3, 0.0));
    auto c_dn = std::vector<std::vector<std::complex<double>>>(3,std::vector<std::complex<double>>(3, 0.0));
    c_up[0][1] = std::complex<double>(1.0,0.0);
    c_dn[0][2] = std::complex<double>(1.0,0.0);


    // constructing the Hamiltonian in operator representation
    qbasis::model<std::complex<double>> tJ;
    tJ.add_orbital(lattice.total_sites(), "tJ");
    qbasis::mopr<std::complex<double>> Sz_total;   // operators representating total Sz
    qbasis::mopr<std::complex<double>> N_total;    // operators representating total N

    for (int m = 0; m < Lx; m++) {
        for (int n = 0; n < Ly; n++) {
            uint32_t site_i0, site_i1, site_i2;
            std::vector<int> work(lattice.dimension());
            lattice.coor2site({m,n}, 0, site_i0, work); // obtain site label of (x,y)
            lattice.coor2site({m,n}, 1, site_i1, work);
            lattice.coor2site({m,n}, 2, site_i2, work);
            // construct operators on each site
            auto c_up_i0    = qbasis::opr<std::complex<double>>(site_i0,0,true,c_up);
            auto c_dn_i0    = qbasis::opr<std::complex<double>>(site_i0,0,true,c_dn);
            auto c_up_dg_i0 = c_up_i0; c_up_dg_i0.dagger();
            auto c_dn_dg_i0 = c_dn_i0; c_dn_dg_i0.dagger();
            auto Splus_i0   = c_up_dg_i0 * c_dn_i0;
            auto Sminus_i0  = c_dn_dg_i0 * c_up_i0;
            auto Sz_i0      = std::complex<double>(0.5,0.0) * (c_up_dg_i0 * c_up_i0 - c_dn_dg_i0 * c_dn_i0);
            auto N_i0       = (c_up_dg_i0 * c_up_i0 + c_dn_dg_i0 * c_dn_i0);

            auto c_up_i1    = qbasis::opr<std::complex<double>>(site_i1,0,true,c_up);
            auto c_dn_i1    = qbasis::opr<std::complex<double>>(site_i1,0,true,c_dn);
            auto c_up_dg_i1 = c_up_i1; c_up_dg_i1.dagger();
            auto c_dn_dg_i1 = c_dn_i1; c_dn_dg_i1.dagger();
            auto Splus_i1   = c_up_dg_i1 * c_dn_i1;
            auto Sminus_i1  = c_dn_dg_i1 * c_up_i1;
            auto Sz_i1      = std::complex<double>(0.5,0.0) * (c_up_dg_i1 * c_up_i1 - c_dn_dg_i1 * c_dn_i1);
            auto N_i1       = (c_up_dg_i1 * c_up_i1 + c_dn_dg_i1 * c_dn_i1);

            auto c_up_i2    = qbasis::opr<std::complex<double>>(site_i2,0,true,c_up);
            auto c_dn_i2    = qbasis::opr<std::complex<double>>(site_i2,0,true,c_dn);
            auto c_up_dg_i2 = c_up_i2; c_up_dg_i2.dagger();
            auto c_dn_dg_i2 = c_dn_i2; c_dn_dg_i2.dagger();
            auto Splus_i2   = c_up_dg_i2 * c_dn_i2;
            auto Sminus_i2  = c_dn_dg_i2 * c_up_i2;
            auto Sz_i2      = std::complex<double>(0.5,0.0) * (c_up_dg_i2 * c_up_i2 - c_dn_dg_i2 * c_dn_i2);
            auto N_i2       = (c_up_dg_i2 * c_up_i2 + c_dn_dg_i2 * c_dn_i2);

            //    1 -- 0 -> 1
            {
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i0 * c_up_i1 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i1 * c_up_i0 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i0 * c_dn_i1 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i1 * c_dn_i0 ));
                tJ.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i0 * Sminus_i1 + Sminus_i0 * Splus_i1));
                tJ.add_Ham(std::complex<double>(J,0.0) * ( Sz_i0 * Sz_i1 ));
                tJ.add_Ham(std::complex<double>(-0.25*J,0.0) * ( N_i0 * N_i1 ));
            }


            //    1 <- 0 -- 1
            {
                uint32_t site_j1;
                lattice.coor2site({m-1,n}, 1, site_j1, work);
                auto c_up_j1    = qbasis::opr<std::complex<double>>(site_j1,0,true,c_up);
                auto c_dn_j1    = qbasis::opr<std::complex<double>>(site_j1,0,true,c_dn);
                auto c_up_dg_j1 = c_up_j1; c_up_dg_j1.dagger();
                auto c_dn_dg_j1 = c_dn_j1; c_dn_dg_j1.dagger();
                auto Splus_j1   = c_up_dg_j1 * c_dn_j1;
                auto Sminus_j1  = c_dn_dg_j1 * c_up_j1;
                auto Sz_j1      = std::complex<double>(0.5,0.0) * (c_up_dg_j1 * c_up_j1 - c_dn_dg_j1 * c_dn_j1);
                auto N_j1       = (c_up_dg_j1 * c_up_j1 + c_dn_dg_j1 * c_dn_j1);
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i0 * c_up_j1 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j1 * c_up_i0 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i0 * c_dn_j1 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j1 * c_dn_i0 ));
                tJ.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i0 * Sminus_j1 + Sminus_i0 * Splus_j1));
                tJ.add_Ham(std::complex<double>(J,0.0) * ( Sz_i0 * Sz_j1 ));
                tJ.add_Ham(std::complex<double>(-0.25*J,0.0) * ( N_i0 * N_j1 ));
            }


            //   2
            //    ^
            //     \
            //      1
            //       \
            //        \
            //         2
            {
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i1 * c_up_i2 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i2 * c_up_i1 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i1 * c_dn_i2 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i2 * c_dn_i1 ));
                tJ.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i1 * Sminus_i2 + Sminus_i1 * Splus_i2));
                tJ.add_Ham(std::complex<double>(J,0.0) * ( Sz_i1 * Sz_i2 ));
                tJ.add_Ham(std::complex<double>(-0.25*J,0.0) * ( N_i1 * N_i2 ));
            }


            //   2
            //    \
            //     \
            //      1
            //       \
            //        v
            //         2
            {
                uint32_t site_j2;
                lattice.coor2site({m+1,n-1}, 2, site_j2, work);
                auto c_up_j2    = qbasis::opr<std::complex<double>>(site_j2,0,true,c_up);
                auto c_dn_j2    = qbasis::opr<std::complex<double>>(site_j2,0,true,c_dn);
                auto c_up_dg_j2 = c_up_j2; c_up_dg_j2.dagger();
                auto c_dn_dg_j2 = c_dn_j2; c_dn_dg_j2.dagger();
                auto Splus_j2   = c_up_dg_j2 * c_dn_j2;
                auto Sminus_j2  = c_dn_dg_j2 * c_up_j2;
                auto Sz_j2      = std::complex<double>(0.5,0.0) * (c_up_dg_j2 * c_up_j2 - c_dn_dg_j2 * c_dn_j2);
                auto N_j2       = (c_up_dg_j2 * c_up_j2 + c_dn_dg_j2 * c_dn_j2);
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i1 * c_up_j2 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j2 * c_up_i1 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i1 * c_dn_j2 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j2 * c_dn_i1 ));
                tJ.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i1 * Sminus_j2 + Sminus_i1 * Splus_j2));
                tJ.add_Ham(std::complex<double>(J,0.0) * ( Sz_i1 * Sz_j2 ));
                tJ.add_Ham(std::complex<double>(-0.25*J,0.0) * ( N_i1 * N_j2 ));
            }


            //         0
            //        /
            //       /
            //      2
            //     /
            //    v
            //   0
            {
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i2 * c_up_i0 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i0 * c_up_i2 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i2 * c_dn_i0 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i0 * c_dn_i2 ));
                tJ.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i2 * Sminus_i0 + Sminus_i2 * Splus_i0));
                tJ.add_Ham(std::complex<double>(J,0.0) * ( Sz_i2 * Sz_i0 ));
                tJ.add_Ham(std::complex<double>(-0.25*J,0.0) * ( N_i2 * N_i0 ));
            }


            //         0
            //        ^
            //       /
            //      2
            //     /
            //    /
            //   0
            {
                uint32_t site_j0;
                lattice.coor2site({m,n+1}, 0, site_j0, work);
                auto c_up_j0    = qbasis::opr<std::complex<double>>(site_j0,0,true,c_up);
                auto c_dn_j0    = qbasis::opr<std::complex<double>>(site_j0,0,true,c_dn);
                auto c_up_dg_j0 = c_up_j0; c_up_dg_j0.dagger();
                auto c_dn_dg_j0 = c_dn_j0; c_dn_dg_j0.dagger();
                auto Splus_j0   = c_up_dg_j0 * c_dn_j0;
                auto Sminus_j0  = c_dn_dg_j0 * c_up_j0;
                auto Sz_j0      = std::complex<double>(0.5,0.0) * (c_up_dg_j0 * c_up_j0 - c_dn_dg_j0 * c_dn_j0);
                auto N_j0       = (c_up_dg_j0 * c_up_j0 + c_dn_dg_j0 * c_dn_j0);
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i2 * c_up_j0 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j0 * c_up_i2 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i2 * c_dn_j0 ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j0 * c_dn_i2 ));
                tJ.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i2 * Sminus_j0 + Sminus_i2 * Splus_j0));
                tJ.add_Ham(std::complex<double>(J,0.0) * ( Sz_i2 * Sz_j0 ));
                tJ.add_Ham(std::complex<double>(-0.25*J,0.0) * ( N_i2 * N_j0 ));
            }

            // total Sz operator
            Sz_total += (Sz_i0 + Sz_i1 + Sz_i2);
            // total N operator
            N_total  += (N_i0 + N_i1 + N_i2);
        }
    }

    // constructing the Hilbert space basis
    tJ.enumerate_basis_full({Sz_total, N_total}, {Sz_total_val, N_total_val});


    // optional, will use more memory and give higher speed
    // generating matrix of the Hamiltonian in the full Hilbert space
    tJ.generate_Ham_sparse_full();
    std::cout << std::endl;


    // obtaining the eigenvals of the matrix
    tJ.locate_E0_lanczos(0);
    std::cout << std::endl;


    // for the parameters considered, we should obtain:
    assert(std::abs(tJ.eigenvals_full[0] + 15.41931496) < 1e-8);
}
