#include <iostream>
#include <iomanip>
#include "qbasis.h"

// tJ model on chain lattice
int main() {
    qbasis::initialize(true);
    std::cout << std::setprecision(10);
    // parameters
    double t = 1.0;
    double J = 1.0;
    int Lx = 12;
    double N_total_val = 8;
    double Sz_total_val = 0;

    std::cout << "Lx =      " << Lx << std::endl;
    std::cout << "t =       " << t << std::endl;
    std::cout << "J =       " << J << std::endl;
    std::cout << "N =       " << N_total_val << std::endl;
    std::cout << "Sz =      " << Sz_total_val << std::endl << std::endl;


    // lattice object
    std::vector<std::string> bc{"pbc"};
    qbasis::lattice lattice("chain",{static_cast<uint32_t>(Lx)},bc);


    // local matrix representation
    auto c_up = std::vector<std::vector<std::complex<double>>>(3,std::vector<std::complex<double>>(3, 0.0));
    auto c_dn = std::vector<std::vector<std::complex<double>>>(3,std::vector<std::complex<double>>(3, 0.0));
    c_up[0][1] = std::complex<double>(1.0,0.0);
    c_dn[0][2] = std::complex<double>(1.0,0.0);


    // constructing the Hamiltonian in operator representation
    qbasis::model<std::complex<double>> tJ(lattice);
    tJ.add_orbital(lattice.total_sites(), "tJ");
    qbasis::mopr<std::complex<double>> Sz_total;   // operators representating total Sz
    qbasis::mopr<std::complex<double>> N_total;    // operators representating total N

    for (int m = 0; m < Lx; m++) {
            uint32_t site_i, site_j;
            std::vector<int> work(lattice.dimension());
            lattice.coor2site({m},   0, site_i, work); // obtain site label of (x,y)
            lattice.coor2site({m+1}, 0, site_j, work);
            // construct operators on each site
            auto c_up_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c_up);
            auto c_dn_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c_dn);
            auto c_up_dg_i = c_up_i; c_up_dg_i.dagger();
            auto c_dn_dg_i = c_dn_i; c_dn_dg_i.dagger();
            auto Splus_i   = c_up_dg_i * c_dn_i;
            auto Sminus_i  = c_dn_dg_i * c_up_i;
            auto Sz_i      = std::complex<double>(0.5,0.0) * (c_up_dg_i * c_up_i - c_dn_dg_i * c_dn_i);
            auto N_i       = (c_up_dg_i * c_up_i + c_dn_dg_i * c_dn_i);

            auto c_up_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
            auto c_dn_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
            auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
            auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
            auto Splus_j   = c_up_dg_j * c_dn_j;
            auto Sminus_j  = c_dn_dg_j * c_up_j;
            auto Sz_j      = std::complex<double>(0.5,0.0) * (c_up_dg_j * c_up_j - c_dn_dg_j * c_dn_j);
            auto N_j       = (c_up_dg_j * c_up_j + c_dn_dg_j * c_dn_j);

            if (bc[0] == "pbc" || (bc[0] == "obc" && m < Lx - 1)) {
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i * c_up_j ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j * c_up_i ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i * c_dn_j ));
                tJ.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j * c_dn_i ));
                tJ.add_Ham(std::complex<double>(0.5*J,0.0) * (Splus_i * Sminus_j + Sminus_i * Splus_j));
                tJ.add_Ham(std::complex<double>(J,0.0) * ( Sz_i * Sz_j ));
                tJ.add_Ham(std::complex<double>(-0.25*J,0.0) * ( N_i * N_j ));
            }


            // total Sz operator
            Sz_total += Sz_i;
            // total N operator
            N_total  += N_i;

    }

    // constructing the Hilbert space basis
    tJ.enumerate_basis_full({Sz_total, N_total}, {Sz_total_val, N_total_val});


    // optional, will use more memory and give higher speed
    // generating matrix of the Hamiltonian in the full Hilbert space
    tJ.generate_Ham_sparse_full();
    std::cout << std::endl;


    // obtaining the eigenvals of the matrix
    tJ.locate_E0_iram(0,4,8);
    std::cout << std::endl;


    // for the parameters considered, we should obtain:
    assert(std::abs(tJ.eigenvals_full[0] + 9.762087307) < 1e-8);
    assert(std::abs(tJ.eigenvals_full[1] + 9.762087307) < 1e-8);
}
