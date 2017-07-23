#include <iostream>
#include <iomanip>
#include "qbasis.h"

// Kondo Lattice model on a chain
int main() {
    std::cout << std::setprecision(10);
    std::cout << "The gap is bench marked with Tsunetsugu et al. PRB 46, 3175 (1992)." << std::endl;
    std::cout << "E0 not checked by any means yet (haven't setup ALPS correctly)." << std::endl;
    std::cout << "If you know a good reference or tool (or alps scripts), please contact me. Thanks!" << std::endl << std::endl;
    // parameters
    double t = 1;
    double J_Kondo = 1.1;
    double J_RKKY = 0.0;         // artificial RKKY
    int L = 4;
    double Sz_total_val = 0.0;   // not used. to turn on, modify a comment line near the bottom of the file
    double Nelec_total_val = L;

    std::cout << "L =       " << L << std::endl;
    std::cout << "t =       " << t << std::endl;
    std::cout << "J_Kondo = " << J_Kondo << std::endl << std::endl;
    std::cout << "J_RKKY  = " << J_RKKY << std::endl << std::endl;
    std::cout << "N_elec  = " << Nelec_total_val << std::endl;

    // lattice object
    std::vector<std::string> bc{"pbc"};
    qbasis::lattice lattice("chain",std::vector<uint32_t>{static_cast<uint32_t>(static_cast<uint32_t>(L))},bc);

    // local matrix representation
    // electrons:
    auto c_up = std::vector<std::vector<std::complex<double>>>(4,std::vector<std::complex<double>>(4, 0.0));
    auto c_dn = std::vector<std::vector<std::complex<double>>>(4,std::vector<std::complex<double>>(4, 0.0));
    c_up[0][1] = 1.0;
    c_up[2][3] = 1.0;
    c_dn[0][2] = 1.0;
    c_dn[1][3] = -1.0;
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
    // electrons on orbital 0, local spins on orbital 1
    qbasis::model<std::complex<double>> Kondo;
    Kondo.add_orbital(lattice.total_sites(), "electron");
    Kondo.add_orbital(lattice.total_sites(), "spin-1/2");
    qbasis::mopr<std::complex<double>> Nelec_total;   // operators representating total electron number
    qbasis::mopr<std::complex<double>> Sz_total;
    for (int x = 0; x < L; x++) {
        uint32_t site_i, site_j;
        lattice.coor2site(std::vector<int>{x}, 0, site_i); // obtain site label of (x)
        // construct operators on each site
        // electron
        auto c_up_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c_up);
        auto c_dn_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c_dn);
        auto c_up_dg_i = c_up_i; c_up_dg_i.dagger();
        auto c_dn_dg_i = c_dn_i; c_dn_dg_i.dagger();
        auto n_up_i    = c_up_dg_i * c_up_i;
        auto n_dn_i    = c_dn_dg_i * c_dn_i;
        auto splus_i   = c_up_dg_i * c_dn_i;
        auto sminus_i  = c_dn_dg_i * c_up_i;
        auto sz_i      = std::complex<double>(0.5,0.0) * (c_up_dg_i * c_up_i - c_dn_dg_i * c_dn_i);
        // spin
        auto Splus_i   = qbasis::opr<std::complex<double>>(site_i,1,false,Splus);
        auto Sminus_i  = qbasis::opr<std::complex<double>>(site_i,1,false,Sminus);
        auto Sz_i      = qbasis::opr<std::complex<double>>(site_i,1,false,Sz);

        // with neighbor (x+1)
        if (bc[0] == "pbc" || (bc[0] == "obc" && x < L - 1)) {
            lattice.coor2site(std::vector<int>{x+1}, 0, site_j);
            // hopping
            auto c_up_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
            auto c_dn_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
            auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
            auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
            Kondo.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i * c_up_j ));
            Kondo.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j * c_up_i ));
            Kondo.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i * c_dn_j ));
            Kondo.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j * c_dn_i ));
            // spin exchanges
            auto Splus_j   = qbasis::opr<std::complex<double>>(site_j,1,false,Splus);
            auto Sminus_j  = qbasis::opr<std::complex<double>>(site_j,1,false,Sminus);
            auto Sz_j      = qbasis::opr<std::complex<double>>(site_j,1,false,Sz);
            Kondo.add_offdiagonal_Ham(std::complex<double>(0.5 * J_RKKY,0.0) * (Splus_i * Sminus_j + Sminus_i * Splus_j));
            Kondo.add_diagonal_Ham(std::complex<double>(J_RKKY,0.0) * (Sz_i * Sz_j));
        }

        // electron-spin interaction
        Kondo.add_offdiagonal_Ham(std::complex<double>(0.5 * J_Kondo,0.0) * (Splus_i * sminus_i + Sminus_i * splus_i));
        Kondo.add_diagonal_Ham(std::complex<double>(J_Kondo,0.0) * (Sz_i * sz_i));

        // total electron operator
        Nelec_total += (n_up_i + n_dn_i);

        // total Sz operator
        Sz_total += (Sz_i + sz_i);
    }


    // constructing the Hilbert space basis
    //Kondo.enumerate_basis_full(lattice.total_sites(), {"electron","spin-1/2"}, {Nelec_total,Sz_total}, {static_cast<double>(Nelec_total_val),Sz_total_val});
    Kondo.enumerate_basis_full(lattice, {Nelec_total}, {Nelec_total_val});


    std::vector<double> energies;
    for (int i = 0; i < L; i++) {
        // constructing the subspace basis
        Kondo.basis_init_repr_deprecated(std::vector<int>{i}, lattice);

        // generating matrix of the Hamiltonian in the subspace
        Kondo.generate_Ham_sparse_repr();
        std::cout << std::endl;

        // obtaining the lowest eigenvals of the matrix
        Kondo.locate_E0_repr();
        std::cout << std::endl;
        energies.push_back(Kondo.eigenvals_repr[0]);
    }


    // for the parameters considered, we should obtain:
    assert(std::abs(energies[0] + 5.308767882) < 1e-8);
    assert(std::abs(energies[1] + 4.829404016) < 1e-8);
    assert(std::abs(energies[2] + 5.477455514) < 1e-8);
    assert(std::abs(energies[3] + 4.829404016) < 1e-8);


}
