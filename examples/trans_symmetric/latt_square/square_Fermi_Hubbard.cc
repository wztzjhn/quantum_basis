#include <iostream>
#include <iomanip>
#include "qbasis.h"

// Fermi-Hubbard model on square lattice
int main() {
    std::cout << std::setprecision(10);
    // parameters
    double t = 1;
    double U = 1.1;
    int Lx = 4;
    int Ly = 2;
    double Nup_total = 4;
    double Ndn_total = 4;

    std::cout << "Lx =      " << Lx << std::endl;
    std::cout << "Ly =      " << Ly << std::endl;
    std::cout << "t =       " << t << std::endl;
    std::cout << "U =       " << U << std::endl;
    std::cout << "N_up =    " << Nup_total << std::endl;
    std::cout << "N_dn =    " << Ndn_total << std::endl << std::endl;

    // lattice object
    std::vector<std::string> bc{"pbc", "pbc"};
    qbasis::lattice lattice("square",{static_cast<uint32_t>(Lx), static_cast<uint32_t>(Ly)},bc);


    // local matrix representation
    auto c_up = std::vector<std::vector<std::complex<double>>>(4,std::vector<std::complex<double>>(4, 0.0));
    auto c_dn = std::vector<std::vector<std::complex<double>>>(4,std::vector<std::complex<double>>(4, 0.0));
    c_up[0][1] = std::complex<double>(1.0,0.0);
    c_up[2][3] = std::complex<double>(1.0,0.0);
    c_dn[0][2] = std::complex<double>(1.0,0.0);
    c_dn[1][3] = std::complex<double>(-1.0,0.0);


    // constructing the Hamiltonian in operator representation
    qbasis::model<std::complex<double>> Hubbard;
    Hubbard.add_orbital(lattice.total_sites(), "electron");
    qbasis::mopr<std::complex<double>> Nup;   // an operator representating total electron number
    qbasis::mopr<std::complex<double>> Ndown;
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            uint32_t site_i, site_j;
            lattice.coor2site({x,y}, 0, site_i); // obtain site label of (x,y)
            // construct operators on each site
            auto c_up_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c_up);
            auto c_dn_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c_dn);
            auto c_up_dg_i = c_up_i; c_up_dg_i.dagger();
            auto c_dn_dg_i = c_dn_i; c_dn_dg_i.dagger();
            auto n_up_i    = c_up_dg_i * c_up_i;
            auto n_dn_i    = c_dn_dg_i * c_dn_i;

            // hopping to neighbor (x+1, y)
            if (bc[0] == "pbc" || (bc[0] == "obc" && x < Lx - 1)) {
                lattice.coor2site({x+1,y}, 0, site_j);
                auto c_up_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
                auto c_dn_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
                auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
                auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
                Hubbard.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i * c_up_j ));
                Hubbard.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j * c_up_i ));
                Hubbard.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i * c_dn_j ));
                Hubbard.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j * c_dn_i ));
            }

            // hopping to neighbor (x, y+1)
            if (bc[1] == "pbc" || (bc[1] == "obc" && y < Ly - 1)) {
                lattice.coor2site({x,y+1}, 0, site_j);
                auto c_up_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
                auto c_dn_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
                auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
                auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
                Hubbard.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i * c_up_j ));
                Hubbard.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j * c_up_i ));
                Hubbard.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i * c_dn_j ));
                Hubbard.add_offdiagonal_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j * c_dn_i ));
            }

            // Hubbard repulsion, note that this operator is a sum (over sites) of diagonal matrices
            Hubbard.add_diagonal_Ham(std::complex<double>(U,0.0) * (n_up_i * n_dn_i));

            // total electron operator
            Nup   += n_up_i;
            Ndown += n_dn_i;
        }
    }


    // to use translational symmetry, we first fill the Weisse tables
    Hubbard.fill_Weisse_table(lattice);

    std::vector<double> E0_list;
    for (int m = 0; m < Lx; m++) {
        for (int n = 0; n < Ly; n++) {
            // constructing the Hilbert space basis
            Hubbard.enumerate_basis_repr(lattice, {m,n}, {Nup,Ndown}, {Nup_total,Ndn_total});

            // generating matrix of the Hamiltonian in the subspace
            Hubbard.generate_Ham_sparse_repr(lattice, {m,n});
            std::cout << std::endl;

            // obtaining the lowest eigenvals of the matrix
            Hubbard.locate_E0_repr(4,10);
            std::cout << std::endl;

            E0_list.push_back(Hubbard.eigenvals_repr[0]);
        }
    }


    // for the parameters considered, we should obtain:
    assert(std::abs(E0_list[0] + 14.07605866) < 1e-8);
    assert(std::abs(E0_list[1] + 10.50470669) < 1e-8);
    assert(std::abs(E0_list[2] + 12.16861094) < 1e-8);
    assert(std::abs(E0_list[3] + 12.19847764) < 1e-8);
    assert(std::abs(E0_list[4] + 10.54300366) < 1e-8);
    assert(std::abs(E0_list[5] + 14.03137587) < 1e-8);
    assert(std::abs(E0_list[6] + 12.16861094) < 1e-8);
    assert(std::abs(E0_list[7] + 12.19847764) < 1e-8);



    // -------------------------------------------------------------------------
    // the following is only for bench marking with older version of the code
    std::vector<double> E0_check_list;
    Hubbard.enumerate_basis_full({Nup,Ndown}, {Nup_total,Ndn_total});

    for (int i = 0; i < Lx; i++) {
        for (int j = 0; j < Ly; j++) {
            // constructing the subspace basis
            Hubbard.basis_init_repr_deprecated(lattice, {i,j});

            // generating matrix of the Hamiltonian in the subspace
            Hubbard.generate_Ham_sparse_repr_deprecated();
            std::cout << std::endl;

            // obtaining the lowest eigenvals of the matrix
            Hubbard.locate_E0_repr(2,10);
            std::cout << std::endl;
            E0_check_list.push_back(Hubbard.eigenvals_repr[0]);
        }
    }

    assert(E0_list.size() == E0_check_list.size());
    for (decltype(E0_list.size()) j = 0; j < E0_list.size(); j++) {
        assert(std::abs(E0_list[j]-E0_check_list[j]) < 1e-8);
        std::cout << "Energy comparison: " << E0_list[j] << "\tvs\t" << E0_check_list[j] << std::endl;
    }


}
