#include <iostream>
#include <iomanip>
#include "qbasis.h"

// spinless fermion model on honeycomb lattice
// benchmarked with Capponi et al., prb 92, 085146 (2015), Fig. 2
int main() {
    qbasis::initialize(true);
    std::cout << std::setprecision(10);
    // parameters
    double t = 1;
    double V1 = 4.0;
    int Lx = 3;
    int Ly = 2;
    double N_total = Lx * Ly - 2; // total number of fermions on lattice

    std::cout << "Lx =      " << Lx << std::endl;
    std::cout << "Ly =      " << Ly << std::endl;
    std::cout << "t =       " << t << std::endl;
    std::cout << "V1 =      " << V1 << std::endl;
    std::cout << "N =       " << N_total << std::endl;


    // lattice object
    std::vector<std::string> bc{"pbc", "pbc"};
    qbasis::lattice lattice("honeycomb",{static_cast<uint32_t>(Lx), static_cast<uint32_t>(Ly)},bc);


    // local matrix representation
    auto c = std::vector<std::vector<std::complex<double>>>(2,std::vector<std::complex<double>>(2, 0.0));
    c[0][1] = std::complex<double>(1.0,0.0);


    // initialize the Hamiltonian
    qbasis::model<std::complex<double>> spinless;
    spinless.add_orbital(lattice.total_sites(), "spinless-fermion");
    double constant = 0.0; // the constant energy correction from \sum_{\langle i,j \rangle} V1/4

    qbasis::mopr<std::complex<double>> Nfermion;   // operators representating total fermion number

    // constructing the Hamiltonian in operator representation
    // only enumerating sublattice 0, to avoid double-counting
    /*
                 1       1       1
               /   \   /   \   /
             0       0       0
             |       |       |
             1       1       1
    */
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            uint32_t site_i, site_j;
            std::vector<int> work(lattice.dimension());
            lattice.coor2site({x,y}, 0, site_i, work); // obtain site label of (x,y)
            // construct operators on each site
            auto c_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c);
            auto c_dg_i = c_i; c_dg_i.dagger();
            auto n_i    = c_dg_i * c_i;

            // with right neighbor (x, y), sublattice 1
            {
                lattice.coor2site({x,y}, 1, site_j, work);
                auto c_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c);
                auto c_dg_j = c_j; c_dg_j.dagger();
                auto n_j    = c_dg_j * c_j;
                spinless.add_Ham(std::complex<double>(-t,0.0) * ( c_dg_i * c_j ));
                spinless.add_Ham(std::complex<double>(-t,0.0) * ( c_dg_j * c_i ));
                spinless.add_Ham(std::complex<double>(V1,0.0) * (n_i * n_j));
                spinless.add_Ham(std::complex<double>(-0.5 * V1,0.0) * (n_i + n_j));
                constant += 0.25 * V1;
            }

            // with left neighbor (x-1, y), sublattice 1
            if ( bc[0] == "pbc" || (bc[0] == "obc" && x > 0) ) {
                lattice.coor2site({x-1,y}, 1, site_j, work);
                auto c_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c);
                auto c_dg_j = c_j; c_dg_j.dagger();
                auto n_j    = c_dg_j * c_j;
                spinless.add_Ham(std::complex<double>(-t,0.0) * ( c_dg_i * c_j ));
                spinless.add_Ham(std::complex<double>(-t,0.0) * ( c_dg_j * c_i ));
                spinless.add_Ham(std::complex<double>(V1,0.0) * (n_i * n_j));
                spinless.add_Ham(std::complex<double>(-0.5 * V1,0.0) * (n_i + n_j));
                constant += 0.25 * V1;
            }

             // with bottom neighbor (x, y-1), sublattice 1
            if (bc[1] == "pbc" || (bc[1] == "obc" && y > 0)) {
                lattice.coor2site({x,y-1}, 1, site_j, work);
                auto c_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c);
                auto c_dg_j = c_j; c_dg_j.dagger();
                auto n_j    = c_dg_j * c_j;
                spinless.add_Ham(std::complex<double>(-t,0.0) * ( c_dg_i * c_j ));
                spinless.add_Ham(std::complex<double>(-t,0.0) * ( c_dg_j * c_i ));
                spinless.add_Ham(std::complex<double>(V1,0.0) * (n_i * n_j));
                spinless.add_Ham(std::complex<double>(-0.5 * V1,0.0) * (n_i + n_j));
                constant += 0.25 * V1;
            }

            // total fermion operator
            lattice.coor2site({x,y}, 1, site_j, work);
            auto c_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c);
            auto c_dg_j = c_j; c_dg_j.dagger();
            auto n_j    = c_dg_j * c_j;
            Nfermion += (n_i + n_j);
        }
    }


    // to use translational symmetry, we first fill the Weisse tables
    spinless.fill_Weisse_table(lattice);

    std::vector<double> E0_list;
    for (int m = 0; m < Lx; m++) {
        for (int n = 0; n < Ly; n++) {
            // constructing the Hilbert space basis
            spinless.enumerate_basis_repr({m,n}, {Nfermion}, {N_total});

            // generating matrix of the Hamiltonian in the subspace
            //spinless.generate_Ham_sparse_repr();
            std::cout << std::endl;

            // obtaining the lowest eigenvals of the matrix
            spinless.locate_E0_lanczos(1);
            std::cout << std::endl;

            E0_list.push_back(spinless.eigenvals_repr[0]);

            spinless.locate_Emax_repr(4,10);
            std::cout << std::endl;
        }
    }

    // for the parameters considered, we should obtain:
    assert(std::abs(E0_list[0] + 28.60363167) < 1e-8);
    assert(std::abs(E0_list[1] + 28.27163215) < 1e-8);
    assert(std::abs(E0_list[2] + 28.60363167) < 1e-8);
    assert(std::abs(E0_list[3] + 28.27163215) < 1e-8);
    assert(std::abs(E0_list[4] + 28.60363167) < 1e-8);
    assert(std::abs(E0_list[5] + 28.27163215) < 1e-8);




    // -------------------------------------------------------------------------
    // the following is only for bench marking with older version of the code
    std::vector<double> E0_check_list;
    spinless.enumerate_basis_full({Nfermion}, {N_total});

    for (int i = 0; i < Lx; i++) {
        for (int j = 0; j < Ly; j++) {
            // constructing the subspace basis
            spinless.basis_init_repr_deprecated(lattice, {i,j});

            // generating matrix of the Hamiltonian in the subspace
            spinless.generate_Ham_sparse_repr_deprecated();
            std::cout << std::endl;

            // obtaining the lowest eigenvals of the matrix
            spinless.locate_E0_repr(3,10);
            std::cout << std::endl;
            E0_check_list.push_back(spinless.eigenvals_repr[0]);
        }
    }

    for (decltype(E0_list.size()) j = 0; j < E0_list.size(); j++) {
        std::cout << "E0(j=" << j << ")=\t" << E0_list[j]
        << "\tvs\t" << E0_check_list[j] << std::endl;
    }
}
