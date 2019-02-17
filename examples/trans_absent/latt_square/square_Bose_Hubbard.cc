#include <iostream>
#include <iomanip>
#include "qbasis.h"

// Bose-Hubbard model on square lattice
int main() {
    qbasis::initialize();
    std::cout << std::setprecision(10);
    // parameters
    double t = 1;
    double U = 1.1;
    int Lx = 3;
    int Ly = 3;
    double N_total = 9; // total number of bosons on lattice
    uint8_t Nmax = 2;   // maximal number of bosons on each site, set this # to N_total yields the exact result

    qbasis::extra_info boson_limit; // (optional in other models)
    boson_limit.Nmax = Nmax;

    std::cout << "Lx =      " << Lx << std::endl;
    std::cout << "Ly =      " << Ly << std::endl;
    std::cout << "t =       " << t << std::endl;
    std::cout << "U =       " << U << std::endl;
    std::cout << "Nmax =    " << static_cast<unsigned>(Nmax) << std::endl;
    std::cout << "N =       " << N_total << std::endl;


    // lattice object
    std::vector<std::string> bc{"pbc", "pbc"};
    qbasis::lattice lattice("square",{static_cast<uint32_t>(Lx), static_cast<uint32_t>(Ly)},bc);


    // local matrix representation
    auto b = std::vector<std::vector<std::complex<double>>>(Nmax+1,std::vector<std::complex<double>>(Nmax+1, 0.0));
    for (uint8_t d = 0; d < Nmax; d++)
        b[d][d+1] = std::complex<double>(sqrt(static_cast<double>(d+1)),0.0);


    // initialize the Hamiltonian
    qbasis::model<std::complex<double>> Hubbard(lattice);
    Hubbard.add_orbital(lattice.total_sites(), "boson", boson_limit);

    qbasis::mopr<std::complex<double>> Nboson;   // operators representating total boson number

    // constructing the Hamiltonian in operator representation
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            uint32_t site_i, site_j;
            std::vector<int> work(lattice.dimension());
            lattice.coor2site({x,y}, 0, site_i, work); // obtain site label of (x,y)
            // construct operators on each site
            auto b_i    = qbasis::opr<std::complex<double>>(site_i,0,false,b);
            auto b_dg_i = b_i; b_dg_i.dagger();
            auto n_i    = b_dg_i * b_i;

            // hopping to neighbor (x+1, y)
            if (bc[0] == "pbc" || (bc[0] == "obc" && x < Lx - 1)) {
                lattice.coor2site({x+1,y}, 0, site_j, work);
                auto b_j    = qbasis::opr<std::complex<double>>(site_j,0,false,b);
                auto b_dg_j = b_j; b_dg_j.dagger();
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( b_dg_i * b_j ));
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( b_dg_j * b_i ));
            }

            // hopping to neighbor (x, y+1)
            if (bc[1] == "pbc" || (bc[1] == "obc" && y < Ly - 1)) {
                lattice.coor2site({x,y+1}, 0, site_j, work);
                auto b_j    = qbasis::opr<std::complex<double>>(site_j,0,false,b);
                auto b_dg_j = b_j; b_dg_j.dagger();
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( b_dg_i * b_j ));
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( b_dg_j * b_i ));
            }

            // Hubbard repulsion, note that this operator is a sum (over sites) of diagonal matrices
            Hubbard.add_Ham(std::complex<double>(0.5*U,0.0) * (n_i * n_i - n_i));

            // total boson operator
            Nboson += n_i;
        }
    }


    // constructing the Hilbert space basis
    Hubbard.enumerate_basis_full({Nboson}, {N_total});


    // optional, will use more memory and give higher speed
    // generating matrix of the Hamiltonian in the full Hilbert space
    //Hubbard.generate_Ham_sparse_full();
    //std::cout << std::endl;


    // obtaining the eigenvals of the matrix
    Hubbard.locate_E0_lanczos(0);
    std::cout << std::endl;


    // for the parameters considered, we should obtain:
    assert(std::abs(Hubbard.eigenvals_full[0] + 25.81136094) < 1e-8);
}
