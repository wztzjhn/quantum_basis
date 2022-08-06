#include <iostream>
#include <iomanip>
#include "qbasis.h"
#include <fstream>

#define PI 3.1415926535897932

// Fermi-Hubbard model on square lattice
int main() {
    qbasis::initialize();
    std::cout << std::setprecision(10);
    // parameters
    double t = 1;
    double U = 1.1;
    int Lx = 4;
    int Ly = 2;
    double Nup_total = 4;
    double Ndn_total = 4;
    int step = 150;

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
    qbasis::model<std::complex<double>> Hubbard(lattice);
    Hubbard.add_orbital(lattice.total_sites(), "electron");
    qbasis::mopr<std::complex<double>> Nup;   // operators representating total electron number
    qbasis::mopr<std::complex<double>> Ndown;
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            uint32_t site_i, site_j;
            std::vector<int> work(lattice.dimension());
            lattice.coor2site({x,y}, 0, site_i, work); // obtain site label of (x,y)
            // construct operators on each site
            auto c_up_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c_up);
            auto c_dn_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c_dn);
            auto c_up_dg_i = c_up_i; c_up_dg_i.dagger();
            auto c_dn_dg_i = c_dn_i; c_dn_dg_i.dagger();
            auto n_up_i    = c_up_dg_i * c_up_i;
            auto n_dn_i    = c_dn_dg_i * c_dn_i;

            // hopping to neighbor (x+1, y)
            if (bc[0] == "pbc" || (bc[0] == "obc" && x < Lx - 1)) {
                lattice.coor2site({x+1,y}, 0, site_j, work);
                auto c_up_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
                auto c_dn_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
                auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
                auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i * c_up_j ));
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j * c_up_i ));
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i * c_dn_j ));
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j * c_dn_i ));
            }

            // hopping to neighbor (x, y+1)
            if (bc[1] == "pbc" || (bc[1] == "obc" && y < Ly - 1)) {
                lattice.coor2site({x,y+1}, 0, site_j, work);
                auto c_up_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
                auto c_dn_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
                auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
                auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i * c_up_j ));
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j * c_up_i ));
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i * c_dn_j ));
                Hubbard.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j * c_dn_i ));
            }

            // Hubbard repulsion, note that this operator is a sum (over sites) of diagonal matrices
            Hubbard.add_Ham(std::complex<double>(U,0.0) * (n_up_i * n_dn_i));

            // total electron operator
            Nup   += n_up_i;
            Ndown += n_dn_i;
        }
    }


    // constructing the Hilbert space basis
    Hubbard.enumerate_basis_full({Nup,Ndown}, {Nup_total,Ndn_total});


    // optional, will use more memory and give higher speed
    // generating matrix of the Hamiltonian in the full Hilbert space
    Hubbard.generate_Ham_sparse_full();
    std::cout << std::endl;


    // obtaining the eigenvals of the matrix
    Hubbard.locate_E0_lanczos(0);
    std::cout << std::endl;


    // for the parameters considered, we should obtain:
    assert(std::abs(Hubbard.eigenvals_full[0] + 14.07605866) < 1e-8);


    // measure operators
    qbasis::mopr<std::complex<double>> cupdg1cup5(qbasis::opr<std::complex<double>>(1,0,true,c_up).dagger() *
                      qbasis::opr<std::complex<double>>(5,0,true,c_up));

    auto m1 = Hubbard.measure_full_static(cupdg1cup5, 0, 0);
    std::cout << "cupdg1cup5 = " << m1 << std::endl << std::endl;
    assert(std::abs(m1 - 0.3957690742) < 1e-8);


    // calculate dynamical structure factor Sz(q)*Sz(-q)
    std::string output_name = "L"+std::to_string(Lx)+"x"+std::to_string(Ly)+ "_chi.dat";
    std::ofstream fout(output_name, std::ios::out);
    fout << std::scientific << std::right;
    fout << std::setprecision(18);
    fout << "Lx\t" << Lx << std::endl;
    fout << "Ly\t" << Ly << std::endl;
    fout << "U\t" << U << std::endl;
    fout << "E0\t" << Hubbard.energy_min() << std::endl;
    fout << "Gap\t" << Hubbard.energy_gap() << std::endl << std::endl;


    // prepare list of momentum points for measurement
    std::vector<std::vector<int>> q_list;
    // along (0,0) -> (pi,0)
    for (int m = 0; m < Lx/2; m++) {
        int n = 0;
        q_list.push_back({m,n});
    }
    // along (pi,0) -> (pi,pi)
    for (int n = 0; n <= Ly/2; n++) {
        int m = Lx/2;
        q_list.push_back({m,n});
    }
    // loop over all the q points
    for (auto q : q_list) {
        int m = q[0];
        int n = q[1];
        std::cout << "Q\t" << m << "," << n << std::endl;

        // construct Sz(q)
        qbasis::mopr<std::complex<double>> Szq;
        for (int x = 0; x < Lx; x++) {
            for (int y = 0; y < Ly; y++) {
                double qdotr = 2.0 * PI * (m * x / static_cast<double>(Lx) + n * y / static_cast<double>(Ly));
                auto coeff = 0.5 / sqrt(static_cast<double>(Lx * Ly)) * std::exp(std::complex<double>{0.0,qdotr});
                uint32_t site;
                std::vector<int> work(lattice.dimension());
                lattice.coor2site({x,y}, 0, site, work);
                auto c_up_i    = qbasis::opr<std::complex<double>>(site,0,true,c_up);
                auto c_dn_i    = qbasis::opr<std::complex<double>>(site,0,true,c_dn);
                auto c_up_dg_i = c_up_i; c_up_dg_i.dagger();
                auto c_dn_dg_i = c_dn_i; c_dn_dg_i.dagger();
                auto n_up_i    = c_up_dg_i * c_up_i;
                auto n_dn_i    = c_dn_dg_i * c_dn_i;
                Szq += coeff * (n_up_i - n_dn_i);
            }
        }


        // calculate dynamical correlation functions
        MKL_INT step_used;
        double norm;
        std::vector<double> hessenberg(2*step);
        Hubbard.measure_full_dynamic(Szq, 0, 0, step, step_used, norm, hessenberg.data());
        std::cout << "norm = " << norm << std::endl;
        fout << "Q\t" << m << "," << n << std::endl;
        fout << "nrm2\t" << norm << std::endl;
        if (norm > qbasis::lanczos_precision) {
            fout << "b\t";
            for (MKL_INT i = 0; i < step_used; i++) {
                fout << std::setw(30) << hessenberg[i];
            }
            fout << std::endl;
            fout << "a\t";
            for (MKL_INT i = step; i < step + step_used; i++) {
                fout << std::setw(30) << hessenberg[i];
            }
            fout << std::endl;
        } else {
            fout << "b\t" << std::endl;
            fout << "a\t" << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
    fout.close();

    return 0;
}
