#include <iostream>
#include <iomanip>
#include "qbasis.h"
#include <fstream>

// Fermi-Hubbard model on square lattice
int main() {
    std::cout << std::setprecision(10);
    // parameters
    double t = 1;
    double U = 1.1;
    int Lx = 3;
    int Ly = 3;
    double Nup_total = 5;
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
    qbasis::lattice lattice("square",std::vector<uint32_t>{static_cast<uint32_t>(Lx), static_cast<uint32_t>(Ly)},bc);


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
    qbasis::mopr<std::complex<double>> Nup;   // operators representating total electron number
    qbasis::mopr<std::complex<double>> Ndown;
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            uint32_t site_i, site_j;
            lattice.coor2site(std::vector<int>{x,y}, 0, site_i); // obtain site label of (x,y)
            // construct operators on each site
            auto c_up_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c_up);
            auto c_dn_i    = qbasis::opr<std::complex<double>>(site_i,0,true,c_dn);
            auto c_up_dg_i = c_up_i; c_up_dg_i.dagger();
            auto c_dn_dg_i = c_dn_i; c_dn_dg_i.dagger();
            auto n_up_i    = c_up_dg_i * c_up_i;
            auto n_dn_i    = c_dn_dg_i * c_dn_i;

            // hopping to neighbor (x+1, y)
            if (bc[0] == "pbc" || (bc[0] == "obc" && x < Lx - 1)) {
                lattice.coor2site(std::vector<int>{x+1,y}, 0, site_j);
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
                lattice.coor2site(std::vector<int>{x,y+1}, 0, site_j);
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


    // constructing the Hilbert space basis
    Hubbard.enumerate_basis_full(lattice, Hubbard.dim_target_full, Hubbard.basis_target_full,
                                 {Nup,Ndown}, {Nup_total,Ndn_total});


    // optional, will use more memory and give higher speed
    // generating matrix of the Hamiltonian in the full Hilbert space
    Hubbard.generate_Ham_sparse_full();
    std::cout << std::endl;


    // obtaining the eigenvals of the matrix
    Hubbard.locate_E0_full(2,10);
    std::cout << std::endl;


    // for the parameters considered, we should obtain:
    assert(std::abs(Hubbard.eigenvals_full[0] + 12.68398173) < 1e-8);


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
        q_list.push_back(std::vector<int>{m,n});
    }
    // along (pi,0) -> (pi,pi)
    for (int n = 0; n < Ly/2; n++) {
        int m = Lx/2;
        q_list.push_back(std::vector<int>{m,n});
    }
    // along (pi,pi) -> (0,0)
    for (int m = Lx/2; m >= 0; m--) {
        assert(Lx == Ly);
        int n = m;
        q_list.push_back(std::vector<int>{m,n});
    }
    // loop over all the q points
    for (auto q : q_list) {
        int m = q[0];
        int n = q[1];
        // construct Sz(-q)
        qbasis::mopr<std::complex<double>> Szmq;
        for (int x = 0; x < Lx; x++) {
            for (int y = 0; y < Ly; y++) {
                double qdotr = 2.0 * qbasis::pi * (m * x / static_cast<double>(Lx) + n * y / static_cast<double>(Ly));
                auto coeff = 0.5 / sqrt(static_cast<double>(Lx * Ly)) * std::exp(std::complex<double>{0.0,-qdotr});
                uint32_t site;
                lattice.coor2site(std::vector<int>{x,y}, 0, site);
                auto c_up_i    = qbasis::opr<std::complex<double>>(site,0,true,c_up);
                auto c_dn_i    = qbasis::opr<std::complex<double>>(site,0,true,c_dn);
                auto c_up_dg_i = c_up_i; c_up_dg_i.dagger();
                auto c_dn_dg_i = c_dn_i; c_dn_dg_i.dagger();
                auto n_up_i    = c_up_dg_i * c_up_i;
                auto n_dn_i    = c_dn_dg_i * c_dn_i;
                Szmq += coeff * (n_up_i - n_dn_i);
            }
        }
        // prepare restart state
        std::vector<std::complex<double>> phi0(Hubbard.dim_target_full, 0.0);
        Hubbard.moprXeigenvec_full(Szmq, phi0.data());
        // normalization of restart state
        double phi0_nrm2 = qbasis::nrm2(Hubbard.dim_target_full, phi0.data(), 1);
        double rnorm;
        std::cout << "Q\t" << m << "," << n << std::endl;
        std::cout << "phi_nrm2 = " << phi0_nrm2 << std::endl;
        if (phi0_nrm2 > qbasis::lanczos_precision) {
            qbasis::scal(Hubbard.dim_target_full, 1.0/phi0_nrm2, phi0.data(), 1);
        }
        fout << "Q\t" << m << "," << n << std::endl;
        fout << "nrm2\t" << phi0_nrm2 << std::endl;
        // run Lanczos once again
        std::cout << "Running continued fraction with " << step << " steps" << std::endl;
        std::vector<std::complex<double>> v(Hubbard.dim_target_full * 3);
        std::vector<double> hessenberg(step * 2, 0.0);
        if (phi0_nrm2 > qbasis::lanczos_precision) {
            std::cout << "Running continued fraction with " << step << " steps" << std::endl;
            std::vector<std::complex<double>> v(Hubbard.dim_target_full * 3);
            std::vector<double> hessenberg(step * 2, 0.0);
            if (Hubbard.HamMat_csr_target_full.dimension() == Hubbard.dim_target_full) {
                qbasis::lanczos(0, step, Hubbard.dim_target_full, Hubbard.HamMat_csr_target_full, rnorm, phi0.data(), v.data(), hessenberg.data(), step, false);
            } else {
                qbasis::lanczos(0, step, Hubbard.dim_target_full, Hubbard, rnorm, phi0.data(), v.data(), hessenberg.data(), step, false);
            }
            fout << "b\t";
            for (int i = 0; i < step; i++) {
                fout << std::setw(30) << hessenberg[i];
            }
            fout << std::endl;
            fout << "a\t";
            for (MKL_INT i = step; i < 2 * step; i++) {
                fout << std::setw(30) << hessenberg[i];
            }
            fout << std::endl;
        } else {
            fout << "b\t" << std::endl;
            fout << "a\t" << std::endl;
        }
        std::cout << std::endl << std::endl;
    }


}
