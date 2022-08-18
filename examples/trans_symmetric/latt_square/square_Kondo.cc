#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "qbasis.h"

#define PI 3.1415926535897932

// Kondo Lattice model on square lattice
int main() {
    qbasis::initialize();
    std::cout << std::setprecision(10);
    // parameters
    int Lx, Ly;
    double t = 1;
    double J_Kondo;
    double H;
    double Nelec_total_val;
//    double Sz_total_val;  // also a good quantum number, and can be turned on

    std::cout << "Input Lx: " << std::endl;
    std::cin >> Lx;
    std::cout << Lx << std::endl << std::endl;
    std::cout << "Input Ly: " << std::endl;
    std::cin >> Ly;
    std::cout << Ly << std::endl << std::endl;

    std::cout << "Input J_Kondo: " << std::endl;
    std::cin >> J_Kondo;
    std::cout << J_Kondo << std::endl << std::endl;

    std::cout << "Input H: " << std::endl;
    std::cin >> H;
    std::cout << H << std::endl << std::endl;

    std::cout << "Nelec_total_val: " << std::endl;
    std::cin >> Nelec_total_val;
    std::cout << Nelec_total_val << std::endl << std::endl;

//    std::cout << "Input Sz_total_val: " << std::endl;
//    std::cin >> Sz_total_val;
//    std::cout << Sz_total_val << std::endl << std::endl;

    // lattice object
    std::vector<std::string> bc{"pbc", "pbc"};
    qbasis::lattice lattice("square",{static_cast<uint32_t>(Lx), static_cast<uint32_t>(Ly)},bc);


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
    qbasis::model<std::complex<double>> Kondo(lattice);
    Kondo.add_orbital(lattice.Nsites, "electron");
    Kondo.add_orbital(lattice.Nsites, "spin-1/2");
    qbasis::mopr<std::complex<double>> Nelec_total;   // operators representating total electron number
    qbasis::mopr<std::complex<double>> Sz_total;
    qbasis::mopr<std::complex<double>> Nelec_up, Nelec_dn, Mz_total, mz_total;
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            uint32_t site_i, site_j;
            std::vector<int> work(lattice.dim);
            lattice.coor2site({x,y}, 0, site_i, work); // obtain site label of (x,y)
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

            // with neighbor (x+1, y)
            if (bc[0] == "pbc" || (bc[0] == "obc" && x < Lx - 1)) {
                lattice.coor2site({x+1,y}, 0, site_j, work);
                auto c_up_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
                auto c_dn_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
                auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
                auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
                Kondo.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i * c_up_j ));
                Kondo.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j * c_up_i ));
                Kondo.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i * c_dn_j ));
                Kondo.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j * c_dn_i ));
            }

            // with neighbor (x, y+1)
            if (bc[1] == "pbc" || (bc[1] == "obc" && y < Ly - 1)) {
                lattice.coor2site({x,y+1}, 0, site_j, work);
                auto c_up_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
                auto c_dn_j    = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
                auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
                auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
                Kondo.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_i * c_up_j ));
                Kondo.add_Ham(std::complex<double>(-t,0.0) * ( c_up_dg_j * c_up_i ));
                Kondo.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_i * c_dn_j ));
                Kondo.add_Ham(std::complex<double>(-t,0.0) * ( c_dn_dg_j * c_dn_i ));
            }

            // electron-spin interaction
            Kondo.add_Ham(std::complex<double>(0.5 * J_Kondo,0.0) * (Splus_i * sminus_i + Sminus_i * splus_i));
            Kondo.add_Ham(std::complex<double>(J_Kondo,0.0) * (Sz_i * sz_i));

            // magnetic field
            Kondo.add_Ham(std::complex<double>(-H,0.0) * Sz_i);

            // total electron operator
            Nelec_total += (n_up_i + n_dn_i);
            Nelec_up    += n_up_i;
            Nelec_dn    += n_dn_i;

            // total Sz operator
            Sz_total += (Sz_i + sz_i);
            Mz_total += Sz_i;
            mz_total += sz_i;
        }
    }


    // to use translational symmetry, we first fill the Weisse tables
    Kondo.fill_Weisse_table();

    std::ofstream fout("results.dat", std::ios::out | std::ios::app);
    fout << std::setprecision(10);
    fout << "#(1)" << "\t" << "(2)"  << "\t" << "(3)"  << "\t" << "(4)"  << "\t" << "(5)"  << "\t"
         << "(6)"  << "\t" << "(7)"  << "\t" << "(8)"  << "\t" << "(9)"  << "\t" << "(10)" << "\t"
         << "(11)" << "\t" << "(12)" << "\t" << "(13)" << "\t" << "(14)" << std::endl;
    fout << "Lx" << "\t" << "Ly" << "\t" << "Nelec" << "\t" << "Mz+mz" << "\t" << "J_Kondo" << "\t"
         << "H" << "\t" << "kx" << "\t" << "ky" << "\t" << "level" << "\t" << "E" << "\t"
         << "Nelec_up" << "\t" << "Nelec_dn" << "\t" << "Mz" << "\t" << "mz" << std::endl;

    for (int m = 0; m < Lx; m++) {
        for (int n = 0; n < Ly; n++) {
            double kx = (2.0*PI*m) / Lx;
            double ky = (2.0*PI*n) / Ly;
            // constructing the Hilbert space basis
            //Kondo.enumerate_basis_repr({m,n}, {Nelec_total,Sz_total}, {Nelec_total_val,Sz_total_val});
            Kondo.enumerate_basis_repr({m,n}, {Nelec_total}, {Nelec_total_val});

            // generating matrix of the Hamiltonian in the subspace
            Kondo.generate_Ham_sparse_repr();
            std::cout << std::endl;

            // obtaining the lowest eigenvals of the matrix
            Kondo.locate_E0_iram(qbasis::which_sym::repr, 20, 30);
            std::cout << std::endl;

            // note: eigenvals above +100 are typically faked numbers used in my code, should discard
            for (int l = 0; l < Kondo.nconv && Kondo.eigenvals_repr[l] < 100.0; l++){
                auto res_Nelec_up = Kondo.measure_repr_static(Nelec_up, 0, l);
                auto res_Nelec_dn = Kondo.measure_repr_static(Nelec_dn, 0, l);
                auto res_Mz_total = Kondo.measure_repr_static(Mz_total, 0, l);
                auto res_mz_total = Kondo.measure_repr_static(mz_total, 0, l);

                fout << Lx << "\t" << Ly << "\t"
                     << Nelec_total_val << "\t" << std::real(res_Mz_total+res_mz_total) << "\t"
                     << J_Kondo << "\t" << H << "\t" << kx << "\t" << ky << "\t"
                     << l << "\t" << Kondo.eigenvals_repr[l] << "\t"
                     << std::real(res_Nelec_up) << "\t" << std::real(res_Nelec_dn) << "\t"
                     << std::real(res_Mz_total) << "\t" << std::real(res_mz_total) << std::endl;
            }
        }
    }
    fout.close();

    return 0;
}
