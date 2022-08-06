#include <iostream>
#include <iomanip>
#include "qbasis.h"

int test_chain_Heisenberg_spin_half();
int test_chain_tJ();

int main() {
    test_chain_Heisenberg_spin_half();

    test_chain_tJ();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}

int test_chain_Heisenberg_spin_half() {
    qbasis::initialize();
    std::cout << std::setprecision(10);
    // parameters
    double J = 1.0;
    int L = 16;

    std::cout << "L =       " << L << std::endl;
    std::cout << "J =       " << J << std::endl << std::endl;

    // lattice object
    std::vector<std::string> bc{"pbc"};
    qbasis::lattice lattice("chain",{static_cast<uint32_t>(L)},bc);

    // local matrix representation
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
    qbasis::model<std::complex<double>> Heisenberg(lattice);
    Heisenberg.add_orbital(lattice.total_sites(), "spin-1/2");
    for (int x = 0; x < L; x++) {
        uint32_t site_i, site_j;
        std::vector<int> work(lattice.dimension());
        lattice.coor2site({x}, 0, site_i, work); // obtain site label of (x)
        // construct operators on each site
        // spin
        auto Splus_i   = qbasis::opr<std::complex<double>>(site_i,0,false,Splus);
        auto Sminus_i  = qbasis::opr<std::complex<double>>(site_i,0,false,Sminus);
        auto Sz_i      = qbasis::opr<std::complex<double>>(site_i,0,false,Sz);

        // with neighbor (x+1)
        if (bc[0] == "pbc" || (bc[0] == "obc" && x < L - 1)) {
            lattice.coor2site({x+1}, 0, site_j, work);
            // spin exchanges
            auto Splus_j   = qbasis::opr<std::complex<double>>(site_j,0,false,Splus);
            auto Sminus_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Sminus);
            auto Sz_j      = qbasis::opr<std::complex<double>>(site_j,0,false,Sz);
            Heisenberg.add_Ham(std::complex<double>(0.5 * J,0.0) * (Splus_i * Sminus_j + Sminus_i * Splus_j));
            Heisenberg.add_Ham(std::complex<double>(J,0.0) * (Sz_i * Sz_j));
        }
    }

    // constructing the Hilbert space basis
    Heisenberg.enumerate_basis_full({}, {});

    // optional, will use more memory and give higher speed
    // generating matrix of the Hamiltonian in the full Hilbert space
    Heisenberg.generate_Ham_sparse_full();
    std::cout << std::endl;


    // obtaining the eigenvals of the matrix
    Heisenberg.locate_E0_lanczos(0);
    std::cout << std::endl;


    // for the parameters considered, we should obtain:
    assert(std::abs(Heisenberg.eigenvals_full[0] + 7.142296361) < 1e-8);


    // measure operators
    qbasis::mopr<std::complex<double>> Sz0Sz1(qbasis::opr<std::complex<double>>(0,0,false,Sz) *
        qbasis::opr<std::complex<double>>(1,0,false,Sz));
    qbasis::mopr<std::complex<double>> Sz0Sz2(qbasis::opr<std::complex<double>>(0,0,false,Sz) *
        qbasis::opr<std::complex<double>>(2,0,false,Sz));
    qbasis::mopr<std::complex<double>> Sp0Sm1(qbasis::opr<std::complex<double>>(0,0,false,Splus) *
        qbasis::opr<std::complex<double>>(1,0,false,Sminus));

    auto m1 = Heisenberg.measure_full_static(Sz0Sz1, 0, 0);
    auto m2 = Heisenberg.measure_full_static(Sz0Sz2, 0, 0);
    auto m3 = Heisenberg.measure_full_static(Sp0Sm1, 0, 0);
    std::cout << "Sz0Sz1 = " << m1 << std::endl;
    std::cout << "Sz0Sz2 = " << m2 << std::endl;
    std::cout << "Sp0Sm1 = " << m3 << std::endl << std::endl;

    assert(std::abs(m1 + 0.1487978408) < 1e-8);
    assert(std::abs(m2 - 0.0617414604) < 1e-8);
    assert(std::abs(m3 + 0.2975956817) < 1e-8);

    return 0;
}

int test_chain_tJ() {
    qbasis::initialize();
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
    std::cout << std::endl << std::endl;


    // for the parameters considered, we should obtain:
    assert(std::abs(tJ.eigenvals_full[0] + 9.762087307) < 1e-8);
    assert(std::abs(tJ.eigenvals_full[1] + 9.762087307) < 1e-8);

    return 0;
}
