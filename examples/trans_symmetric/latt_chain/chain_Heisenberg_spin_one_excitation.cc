#include <fstream>
#include <iostream>
#include <iomanip>
#include "qbasis.h"

#define PI 3.1415926535897932

qbasis::mopr<std::complex<double>> Sm_mQ(std::vector<std::vector<std::complex<double>>>&, qbasis::lattice&,
                                         const int &L, const double &Q);

// Heisenberg model on a chain
int main() {
    qbasis::initialize();
    std::cout << std::setprecision(10);
    // parameters
    double J = 1.0;
    int L = 12;

    int Q0 = 0;
    double Sz_total_val = 0.0;

    std::cout << "L =        " << L << std::endl;
    std::cout << "J =        " << J << std::endl << std::endl;
    std::cout << "Sz_total = " << Sz_total_val << std::endl;

    // lattice object
    std::vector<std::string> bc{"pbc"};
    qbasis::lattice lattice("chain",{static_cast<uint32_t>(L)},bc);

    // local matrix representation
    // Spins:
    std::vector<std::vector<std::complex<double>>> Splus(3,std::vector<std::complex<double>>(3));
    std::vector<std::vector<std::complex<double>>> Sminus(3,std::vector<std::complex<double>>(3));
    std::vector<std::complex<double>> Sz(3);

    Splus[0][0] = 0.0;
    Splus[0][1] = std::sqrt(2.0);
    Splus[0][2] = 0.0;
    Splus[1][0] = 0.0;
    Splus[1][1] = 0.0;
    Splus[1][2] = std::sqrt(2.0);
    Splus[2][0] = 0.0;
    Splus[2][1] = 0.0;
    Splus[2][2] = 0.0;

    Sminus[0][0] = 0.0;
    Sminus[0][1] = 0.0;
    Sminus[0][2] = 0.0;
    Sminus[1][0] = std::sqrt(2.0);
    Sminus[1][1] = 0.0;
    Sminus[1][2] = 0.0;
    Sminus[2][0] = 0.0;
    Sminus[2][1] = std::sqrt(2.0);
    Sminus[2][2] = 0.0;

    Sz[0]     = 1.0;
    Sz[1]     = 0.0;
    Sz[2]     = -1.0;

    // constructing the Hamiltonian in operator representation
    qbasis::model<std::complex<double>> Heisenberg(lattice);
    Heisenberg.add_orbital(lattice.total_sites(), "spin-1");
    qbasis::mopr<std::complex<double>> Sz_total;
    for (int x = 0; x < L; x++) {
        uint32_t site_i, site_j;
        std::vector<int> work(lattice.dimension());
        lattice.coor2site({x}, 0, site_i, work); // obtain site label of (x)
        // construct operators on each site
        // spin
        auto Splus_i  = qbasis::opr<std::complex<double>>(site_i,0,false,Splus);
        auto Sminus_i = qbasis::opr<std::complex<double>>(site_i,0,false,Sminus);
        auto Sz_i     = qbasis::opr<std::complex<double>>(site_i,0,false,Sz);

        // with neighbor (x+1)
        if (bc[0] == "pbc" || (bc[0] == "obc" && x < L - 1)) {
            lattice.coor2site({x+1}, 0, site_j, work);
            // spin exchanges
            auto Splus_j  = qbasis::opr<std::complex<double>>(site_j,0,false,Splus);
            auto Sminus_j = qbasis::opr<std::complex<double>>(site_j,0,false,Sminus);
            auto Sz_j     = qbasis::opr<std::complex<double>>(site_j,0,false,Sz);
            Heisenberg.add_Ham(std::complex<double>(0.5 * J,0.0) * (Splus_i * Sminus_j + Sminus_i * Splus_j));
            Heisenberg.add_Ham(std::complex<double>(J,0.0) * (Sz_i * Sz_j));
        }
        Sz_total += Sz_i;
    }

    // to use translational symmetry, we first fill the Weisse tables
    Heisenberg.fill_Weisse_table();

    // constructing the Hilbert space basis
    Heisenberg.enumerate_basis_repr({Q0}, {Sz_total}, {Sz_total_val});

    // generating matrix of the Hamiltonian in the subspace
    Heisenberg.generate_Ham_sparse_repr();
    std::cout << std::endl;

    // obtaining the eigenvals of the matrix
    Heisenberg.locate_E0_lanczos(1,1,1);
    std::cout << std::endl;

    double E_ground = Heisenberg.energy_min();
    Heisenberg.HamMat_csr_repr[0].destroy();

    std::string output_name_pm = "L"+std::to_string(L) + "_pm.dat";
    std::ofstream fout_dnmcs_pm(output_name_pm, std::ios::out);
    fout_dnmcs_pm << std::scientific << std::right;
    fout_dnmcs_pm << "L\t" << L << std::endl;
    fout_dnmcs_pm << "Sz\t" << Sz_total_val << std::endl;
    fout_dnmcs_pm << "E0\t" << E_ground << std::endl << std::endl;

    // measure dynamical structure factor
    const MKL_INT maxit=1000;
    double Hessenberg[2*maxit];
    MKL_INT m;
    double norm;
    for (int x = 0; x <= L/2; x++) {
        double Q = 2.0 * PI * x / static_cast<double>(L);
        std::cout << "Q = " << Q << std::endl;

        std::cout << "----------- S+S- -----------" << std::endl;
        auto SmmQ = Sm_mQ(Sminus, lattice, L, Q);
        Heisenberg.enumerate_basis_repr({Q0 - x}, {Sz_total}, {Sz_total_val-1.0}, 1); // for S-

        Heisenberg.generate_Ham_sparse_repr(1);
        Heisenberg.switch_sec_mat(1);

        Heisenberg.measure_repr_dynamic(SmmQ, 0, 1, maxit, m, norm, Hessenberg);
        Heisenberg.HamMat_csr_repr[1].destroy();

        fout_dnmcs_pm << "Q\t" << Q << std::endl;
        fout_dnmcs_pm << "nrm2\t" << norm << std::endl;
        if (norm > qbasis::lanczos_precision) {
            fout_dnmcs_pm << "b\t";
            for (MKL_INT i = 0; i < m; i++) {
                fout_dnmcs_pm << std::setw(30) << std::setprecision(18) << Hessenberg[i];
            }
            fout_dnmcs_pm << std::endl;
            fout_dnmcs_pm << "a\t";
            for (MKL_INT i = maxit; i < maxit + m; i++) {
                fout_dnmcs_pm << std::setw(30) << std::setprecision(18) << Hessenberg[i];
            }
            fout_dnmcs_pm << std::endl;
        } else {
            fout_dnmcs_pm << "b\t" << std::endl;
            fout_dnmcs_pm << "a\t" << std::endl;
        }
    }
    fout_dnmcs_pm.close();
}


qbasis::mopr<std::complex<double>> Sm_mQ(std::vector<std::vector<std::complex<double>>> &Sminus,
                                         qbasis::lattice &latt,
                                         const int &L, const double &Q)
{
    qbasis::mopr<std::complex<double>> res;
    for (int x = 0; x < L; x++) {
        uint32_t site_i;
        std::vector<int> work(1);
        latt.coor2site({x},   0, site_i, work);
        auto Sm_i = qbasis::opr<std::complex<double>>(site_i,0,false,Sminus);
        res += ( (std::exp(std::complex<double>(0.0, -Q*x)) / std::sqrt(static_cast<double>(L)) ) * Sm_i);
    }
    return res;
}
