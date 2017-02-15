#include <iostream>
#include <iomanip>
#include <fstream>
#include "qbasis.h"


// H = J/2 \sum_{<i,j>} (S_i^+ S_j^- + S_i^- S_j^+) + \Delta J \sum_{<i,j>} (S_i^z S_j^z)
//
// S_i^+ = ( 0  1 )     S_i^- = ( 0  0 )      S_i^z = (0.5   0 )
//         ( 0  0 )             ( 1  0 )              (0  -0.5 )


void run(const MKL_INT &Lx, const MKL_INT &pbc, const double &Delta, const double &h, const MKL_INT &step);

int main() {
    MKL_INT Lx;
    std::cout << "Lx: " << std::endl;
    std::cin >> Lx;
    std::cout << "Lx = " << Lx << std::endl;
    double Delta;
    std::cout << "Delta: " << std::endl;
    std::cin >> Delta;
    std::cout << "Delta = " << Delta << std::endl;
    double h;
    std::cout << "h: " <<std::endl;
    std::cin >> h;
    std::cout << "h = " << h << std::endl;
    MKL_INT step = 150;
    
    run(Lx, 1, Delta, h, step);
    
}

qbasis::mopr<std::complex<double>> Sx_Q(const MKL_INT &Lx, const double &Q)
{
    std::vector<std::vector<std::complex<double>>> Sx(2,std::vector<std::complex<double>>(2));
    Sx[0][0] = 0.0;
    Sx[0][1] = 0.5;
    Sx[1][0] = 0.5;
    Sx[1][1] = 0.0;
    qbasis::mopr<std::complex<double>> res;
    for (MKL_INT j = 0; j < Lx; j++) {
        qbasis::opr<std::complex<double>> sjx(j,0,false,Sx);
        res += ( (std::exp(std::complex<double>(0.0, -Q*j)) / std::sqrt(static_cast<double>(Lx)) ) * sjx);
    }
    return res;
}

qbasis::mopr<std::complex<double>> Sy_Q(const MKL_INT &Lx, const double &Q)
{
    std::vector<std::vector<std::complex<double>>> Sy(2,std::vector<std::complex<double>>(2));
    Sy[0][0] = 0.0;
    Sy[0][1] = std::complex<double>(0.0, -0.5);
    Sy[1][0] = std::complex<double>(0.0, 0.5);
    Sy[1][1] = 0.0;
    qbasis::mopr<std::complex<double>> res;
    for (MKL_INT j = 0; j < Lx; j++) {
        qbasis::opr<std::complex<double>> sjy(j,0,false,Sy);
        res += ( (std::exp(std::complex<double>(0.0, -Q*j)) / std::sqrt(static_cast<double>(Lx)) ) * sjy);
    }
    return res;
}

qbasis::mopr<std::complex<double>> Sz_Q(const MKL_INT &Lx, const double &Q)
{
    std::vector<std::vector<std::complex<double>>> Sz(2,std::vector<std::complex<double>>(2));
    Sz[0][0] = 0.5;
    Sz[0][1] = 0.0;
    Sz[1][0] = 0.0;
    Sz[1][1] = -0.5;
    qbasis::mopr<std::complex<double>> res;
    for (MKL_INT j = 0; j < Lx; j++) {
        qbasis::opr<std::complex<double>> sjz(j,0,false,Sz);
        res += ( (std::exp(std::complex<double>(0.0, -Q*j)) / std::sqrt(static_cast<double>(Lx)) ) * sjz);
    }
    return res;
}


void run(const MKL_INT &Lx, const MKL_INT &pbc, const double &Delta, const double &h, const MKL_INT &step) {
    // define the basic operators: S+, S-, Sz
    std::vector<std::vector<std::complex<double>>> Splus(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sminus(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sx(2,std::vector<std::complex<double>>(2));
    std::vector<std::complex<double>> Sz(2);
    Splus[0][0] = 0.0;
    Splus[0][1] = 1.0;
    Splus[1][0] = 0.0;
    Splus[1][1] = 0.0;
    Sminus[0][0] = 0.0;
    Sminus[0][1] = 0.0;
    Sminus[1][0] = 1.0;
    Sminus[1][1] = 0.0;
    Sx[0][0] = 0.0;
    Sx[0][1] = 0.5;
    Sx[1][0] = 0.5;
    Sx[1][1] = 0.0;
    Sz[0] = 0.5;
    Sz[1] = -0.5;
    
    // create Hamiltonian for xxz model on a chain
    qbasis::model<std::complex<double>> xxz;
    for (MKL_INT i = 0; i < Lx; i++) {
        qbasis::opr<std::complex<double>> six(i,0,false,Sx);
        if (i % 2 == 0) {
            xxz.add_offdiagonal_Ham(std::complex<double>(-h, 0.0) * six);
        } else {
            xxz.add_offdiagonal_Ham(std::complex<double>( h, 0.0) * six);
        }
        if (pbc == 0 && i == Lx-1) break;
        MKL_INT j = (i < Lx-1 ? i+1 : 0);
        qbasis::opr<std::complex<double>> sip(i,0,false,Splus);
        qbasis::opr<std::complex<double>> sim(i,0,false,Sminus);
        qbasis::opr<std::complex<double>> sjp(j,0,false,Splus);
        qbasis::opr<std::complex<double>> sjm(j,0,false,Sminus);
        qbasis::opr<std::complex<double>> siz(i,0,false,Sz);
        qbasis::opr<std::complex<double>> sjz(j,0,false,Sz);
        xxz.add_offdiagonal_Ham(std::complex<double>(0.5,0.0) * (sip * sjm + sim * sjp));
        xxz.add_diagonal_Ham(std::complex<double>(Delta,0.0) * (siz * sjz));
    }
    
    // create the basis, without any symmetry, this part later can be parallelized in the library
    xxz.dim_all = qbasis::int_pow(2, Lx);
    std::cout << "dim_all = " << xxz.dim_all << std::endl;
    xxz.basis_all = std::vector<qbasis::mbasis_elem>(xxz.dim_all,qbasis::mbasis_elem(Lx,{"spin-1/2"}));
    xxz.enumerate_basis_all();
    std::cout << std::endl;
    
    // sort basis
    xxz.sort_basis_all();
    std::cout << std::endl;
    
    // generating Hamiltonian matrix, using only upper triangle
    xxz.generate_Ham_all_sparse(true);
    std::cout << std::endl;
    
    
    // run Lanczos to obtain eigenvals
    xxz.locate_Emax();
    std::cout << std::endl;
    
    xxz.locate_E0();
    assert(xxz.energy_gap() > qbasis::lanczos_precision); // first only consider when the system is gapped (at least finite size gap)
    std::cout << std::endl;
    

    std::string output_name =   "L"+std::to_string(Lx)
                              + "_Delta_"+std::to_string(Delta).substr(0,4)
                              + "_h_"+std::to_string(h).substr(0,4)
                              + ".dat";
    std::ofstream fout(output_name, std::ios::out);
    fout << std::scientific << std::right;
    fout << "L\t" << Lx << std::endl;
    fout << "Delta\t" << Delta << std::endl;
    fout << "h\t" << h << std::endl;
    fout << "Emax\t" << xxz.energy_max() << std::endl;
    fout << "E0\t" << xxz.energy_min() << std::endl;
    fout << "Gap\t" << xxz.energy_gap() << std::endl << std::endl;
    
    
    //std::vector<std::complex<double>> v0(xxz.dim_all, 1.0);
    
    for (MKL_INT m = 0; m <= Lx; m++) {
        double Q = 2.0 * qbasis::pi * m / static_cast<double>(Lx);
        auto SxQ = Sx_Q(Lx, Q);
        auto SyQ = Sy_Q(Lx, Q);
        auto SzQ = Sz_Q(Lx, Q);
        
        // obtain the new starting vector |phi_0_x> = S^x(-q) |varphi_0>
        std::cout << "Generating S^x(-q) | varphi_0 >, with q = " << Q << std::endl;
        std::vector<std::complex<double>> phi0_x(xxz.dim_all, 0.0);
        xxz.moprXeigenvec(SxQ, phi0_x.data());
        double phi0_x_nrm2 = qbasis::nrm2(xxz.dim_all, phi0_x.data(), 1);
        double phi0_x_nrm2_copy;
        if (phi0_x_nrm2 > qbasis::lanczos_precision) {
            std::cout << "phi_x_nrm2 = " << phi0_x_nrm2 << std::endl;
            qbasis::scal(xxz.dim_all, 1.0/phi0_x_nrm2, phi0_x.data(), 1);
            phi0_x_nrm2_copy = qbasis::nrm2(xxz.dim_all, phi0_x.data(), 1);
        }
        
        std::cout << "Generating S^y(-q) | varphi_0 >, with q = " << Q << std::endl;
        std::vector<std::complex<double>> phi0_y(xxz.dim_all, 0.0);
        xxz.moprXeigenvec(SyQ, phi0_y.data());
        double phi0_y_nrm2 = qbasis::nrm2(xxz.dim_all, phi0_y.data(), 1);
        double phi0_y_nrm2_copy;
        if (phi0_y_nrm2 > qbasis::lanczos_precision) {
            std::cout << "phi_y_nrm2 = " << phi0_y_nrm2 << std::endl;
            qbasis::scal(xxz.dim_all, 1.0/phi0_y_nrm2, phi0_y.data(), 1);
            phi0_y_nrm2_copy = qbasis::nrm2(xxz.dim_all, phi0_y.data(), 1);
        }
        
        std::cout << "Generating S^z(-q) | varphi_0 >, with q = " << Q << std::endl;
        std::vector<std::complex<double>> phi0_z(xxz.dim_all, 0.0);
        xxz.moprXeigenvec(SzQ, phi0_z.data());
        double phi0_z_nrm2 = qbasis::nrm2(xxz.dim_all, phi0_z.data(), 1);
        double phi0_z_nrm2_copy;
        if (phi0_z_nrm2 > qbasis::lanczos_precision) {
            std::cout << "phi_z_nrm2 = " << phi0_z_nrm2 << std::endl;
            qbasis::scal(xxz.dim_all, 1.0/phi0_z_nrm2, phi0_z.data(), 1);
            phi0_z_nrm2_copy = qbasis::nrm2(xxz.dim_all, phi0_z.data(), 1);
        }
        
        std::cout << "Running continued fraction with " << step << " steps" << std::endl;
        std::vector<std::complex<double>> v(xxz.dim_all * 3);
        std::vector<double> hessenberg_x(step * 2, 0.0);
        std::vector<double> hessenberg_y(step * 2, 0.0);
        std::vector<double> hessenberg_z(step * 2, 0.0);
        fout << "Q\t" << Q << std::endl;
        fout << "nrm2_x\t" << phi0_x_nrm2 << std::endl;
        if (phi0_x_nrm2 > qbasis::lanczos_precision) {
            qbasis::lanczos(0, step, xxz.HamMat_csr, phi0_x_nrm2_copy, phi0_x.data(), v.data(), hessenberg_x.data(), step, false);
            fout << "b_x\t";
            for (MKL_INT i = 0; i < step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_x[i];
            }
            fout << std::endl;
            fout << "a_x\t";
            for (MKL_INT i = step; i < 2 * step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_x[i];
            }
            fout << std::endl;
        } else {
            fout << "b_x\t" << std::endl;
            fout << "a_x\t" << std::endl;
        }
        fout << "nrm2_y\t" << phi0_y_nrm2 << std::endl;
        if (phi0_y_nrm2 > qbasis::lanczos_precision) {
            qbasis::lanczos(0, step, xxz.HamMat_csr, phi0_y_nrm2_copy, phi0_y.data(), v.data(), hessenberg_y.data(), step, false);
            fout << "b_y\t";
            for (MKL_INT i = 0; i < step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_y[i];
            }
            fout << std::endl;
            fout << "a_y\t";
            for (MKL_INT i = step; i < 2 * step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_y[i];
            }
            fout << std::endl;
        } else {
            fout << "b_y\t" << std::endl;
            fout << "a_y\t" << std::endl;
        }
        fout << "nrm2_z\t" << phi0_z_nrm2 << std::endl;
        if (phi0_z_nrm2 > qbasis::lanczos_precision) {
            qbasis::lanczos(0, step, xxz.HamMat_csr, phi0_z_nrm2_copy, phi0_z.data(), v.data(), hessenberg_z.data(), step, false);
            fout << "b_z\t";
            for (MKL_INT i = 0; i < step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_z[i];
            }
            fout << std::endl;
            fout << "a_z\t";
            for (MKL_INT i = step; i < 2 * step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_z[i];
            }
            fout << std::endl;
        } else {
            fout << "b_z\t" << std::endl;
            fout << "a_z\t" << std::endl;
        }
        fout << std::endl;
        std::cout << std::endl;
    }
    fout.close();
    std::cout << "------" << std::endl;
    
}
